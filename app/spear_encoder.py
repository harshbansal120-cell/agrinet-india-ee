"""Torch-free inference for the SPEAR PlanetScope SpectralMAE (embed 32, depth 6, heads 4).
Architecture ported from https://github.com/udaiveersingh/SPEAR (MIT), mae/planet_spectral_mae_model.py.
Loads the .pt checkpoint with a minimal unpickler (no PyTorch needed), so the deploy image stays small."""
import zipfile, pickle, collections, math
import numpy as np

LAMBDA = np.array([443, 490, 531, 565, 665, 705, 865, 940], np.float32)
FWHM = np.array([20, 50, 35, 35, 30, 15, 40, 50], np.float32)
D, HEADS, ENC_DEPTH, DEC_DEPTH, FF = 32, 4, 6, 3, 16

def load_state(path):
    z = zipfile.ZipFile(path)
    pk = [n for n in z.namelist() if n.endswith("data.pkl")][0]; root = pk[: -len("data.pkl")]
    class Storage:
        def __init__(s, key): s.key = key
    def rebuild(st, offset, size, stride, *a):
        raw = np.frombuffer(z.read(f"{root}data/{st.key}"), dtype=np.float32)
        n = int(np.prod(size)) if len(size) else 1
        return raw[offset: offset + n].reshape(size).copy()
    class U(pickle.Unpickler):
        def find_class(s, mod, name):
            if name == "_rebuild_tensor_v2": return rebuild
            if mod == "collections" and name == "OrderedDict": return collections.OrderedDict
            if name.endswith("Storage"): return name
            return lambda *a, **k: None
        def persistent_load(s, pid): return Storage(pid[2])
    return dict(U(z.open(pk)).load())

def _ln(x, w, b, eps=1e-5):
    m = x.mean(-1, keepdims=True); v = x.var(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b

def _gelu(x):
    from math import sqrt
    from scipy.special import erf
    return 0.5 * x * (1 + erf(x / sqrt(2)))

def _mha(q_in, kv_in, W, b, Wo, bo):
    B, Lq, _ = q_in.shape; Lk = kv_in.shape[1]; hd = D // HEADS
    q = q_in @ W[:D].T + b[:D]; k = kv_in @ W[D:2 * D].T + b[D:2 * D]; v = kv_in @ W[2 * D:].T + b[2 * D:]
    q = q.reshape(B, Lq, HEADS, hd).transpose(0, 2, 1, 3); k = k.reshape(B, Lk, HEADS, hd).transpose(0, 2, 1, 3)
    v = v.reshape(B, Lk, HEADS, hd).transpose(0, 2, 1, 3)
    a = q @ k.transpose(0, 1, 3, 2) / math.sqrt(hd); a = np.exp(a - a.max(-1, keepdims=True)); a /= a.sum(-1, keepdims=True)
    return (a @ v).transpose(0, 2, 1, 3).reshape(B, Lq, D) @ Wo.T + bo

def _ffn(x, p, pre):
    return np.maximum(x @ p[pre + "linear1.weight"].T + p[pre + "linear1.bias"], 0) @ p[pre + "linear2.weight"].T + p[pre + "linear2.bias"]

def _meta(p, prefix, lam, fw):
    l = (lam - 400.0) / 2100.0; f = fw / 2100.0; k = np.arange(1, FF + 1, dtype=np.float32)
    ang = 2 * math.pi * l[..., None] * k
    feats = np.concatenate([l[..., None], f[..., None], np.sin(ang), np.cos(ang)], -1).astype(np.float32)
    h = _gelu(feats @ p[prefix + ".mlp.0.weight"].T + p[prefix + ".mlp.0.bias"])
    return h @ p[prefix + ".mlp.2.weight"].T + p[prefix + ".mlp.2.bias"]

class PlanetEncoder:
    def __init__(self, ckpt):
        s = load_state(ckpt)
        self.p = {k.removeprefix("encoder."): v for k, v in s.items() if k.startswith("encoder.")}
        self.dec = {k.removeprefix("decoder."): v for k, v in s.items() if k.startswith("decoder.")}

    def _encode(self, xs, keep):
        """xs (B,8) scaled; keep (B,K) visible band idx. Returns cls (B,D), visible outputs (B,K,D)."""
        p, B = self.p, xs.shape[0]
        tok = xs[..., None] * p["value_proj.weight"][:, 0] + p["value_proj.bias"] + _meta(p, "meta", np.tile(LAMBDA, (B, 1)), np.tile(FWHM, (B, 1)))
        vis = np.take_along_axis(tok, keep[..., None], 1)
        x = np.concatenate([np.broadcast_to(p["cls"], (B, 1, D)), vis], 1)
        for i in range(ENC_DEPTH):
            q = f"encoder.layers.{i}."
            h = _ln(x, p[q + "norm1.weight"], p[q + "norm1.bias"])
            x = x + _mha(h, h, p[q + "self_attn.in_proj_weight"], p[q + "self_attn.in_proj_bias"], p[q + "self_attn.out_proj.weight"], p[q + "self_attn.out_proj.bias"])
            x = x + _ffn(_ln(x, p[q + "norm2.weight"], p[q + "norm2.bias"]), p, q)
        return x[:, 0], x[:, 1:]

    def embed(self, xs, batch=8192):
        """CLS embeddings (N,32) from scaled 8-band spectra with no masking."""
        xs = np.asarray(xs, np.float32); keep = np.tile(np.arange(8), (min(batch, len(xs)), 1)); out = []
        for i in range(0, len(xs), batch):
            b = xs[i:i + batch]; out.append(self._encode(b, keep[: len(b)])[0])
        return np.concatenate(out)

    def reconstruct(self, xs, mask_ratio=0.6, seed=0):
        """Full MAE forward (encoder+decoder) — used only to verify the port against the paper's R2."""
        rng = np.random.default_rng(seed); B = len(xs); K = int(round((1 - mask_ratio) * 8))
        order = np.argsort(rng.random((B, 8)), 1); keep, msk = order[:, :K], order[:, K:]
        cls, vis = self._encode(xs, keep); d = self.dec; lam = np.tile(LAMBDA, (B, 1)); fw = np.tile(FWHM, (B, 1))
        mem = np.concatenate([cls[:, None], vis], 1) @ d["mem_proj.weight"].T + d["mem_proj.bias"]
        mv = _meta(d, "meta_m", np.take_along_axis(lam, keep, 1), np.take_along_axis(fw, keep, 1))
        mem = mem + np.concatenate([np.zeros((B, 1, D), np.float32), mv], 1)
        x = d["dec_mask_token"] + _meta(d, "meta_q", np.take_along_axis(lam, msk, 1), np.take_along_axis(fw, msk, 1))
        for i in range(DEC_DEPTH):
            q = f"decoder.layers.{i}."
            h = _ln(x, d[q + "norm1.weight"], d[q + "norm1.bias"])
            x = x + _mha(h, h, d[q + "self_attn.in_proj_weight"], d[q + "self_attn.in_proj_bias"], d[q + "self_attn.out_proj.weight"], d[q + "self_attn.out_proj.bias"])
            h = _ln(x, d[q + "norm2.weight"], d[q + "norm2.bias"])
            x = x + _mha(h, mem, d[q + "multihead_attn.in_proj_weight"], d[q + "multihead_attn.in_proj_bias"], d[q + "multihead_attn.out_proj.weight"], d[q + "multihead_attn.out_proj.bias"])
            x = x + _ffn(_ln(x, d[q + "norm3.weight"], d[q + "norm3.bias"]), d, q)
        pred = (x @ d["head.weight"].T + d["head.bias"])[..., 0]
        return pred, np.take_along_axis(xs, msk, 1), msk
