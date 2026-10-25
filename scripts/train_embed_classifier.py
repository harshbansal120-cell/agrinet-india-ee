"""Train the land-cover head on SPEAR MAE embeddings and compare with a raw-band baseline (same rows).
Usage: python scripts/train_embed_classifier.py /path/to/PlanetScope [n_train] [n_test]"""
import sys, joblib, time, numpy as np, pandas as pd
sys.path.insert(0, "."); from app.spear_encoder import PlanetEncoder
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.metrics import f1_score, accuracy_score, classification_report
src = sys.argv[1]; ntr = int(sys.argv[2]) if len(sys.argv) > 2 else 300000; nte = int(sys.argv[3]) if len(sys.argv) > 3 else 150000
B = [f"B{i}" for i in range(1, 9)]; sc = joblib.load("models/planet_scaler_32.joblib"); enc = PlanetEncoder("models/planet_mae_ckpt_32.pt")
def load(split, n):
    d = pd.read_parquet(f"{src}/merged_balanced_{split}_planet.parquet", columns=B + ["classification"]).replace([np.inf, -np.inf], np.nan).dropna()
    d = d.sample(n, random_state=0); xs = sc.transform(d[B].values.astype(np.float32)).astype(np.float32)
    t = time.time(); e = enc.embed(xs); print(split, "embedded", e.shape, round(time.time() - t), "s", flush=True)
    return xs, e, d.classification.values
xtr, etr, ytr = load("train", ntr); xte, ete, yte = load("test", nte)
res = {}
for name, (a, b) in {"raw bands": (xtr, xte), "MAE embedding": (etr, ete), "embedding + bands": (np.c_[etr, xtr], np.c_[ete, xte])}.items():
    m = HGB(max_iter=150, random_state=0).fit(a, ytr); p = m.predict(b)
    res[name] = m; print(f"{name:18s} acc={accuracy_score(yte, p):.3f} macroF1={f1_score(yte, p, average='macro'):.3f}", flush=True)
joblib.dump(res["embedding + bands"], "data/classifier_emb.joblib")
print(classification_report(yte, res["embedding + bands"].predict(np.c_[ete, xte]), digits=3))
