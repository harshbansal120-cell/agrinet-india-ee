"""Sanity-check the NumPy port: masked-band reconstruction R2 should be near the paper's ~0.958 for Planet."""
import sys, joblib, numpy as np, pandas as pd
sys.path.insert(0, "."); from app.spear_encoder import PlanetEncoder
src = sys.argv[1]; B = [f"B{i}" for i in range(1, 9)]
df = pd.read_parquet(f"{src}/merged_balanced_train_planet.parquet", columns=B).replace([np.inf, -np.inf], np.nan).dropna().sample(20000, random_state=1)
xs = joblib.load("models/planet_scaler_32.joblib").transform(df.values.astype(np.float32)).astype(np.float32)
enc = PlanetEncoder("models/planet_mae_ckpt_32.pt")
pred, tgt, _ = enc.reconstruct(xs)
print("masked-band R2:", round(1 - ((pred - tgt) ** 2).sum() / ((tgt - tgt.mean()) ** 2).sum(), 4))
print("baseline (predict mean) R2 = 0.0; embedding shape:", enc.embed(xs[:5]).shape)
