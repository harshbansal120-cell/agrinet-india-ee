"""Train a land-cover classifier on the 8 PlanetScope bands (+NDVI). Prints per-class F1 on the test set.
Usage: python scripts/train_classifier.py path/to/PlanetScope
NOTE: swap the features for your MAE encoder embeddings once you share the encoder class definition."""
import sys, joblib, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report
src = sys.argv[1] if len(sys.argv) > 1 else "."
B = [f"B{i}" for i in range(1,9)]
def feats(d):
    X = d[B].astype("float32").copy()
    X["ndvi"] = (d.B8 - d.B6) / (d.B8 + d.B6 + 1e-6)
    return X
tr = pd.read_parquet(f"{src}/merged_balanced_train_planet.parquet", columns=B+["classification"]).sample(800_000, random_state=0)
te = pd.read_parquet(f"{src}/merged_balanced_test_planet.parquet", columns=B+["classification"]).sample(500_000, random_state=0)
m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, random_state=0).fit(feats(tr), tr.classification)
print(classification_report(te.classification, m.predict(feats(te)), digits=3))
joblib.dump(m, "data/classifier.joblib")
