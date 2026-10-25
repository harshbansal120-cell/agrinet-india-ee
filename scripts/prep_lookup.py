"""Build data/lookup.parquet: one row per ~150 m geohash cell (mean bands, majority label, latest date).
Memory-light: aggregates each source file separately, then merges.
Usage: python scripts/prep_lookup.py path/to/PlanetScope"""
import sys, gc, pandas as pd
src = sys.argv[1] if len(sys.argv) > 1 else "."
B = [f"B{i}" for i in range(1, 9)]
parts = []
for s in ("train", "test"):
    df = pd.read_parquet(f"{src}/merged_balanced_{s}_planet.parquet",
                         columns=["Lat", "Lon", "classification", "geohash"] + B + ["date", "cloud_cover"])
    df = df[df.cloud_cover.fillna(0) < 0.3]
    df["date"] = df["date"].astype(str)
    for c in ["Lat", "Lon"] + B: df[c] = df[c].astype("float32")
    df["classification"] = df["classification"].astype("int8")
    # majority label per cell
    lab = df.groupby(["geohash", "classification"], observed=True).size().reset_index(name="c")
    lab = lab.sort_values("c").drop_duplicates("geohash", keep="last").set_index("geohash")
    g = df.groupby("geohash", observed=True)
    a = g[["Lat", "Lon"] + B].mean()
    a["label"], a["lab_c"] = lab["classification"], lab["c"]
    a["date"], a["n"] = g["date"].max(), g.size()
    parts.append(a); del df, g, lab; gc.collect()
al = pd.concat(parts).reset_index()
cols = ["Lat", "Lon"] + B
for c in cols: al[c] = al[c] * al["n"]
sums = al.groupby("geohash")[cols + ["n"]].sum()
out = sums[cols].div(sums["n"], axis=0)
top = al.sort_values("lab_c").drop_duplicates("geohash", keep="last").set_index("geohash")
out["label"] = top["label"].astype(int)
out["date"] = al.groupby("geohash")["date"].max()
out["n"] = sums["n"]
out.reset_index().to_parquet("data/lookup.parquet")
print("cells:", len(out))
