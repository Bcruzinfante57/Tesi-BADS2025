# random_search_clustering.py
import os, glob, random, json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import AgglomerativeClustering, KMeans

import umap
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.spatial import ConvexHull

# ---------------------------
# Config
# ---------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED)

# EITHER point to a single CSV, or a directory of many CSVs:
SINGLE_FEATURES_CSV = "features_anteojos_forma_autoencoder.csv"
FEATURES_DIR = None  # e.g., "all_features_csvs" (set to a folder to compare many CSVs)

IMAGES_DIR = "imagenes_bottega"   # folder that contains your original images referenced in `imagen` column
N_TRIALS = 50

K_RANGE = list(range(2, 13))  # candidate cluster counts
USE_AGGLO = True
USE_KMEANS = True

# ---------------------------
# Helpers
# ---------------------------
def load_feature_tables() -> Dict[str, pd.DataFrame]:
    tables = {}
    if FEATURES_DIR and os.path.isdir(FEATURES_DIR):
        for path in glob.glob(os.path.join(FEATURES_DIR, "*.csv")):
            name = os.path.basename(path)
            df = pd.read_csv(path)
            assert "imagen" in df.columns, f"'imagen' column is missing in {name}"
            tables[name] = df
    else:
        df = pd.read_csv(SINGLE_FEATURES_CSV)
        assert "imagen" in df.columns, "'imagen' column is missing"
        tables[os.path.basename(SINGLE_FEATURES_CSV)] = df
    return tables

def maybe_pca(X: np.ndarray, pca_dim: int | None):
    if pca_dim is None or pca_dim >= X.shape[1]:
        return X, None
    p = PCA(n_components=pca_dim, random_state=SEED)
    return p.fit_transform(X), p

def composite_score(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    # Beware: some metrics require >1 clusters and that each cluster has >1 sample
    # We’ll guard against degeneracy.
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return {"sil": -1e9, "dbi": 1e9, "ch": -1e9, "score": -1e9}
    try:
        sil = silhouette_score(X, labels, metric='euclidean')
    except Exception:
        sil = -1e9
    try:
        dbi = davies_bouldin_score(X, labels)
    except Exception:
        dbi = 1e9
    try:
        ch = calinski_harabasz_score(X, labels)
    except Exception:
        ch = -1e9

    # Combine (higher better). We invert DBI.
    # Scale terms roughly; you can tweak weights for your dataset.
    score = (sil) + (0.002 * ch) + (1.0 / (1.0 + dbi))
    return {"sil": float(sil), "dbi": float(dbi), "ch": float(ch), "score": float(score)}

def draw_umap_thumbnail_plot(X2d: np.ndarray, labels: np.ndarray, names: list[str], out_png: str):
    k = len(np.unique(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, k)))
    fig, ax = plt.subplots(figsize=(16, 12))

    # Thumbnails
    for i, (x0, y0) in enumerate(X2d):
        img_path = os.path.join(IMAGES_DIR, names[i])
        try:
            img = Image.open(img_path)
            img.thumbnail((65, 65))
            imagebox = OffsetImage(img, zoom=1.0)
            ab = AnnotationBbox(imagebox, (x0, y0), frameon=False, zorder=3)
            ax.add_artist(ab)
        except Exception as e:
            print(f"⚠️ Could not load {names[i]}: {e}")

    # Cluster outlines (per label)
    for i, color in zip(sorted(np.unique(labels)), colors):
        pts = X2d[labels == i]
        if len(pts) >= 3 and np.ptp(pts[:, 0]) > 1e-5 and np.ptp(pts[:, 1]) > 1e-5:
            try:
                hull = ConvexHull(pts)
                for simplex in hull.simplices:
                    ax.plot(pts[simplex, 0], pts[simplex, 1], linestyle='dotted', color=color, linewidth=2)
            except Exception as e:
                print(f"⚠️ Hull error for cluster {i}: {e}")

    ax.set_title(f"Visual clusters (UMAP, k={k})", fontsize=16)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"📸 Saved {out_png}")

# ---------------------------
# Random search over clustering pipeline
# ---------------------------
def sample_trial_cfg(valid_pca_dims) -> Dict[str, Any]:
    cfg = {
        "standardize": random.choice([True, False]),
        "l2norm": random.choice([True, False]),
        "pca_dim": random.choice(valid_pca_dims),   # ← use filtered list
        "metric": random.choice(["euclidean", "cosine"]),
        "algo": random.choice([a for a, use in (("agglom", USE_AGGLO), ("kmeans", USE_KMEANS)) if use]),
        "k": random.choice(K_RANGE),
        "linkage": random.choice(["ward", "average", "complete"]),
        "kmeans_init": random.choice(["k-means++", "random"]),
        "kmeans_n_init": random.choice([10, 20, 50]),
    }
    if cfg["algo"] == "agglom" and cfg["linkage"] == "ward":
        cfg["metric"] = "euclidean"
    return cfg

 
def preprocess(X: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    Xp = X.copy()
    if cfg["standardize"]:
        Xp = StandardScaler().fit_transform(Xp)
    if cfg["l2norm"]:
        Xp = normalize(Xp, norm="l2")
    Xp, _ = maybe_pca(Xp, cfg["pca_dim"])
    # For silhouette/DBI/CH we’ll use euclidean on Xp. (Silhouette metric was set inside scorer.)
    # If cfg["metric"] == "cosine" and algo needs distances, we mostly let the algo handle it.
    return Xp

def cluster(Xp: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    k = cfg["k"]
    if cfg["algo"] == "agglom":
        linkage = cfg["linkage"]
        if linkage == "ward":
            model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        else:
            # sklearn 1.4+: use metric param, earlier used affinity (deprecated)
            model = AgglomerativeClustering(n_clusters=k, linkage=linkage, metric=cfg["metric"])
        labels = model.fit_predict(Xp)
    else:
        model = KMeans(n_clusters=k, init=cfg["kmeans_init"], n_init=cfg["kmeans_n_init"], random_state=SEED)
        labels = model.fit_predict(Xp)
    return labels

def run_search_on_table(name: str, df: pd.DataFrame):
    names = df["imagen"].tolist()
    X = df.drop(columns=["imagen"]).values.astype(np.float32)

    # Build feasible PCA dims for this X
    max_dim = max(0, min(X.shape[0] - 1, X.shape[1]))
    candidate_dims = [32, 64, 128, 256]
    valid_pca_dims = [None] + [d for d in candidate_dims if d <= max_dim]
    if len(valid_pca_dims) == 1:
        print(f"[{name}] PCA skipped (max feasible PCA dim is {max_dim}).")

    best = None
    trials = []
    for t in range(1, N_TRIALS + 1):
        cfg = sample_trial_cfg(valid_pca_dims)
        Xp = preprocess(X, cfg)
        try:
            labels = cluster(Xp, cfg)
            scores = composite_score(Xp, labels)
        except Exception as e:
            # Skip incompatible combos
            scores = {"sil": -1e9, "dbi": 1e9, "ch": -1e9, "score": -1e9}
        rec = {**cfg, **scores, "features_file": name, "trial": t}
        trials.append(rec)
        if (best is None) or (rec["score"] > best["score"]):
            best = rec
        print(f"[{name}] Trial {t:02d} | score={rec['score']:.4f} | sil={rec['sil']:.4f} | dbi={rec['dbi']:.3f} | ch={rec['ch']:.1f} | cfg={ {k:rec[k] for k in ['algo','k','linkage','pca_dim','standardize','l2norm','metric']} }")

    trials_df = pd.DataFrame(trials).sort_values("score", ascending=False)
    return best, trials_df

def main():
    tables = load_feature_tables()
    global_best = None
    all_trials = []

    for fname, df in tables.items():
        best, trials_df = run_search_on_table(fname, df)
        all_trials.append(trials_df)
        if (global_best is None) or (best["score"] > global_best["score"]):
            global_best = {"best": best, "df": df, "trials": trials_df}

    # Save search logs
    summary_df = pd.concat(all_trials, ignore_index=True)
    summary_df.to_csv("clustering_search_summary.csv", index=False)
    print("\n📝 Saved clustering_search_summary.csv")
    print("\n🏆 Best overall config:")
    print(json.dumps(global_best["best"], indent=2))

    # Apply best config and export labels
    best = global_best["best"]
    df = global_best["df"].copy()
    X = df.drop(columns=["imagen"]).values.astype(np.float32)
    Xp = preprocess(X, best)
    labels = cluster(Xp, best)
    df_out = df.copy()
    df_out["cluster"] = labels
    out_csv = "best_clusters_assignment.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"📄 Saved {out_csv}")

    # Make a UMAP plot for the best solution (for visualization only)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=SEED)
    X_umap = reducer.fit_transform(Xp)
    out_png = "clusters_umap_best.png"
    draw_umap_thumbnail_plot(X_umap, labels, df_out["imagen"].tolist(), out_png)

if __name__ == "__main__":
    main()
