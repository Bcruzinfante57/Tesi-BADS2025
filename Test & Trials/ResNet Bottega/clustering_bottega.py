import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def safe_imports() -> bool:
    has_hdbscan = True
    try:
        import pandas as pd  # noqa: F401
        from sklearn.preprocessing import StandardScaler  # noqa: F401
        from sklearn.decomposition import PCA  # noqa: F401
        from sklearn.cluster import KMeans, AgglomerativeClustering  # noqa: F401
        from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score  # noqa: F401
        try:
            import hdbscan  # noqa: F401
        except ModuleNotFoundError:
            has_hdbscan = False
        import umap  # noqa: F401
        import matplotlib.pyplot as plt  # noqa: F401
    except ModuleNotFoundError as e:
        missing = str(e).split("No module named ")[-1].strip("'\"")
        print(
            f"Missing dependency: {missing}.\n"
            "Please install: pip install pandas scikit-learn umap-learn matplotlib",
        )
        raise
    return has_hdbscan


def load_embeddings(csv_path: Path):
    import pandas as pd

    df = pd.read_csv(csv_path)
    # Detect feature columns (all numeric except possibly 'path')
    feature_cols = [c for c in df.columns if c != "path" and np.issubdtype(df[c].dtype, np.number)]
    if not feature_cols:
        raise ValueError("No numeric feature columns found in embeddings CSV.")
    X = df[feature_cols].to_numpy(dtype=np.float32)
    paths = df["path"].astype(str).tolist() if "path" in df.columns else [str(i) for i in range(len(df))]
    return X, paths, feature_cols


def scale_and_reduce(X: np.ndarray, use_pca: bool, pca_dim: int, random_state: int) -> Tuple[np.ndarray, Dict]:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    meta = {}
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    meta["scaler"] = scaler
    if use_pca and Xs.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim, random_state=random_state)
        Xr = pca.fit_transform(Xs)
        meta["pca"] = pca
        return Xr, meta
    return Xs, meta


@dataclass
class RunResult:
    algo: str
    params: Dict
    labels: np.ndarray
    n_clusters: int
    silhouette: Optional[float]
    ch: Optional[float]
    db: Optional[float]
    noise_frac: float = 0.0


def compute_scores(labels: np.ndarray, X: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[float], float, int]:
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

    labels = np.asarray(labels)
    mask = labels >= 0
    n_clusters = len(set(labels[mask])) if mask.any() else 0
    noise_frac = float((labels < 0).sum()) / float(len(labels)) if len(labels) else 0.0

    if n_clusters < 2 or mask.sum() < 2:
        return None, None, None, noise_frac, n_clusters

    Xc = X[mask]
    lc = labels[mask]
    try:
        sil = float(silhouette_score(Xc, lc))
    except Exception:
        sil = None
    try:
        ch = float(calinski_harabasz_score(Xc, lc))
    except Exception:
        ch = None
    try:
        db = float(davies_bouldin_score(Xc, lc))
    except Exception:
        db = None
    return sil, ch, db, noise_frac, n_clusters


def try_kmeans(X: np.ndarray, k: int, random_state: int) -> RunResult:
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = km.fit_predict(X)
    sil, ch, db, noise_frac, n_clusters = compute_scores(labels, X)
    return RunResult("KMeans", {"k": k}, labels, n_clusters, sil, ch, db, noise_frac)


def try_agglomerative(X: np.ndarray, k: int) -> RunResult:
    from sklearn.cluster import AgglomerativeClustering

    agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = agg.fit_predict(X)
    sil, ch, db, noise_frac, n_clusters = compute_scores(labels, X)
    return RunResult("Agglomerative", {"k": k, "linkage": "ward"}, labels, n_clusters, sil, ch, db, noise_frac)


def try_hdbscan(X: np.ndarray, min_cluster_size: int, min_samples: Optional[int] = None) -> RunResult:
    import hdbscan

    hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = hdb.fit_predict(X)
    sil, ch, db, noise_frac, n_clusters = compute_scores(labels, X)
    return RunResult(
        "HDBSCAN",
        {"min_cluster_size": int(min_cluster_size), "min_samples": None if min_samples is None else int(min_samples)},
        labels,
        n_clusters,
        sil,
        ch,
        db,
        noise_frac,
    )


def pick_best(runs: List[RunResult]) -> RunResult:
    # Primary: highest silhouette; fallback: highest CH; tie-breaker: lowest DB
    valid = [r for r in runs if r.silhouette is not None]
    if not valid:
        valid = [r for r in runs if r.ch is not None]
    if not valid:
        valid = [r for r in runs if r.db is not None]
    if not valid:
        # pick the one with most clusters as last resort
        return max(runs, key=lambda r: r.n_clusters)

    def score_key(r: RunResult):
        sil = r.silhouette if r.silhouette is not None else -np.inf
        ch = r.ch if r.ch is not None else -np.inf
        db = r.db if r.db is not None else np.inf
        return (sil, ch, -db)

    return max(valid, key=score_key)


def run_umap_plot(X: np.ndarray, labels: np.ndarray, out_path: Path, title: str = ""):
    import umap
    import matplotlib.pyplot as plt

    reducer = umap.UMAP(n_components=2, random_state=42)
    emb2d = reducer.fit_transform(X)
    labs = np.asarray(labels)

    plt.figure(figsize=(8, 6), dpi=160)
    n_clusters = len(set(labs[labs >= 0]))
    unique = sorted(set(labs))
    # Build colors (reserve gray for noise)
    import matplotlib as mpl

    cmap = plt.get_cmap("tab20")
    colors = {}
    color_iter = 0
    for u in unique:
        if u == -1:
            colors[u] = (0.6, 0.6, 0.6, 0.6)
        else:
            colors[u] = cmap(color_iter % 20)
            color_iter += 1

    for u in unique:
        mask = labs == u
        plt.scatter(emb2d[mask, 0], emb2d[mask, 1], s=10, c=[colors[u]], label=f"{u}")

    ttl = title or f"UMAP clusters: {n_clusters}"
    plt.title(ttl)
    # Only show legend if clusters are few
    if n_clusters <= 15:
        plt.legend(title="Cluster", markerscale=2, fontsize=8)
    plt.tight_layout()
    out_path = out_path.with_suffix(".jpg")
    # Save JPEG without unsupported quality kwarg for broad backend compatibility
    plt.savefig(out_path, format="jpg")
    plt.close()
    return out_path


def main():
    has_hdbscan = safe_imports()
    import pandas as pd

    parser = argparse.ArgumentParser(description="Cluster embeddings with KMeans, HDBSCAN, Agglomerative and plot UMAP")
    parser.add_argument(
        "--embeddings-csv",
        type=str,
        default="bottega_resnet50_embeddings.csv",
        help="Embeddings CSV file (from filter_and_embed_bottega.py)",
    )
    parser.add_argument("--min-k", type=int, default=2, help="Minimum k for KMeans/Agglomerative")
    parser.add_argument("--max-k", type=int, default=20, help="Maximum k for KMeans/Agglomerative")
    parser.add_argument(
        "--hdbscan-min-cluster-size",
        type=int,
        nargs="*",
        default=[5, 10, 20, 30],
        help="HDBSCAN min_cluster_size values to try",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--use-pca", action="store_true", help="Apply PCA to 50 dims before clustering")
    parser.add_argument("--pca-dim", type=int, default=50)
    parser.add_argument("--umap-out", type=str, default="clusters_umap_best.jpg")
    parser.add_argument("--assignments-out", type=str, default="clusters_best.csv")
    parser.add_argument("--summary-out", type=str, default="clustering_bottega_summary.csv")

    args = parser.parse_args()

    X_raw, paths, feature_cols = load_embeddings(Path(args.embeddings_csv))
    X, tr_meta = scale_and_reduce(X_raw, args.use_pca, args.pca_dim, args.random_state)

    runs: List[RunResult] = []
    # KMeans and Agglomerative grid
    for k in range(args.min_k, args.max_k + 1):
        try:
            runs.append(try_kmeans(X, k, args.random_state))
        except Exception as e:
            print(f"KMeans(k={k}) failed: {e}")
        try:
            runs.append(try_agglomerative(X, k))
        except Exception as e:
            print(f"Agglomerative(k={k}) failed: {e}")

    # HDBSCAN settings
    if has_hdbscan:
        for mcs in args.hdbscan_min_cluster_size:
            try:
                runs.append(try_hdbscan(X, mcs))
            except Exception as e:
                print(f"HDBSCAN(min_cluster_size={mcs}) failed: {e}")
    else:
        print("hdbscan not installed; skipping HDBSCAN runs. Install with: pip install hdbscan")

    if not runs:
        raise RuntimeError("No clustering runs completed.")

    # Best per algorithm (no minimum cluster constraint)
    by_algo: Dict[str, List[RunResult]] = {}
    for r in runs:
        by_algo.setdefault(r.algo, []).append(r)

    best_per_algo: Dict[str, Optional[RunResult]] = {algo: pick_best(rs) if rs else None for algo, rs in by_algo.items()}
    # Overall best among all runs
    best = pick_best(runs)

    # Save summary
    rows = []
    for r in runs:
        rows.append(
            {
                "algo": r.algo,
                **{f"param_{k}": v for k, v in r.params.items()},
                "n_clusters": r.n_clusters,
                "silhouette": r.silhouette,
                "calinski_harabasz": r.ch,
                "davies_bouldin": r.db,
                "noise_frac": r.noise_frac,
            }
        )
    pd.DataFrame(rows).to_csv(args.summary_out, index=False)
    print(f"Wrote summary to {args.summary_out}")

    # Save best assignments
    out_assign = Path(args.assignments_out)
    pd.DataFrame({"path": paths, "cluster": best.labels}).to_csv(out_assign, index=False)
    print(f"Overall best: {best.algo} with params {best.params} -> {best.n_clusters} clusters")
    print(f"Wrote assignments to {out_assign}")

    # UMAP plot
    umap_path = run_umap_plot(X, best.labels, Path(args.umap_out), title=f"{best.algo} ({best.n_clusters} clusters)")
    print(f"Saved UMAP plot to {umap_path}")

    # Save best artifacts per algorithm too
    for algo, br in best_per_algo.items():
        if br is None:
            print(f"No runs completed for {algo}")
            continue
        stem = algo.lower()
        assign_path = Path(f"clusters_best_{stem}.csv")
        pd.DataFrame({"path": paths, "cluster": br.labels}).to_csv(assign_path, index=False)
        print(f"Best {algo}: params {br.params} -> {br.n_clusters} clusters; wrote {assign_path}")
        umap_algo = Path(f"clusters_umap_best_{stem}.jpg")
        run_umap_plot(X, br.labels, umap_algo, title=f"{algo} ({br.n_clusters} clusters)")


if __name__ == "__main__":
    main()
