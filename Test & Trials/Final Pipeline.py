import time, re, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import wandb
import wandb.proto.wandb_internal_pb2 as p
from typing import Dict, List, Tuple, Any

# Libraries for clustering and dimensionality reduction
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
from PIL import Image, ImageOps
from skimage.feature import local_binary_pattern
from skimage.filters import gabor

# Conditional imports
try:
    import torch
    import timm
    import torchvision.transforms as T 
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from skimage.feature import hog
    from skimage.color import rgb2gray
    from skimage.measure import regionprops, moments_central, moments_hu
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    
try:
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist # Required for dispersion calculation
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

import cv2

# --- 1. Filter and Data Configurations ---

BRANDS_TO_PROCESS = ["Bottega Veneta"] #["Bottega Veneta", "Dolce&Gabbana", "Fendi", "Prada", "YSL"]
MIN_PRICE, MAX_PRICE = 0, 1000 

# Options: "separation" or "dispersion" 
CLUSTER_ORDER_BY = "separation" 

BRAND_FOLDERS = { 
    "Bottega Veneta": "images_bottega", 
    "Dolce&Gabbana": "images_D&G", 
    "Fendi": "images_Fendi", 
    "Prada": "images_Prada", 
    "YSL": "images_YSL" 
} 
# ----------------------------------------------------


# IMAGEN PREPROCESSING #
_saved_first_crop = False
_saved_first_final = False

def crop_white_bg(pil_img, white_thresh=235, padding=0.15):
    """Crops white background using contours and adds padding to keep full glasses centered."""
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Mask for non-white regions
    mask = gray < white_thresh
    mask = mask.astype(np.uint8) * 255

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return pil_img  # fallback
    
    # Take largest contour (should be the glasses)
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)

    # Add proportional padding
    pad_x = int(w * padding)
    pad_y = int(h * padding)
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(img.shape[1], x + w + pad_x)
    y1 = min(img.shape[0], y + h + pad_y)

    cropped = img[y0:y1, x0:x1]
    return Image.fromarray(cropped)

def load_and_preprocess(p, target=(224, 224)):
    """Loads, crops, and centers an image for model input."""
    global _saved_first_crop, _saved_first_final

    img = Image.open(p).convert("RGB")
    img = crop_white_bg(img)

    # Save crop example (only once)
    if not _saved_first_crop:
        try:
            img.save("example_crop.jpg")
            print("✅ Saved 'example_crop.jpg' (raw crop example).")
        except:
             pass # Suppress save errors if folder is read-only
        _saved_first_crop = True


    # Create white background and center the object
    bg = Image.new("RGB", target, (255, 255, 255))
    img_contained = ImageOps.contain(img, target)

    paste_x = (target[0] - img_contained.width) // 2
    paste_y = (target[1] - img_contained.height) // 2
    bg.paste(img_contained, (paste_x, paste_y))

    # Save final centered example (only once)
    if not _saved_first_final:
        try:
            bg.save("example_final.jpg")
            print("✅ Saved 'example_final.jpg' (final centered image).")
        except:
             pass # Suppress save errors if folder is read-only
        _saved_first_final = True

    return bg

    
# --- Feature Extraction with ViT and classical methods ---

def vit_embedding(img_pil, model, device):
   # Extracts a feature vector using a pre-trained ViT model.
        
    if img_pil.mode != 'RGB':
        img_pil = img_pil.convert('RGB')
        
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    x = transform(img_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        feat = model.forward_features(x) if hasattr(model, "forward_features") else model(x)
        if feat.ndim == 3:
            feat = feat.mean(dim=1)
        return feat.cpu().numpy().squeeze()

def color_hist_features(img, bins=32):
    """Extracts color histogram features."""
    a = np.array(img).astype(float) / 255.0
    feats = []
    for c in range(3):
        h, _ = np.histogram(a[:,:,c], bins=bins, range=(0,1))
        feats.append(h.astype(float))
    feat = np.concatenate(feats)
    feat = feat / (np.linalg.norm(feat) + 1e-8)
    return feat

def hog_features(img, pixels_per_cell=(6, 6)):
    """Extracts Histogram of Oriented Gradients (HOG) features with higher precision."""
    if not HAS_SKIMAGE:
        return np.array([])
    gray = rgb2gray(np.array(img))
    return hog(gray, orientations=12, pixels_per_cell=pixels_per_cell, cells_per_block=(2,2), feature_vector=True)

def shape_properties_features(img):
    """
    Extracts geometric properties and Hu Moments from the image.
    Total: ~13 features
    """
    if not HAS_SKIMAGE:
        return np.zeros(13)
    
    gray = rgb2gray(np.array(img))
    binary = gray < gray.mean()
    props = regionprops(binary.astype(int))
    
    if not props:
        return np.zeros(13)
    
    prop = props[0]
    
    mu = moments_central(binary.astype(int))
    hu_moments = moments_hu(mu)
    
    area = prop.area
    perimeter = prop.perimeter
    circularity = (4 * np.pi * area) / (perimeter**2 + 1e-8)
    aspect_ratio = prop.major_axis_length / (prop.minor_axis_length + 1e-8)
    solidity = prop.solidity
    eccentricity = prop.eccentricity
    
    feats = np.concatenate([
        hu_moments,
        np.array([area, circularity, eccentricity, solidity, aspect_ratio])
    ])
    
    feats[7:] = feats[7:] / (np.linalg.norm(feats[7:]) + 1e-8)

    
    return feats

def texture_gabor_features(img, frequencies=[0.1, 0.2, 0.3, 0.4]):
    """
    Extracts texture features using Gabor filters.
    Returns a vector with the mean and variance of each filter's response.
    """
    gray = rgb2gray(np.array(img))
    feats = []
    for f in frequencies:
        filt_real, filt_imag = gabor(gray, frequency=f)
        feats.append(filt_real.mean())
        feats.append(filt_real.var())
        feats.append(filt_imag.mean())
        feats.append(filt_imag.var())
    return np.array(feats, dtype=float)


# --- DATA LOADING & FILTERING ---

def load_data_and_filter(brands: List[str], brand_folders: Dict[str, str], 
                         min_price: float, max_price: float) -> Tuple[List[Path], pd.DataFrame]:
    """
    Loads image paths and price data for the specified brands and filters by price range.
    Assumes a 'products.csv' file in each brand folder with 'file_name' and 'price_eur'.
    """
    all_filtered_images = []
    all_price_data = []

    for brand in brands:
        folder_name = brand_folders.get(brand)
        if not folder_name:
            print(f"⚠️ Carpeta no definida para la marca: {brand}")
            continue

        brand_dir = Path(folder_name)
        price_csv_path = brand_dir / "products.csv"
        
        if not brand_dir.is_dir():
            print(f"⚠️ Directorio no encontrado: {brand_dir.resolve()}")
            continue

        # 1. Load Price Data
        if not price_csv_path.exists():
            print(f"⚠️ CSV de precios no encontrado en: {price_csv_path}. Omitiendo marca {brand}.")
            continue

        try:
            price_df = pd.read_csv(price_csv_path)
            # Ensure required columns exist
            if 'file_name' not in price_df.columns or 'price_eur' not in price_df.columns:
                print(f"⚠️ CSV {price_csv_path} no tiene columnas 'file_name' o 'price_eur'. Omitiendo.")
                continue
            
            # Add brand column
            price_df['brand'] = brand
            
            # 2. Load Image Paths
            ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
            img_files_in_folder = {p.name: p for p in brand_dir.glob("*") if p.suffix.lower() in ALLOWED_IMAGE_SUFFIXES}
            
            # 3. Merge and Filter by File Existence & Price
            price_df['image_path'] = price_df['file_name'].apply(
                lambda x: img_files_in_folder.get(x)
            )

            # Filter out entries where the image file does not exist
            valid_df = price_df[price_df['image_path'].notna()].copy()
            
            if valid_df.empty:
                 print(f"⚠️ No se encontraron imágenes válidas para la marca {brand} que coincidan con el CSV.")
                 continue

            # Filter by Price Range
            price_filtered_df = valid_df[
                (valid_df['price_eur'] >= min_price) & (valid_df['price_eur'] <= max_price)
            ].copy()
            
            if price_filtered_df.empty:
                print(f"⚠️ Marca {brand} no tiene productos en el rango ${min_price} - ${max_price}.")
                continue
            
            all_price_data.append(price_filtered_df)
            all_filtered_images.extend(price_filtered_df['image_path'].tolist())

        except Exception as e:
            print(f"Error procesando datos para la marca {brand}: {e}")
            
    if not all_price_data:
        print("🛑 No se encontró data para ninguna marca/filtro. Terminando.")
        return [], pd.DataFrame()
        
    final_price_df = pd.concat(all_price_data, ignore_index=True)
    
    print(f"\n✅ Total de {len(all_filtered_images)} imágenes cargadas y filtradas.")
    print(f"   Rango de precios: ${min_price} - ${max_price}")
    print(f"   Marcas incluidas: {', '.join(brands)}")
    
    return all_filtered_images, final_price_df.drop(columns=['image_path'])


# --- NEW PLOTTING FUNCTIONS ---
def plot_clustered_images(emb_2d, labels, names, out_file, title, price_df_for_display: pd.DataFrame, show_outliers=False):
    """
    Creates a grid of images, organized by cluster, with colored borders and summary stats.
    """
    
    # Prepara un DataFrame para el merge y el cálculo de stats
    image_names_stem = [Path(n).name for n in names]
    
    # Crea un DF temporal para el merge, asegurando que los índices de labels coincidan
    temp_df = pd.DataFrame({
        'file_name': image_names_stem,
        'cluster_id': labels
    })
    
    # Merge con los datos originales (filtrados)
    # Usamos 'file_name' (el nombre original del archivo)
    merged_df = price_df_for_display.merge(temp_df, on='file_name', how='inner')
    
    if merged_df.empty:
        print("Error: No se pudo hacer merge de datos de precio con etiquetas de clúster.")
        return
        
    # Agrupa para obtener estadísticas por clúster
    cluster_stats = merged_df.groupby('cluster_id')['price_eur'].agg(
        count='count', mean_price='mean', min_price='min', max_price='max'
    ).round(2).to_dict('index')

    
    if show_outliers:
        unique_labels = sorted(list(set(labels)))
    else:
        unique_labels = sorted([l for l in list(set(labels)) if l != -1])

    n_clusters = len(unique_labels)
    if n_clusters == 0:
        print("No clusters to plot.")
        return
    
    plt.rcParams['figure.constrained_layout.use'] = False
    # Ajusta el tamaño de la figura para acomodar más clusters verticalmente
    fig, axes = plt.subplots(n_clusters, 1, figsize=(15, 3.5 * n_clusters))

    if n_clusters == 1:
        axes = [axes]

    # Asigna colores
    # Si hay outliers (-1), los pondremos en gris
    if -1 in unique_labels:
        # Excluye -1 para la paleta de colores Set3
        palette_labels = [l for l in unique_labels if l != -1]
        colors = plt.cm.Set3(np.linspace(0, 1, len(palette_labels)))
        # Inserta el gris para el label -1 (outlier)
        outlier_color = [0.7, 0.7, 0.7, 1] 
        if -1 in unique_labels:
             # Si -1 está presente, debemos ajustar el índice. 
             # Como lo ordenamos, -1 será el primero si está presente.
             if unique_labels[0] == -1:
                 colors = np.insert(colors, 0, outlier_color, axis=0)
             else:
                 # Esto no debería pasar si sorted(list(set(labels))) se usa
                 pass
    else:
        colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))

    
    plt.suptitle(title, fontsize=20, fontweight='bold')

    for i, label in enumerate(unique_labels):
        cluster_images = [Path(name).name for name, l in zip(names, labels) if l == label]
        cluster_ax = axes[i]
        n_images = len(cluster_images)
        if n_images == 0: continue
        
        # Genera el título con estadísticas
        stats = cluster_stats.get(label, {'count': n_images, 'mean_price': 0, 'min_price': 0, 'max_price': 0})
        
        if label == -1:
            cluster_title = f"Outliers (n={stats['count']})"
            color_index = i # El índice del color gris si -1 es el primero
        else:
            cluster_title = (
                f"Cluster {label} (n={stats['count']}) | "
                f"Avg Price: ${stats['mean_price']:.2f} | "
                f"Range: ${stats['min_price']:.2f} - ${stats['max_price']:.2f}"
            )
            color_index = i # El índice dentro del array de colores (ya ajustado si -1 existe)
            if -1 in unique_labels and unique_labels[0] == -1:
                 color_index = i # Si -1 es el primero, los demás están desplazados
        
        
        # Configura el layout de la cuadrícula de imágenes
        cols = min(10, n_images)
        rows = int(np.ceil(n_images / cols))
        img_w = 1.0 / cols
        
        for j, img_name_stem in enumerate(cluster_images):
            try:
                # Rebusca la imagen preprocesada por su nombre de archivo completo
                img_key = next(n for n in names if Path(n).name == img_name_stem)
                img = preprocessed_images[img_key]
                
                # Coordenadas relativas para el inset
                col_idx = j % cols
                row_idx = j // cols
                
                # Hacemos que la altura se ajuste dinámicamente según la cantidad de filas
                img_h = 1.0 / rows
                
                ax_img = cluster_ax.inset_axes(
                    [col_idx * img_w, 1.0 - (row_idx + 1) * img_h, img_w, img_h],
                    transform=cluster_ax.transAxes
                )
                
                ax_img.imshow(img)
                ax_img.set_xticks([])
                ax_img.set_yticks([])

                # Color del borde
                border_color = 'red' if label == -1 else colors[color_index]
                for spine in ax_img.spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(3)

            except Exception as e:
                print(f"Error cargando imagen {img_name_stem} para plotting: {e}")
        
        # Ajustar la altura del eje para que cubra todas las filas
        cluster_ax.set_ylim(0, 1) 
        
        # Establecer el título del clúster
        cluster_ax.set_title(cluster_title, loc='left', fontsize=14, pad=10)
        cluster_ax.set_axis_off() # Oculta el eje principal del clúster

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_file, dpi=200, bbox_inches='tight')
    plt.close()

def plot_dendrogram(X, names, out_file, title):
    """
    Plots a dendrogram for hierarchical clustering.
    """
    if not HAS_SCIPY:
        print("Scipy is not installed. Please install with 'pip install scipy' to plot dendrogram.")
        return
        
    linked_matrix = linkage(X, method='ward')

    plt.figure(figsize=(20, 10))
    dendrogram(linked_matrix, labels=names, leaf_rotation=90, leaf_font_size=8)
    plt.title(title, fontsize=20, fontweight='bold')
    plt.ylabel('Distance')
    plt.xlabel('Images')
    plt.savefig(out_file, dpi=200, bbox_inches='tight')
    plt.close()

# --- Clustering Functions (Reordering & Stats) ---

def cluster_dispersion_pairwise(X, labels):
    """"
    Returns the dispersion (average distance between pairs) of each cluster.
    Only considers clusters with 2 or more points.
    """
    dispersions = {}
    for cluster in set(labels):
        if cluster == -1:  #ignore outliers if any
            continue
        cluster_points = X[np.array(labels) == cluster]
        if len(cluster_points) <= 1:
            dispersions[cluster] = 0.0
        else:
            # pdist calculates all distances between pairs of points (euclidean by default)
            pairwise_dists = pdist(cluster_points, metric="euclidean")
            dispersions[cluster] = float(pairwise_dists.mean())
    return dispersions

def reorder_labels_by_dispersion(X, labels):
    """
    Reorders cluster labels (0, 1, 2, ...) based on decreasing intra-cluster dispersion.
    Clusters with the highest average pairwise distance (most dispersed) are ranked first.
    """
    dispersion_map = cluster_dispersion_pairwise(X, labels)
    
    # Get clusters, excluding outliers (-1)
    clusters = [c for c in sorted(set(labels)) if c != -1]
    
    # Sort clusters by dispersion (descending = most dispersed first)
    sorted_clusters = sorted(clusters, key=lambda c: dispersion_map.get(c, 0.0), reverse=True)

    # Relabel to 0..K-1 in this order
    cluster_order = {old: new for new, old in enumerate(sorted_clusters)}
    new_labels = np.array([cluster_order[l] if l in cluster_order else -1 for l in labels])

    # Build a ranked printable list
    ranked = []
    for r, c in enumerate(sorted_clusters, 1):
        ranked.append((r, c, dispersion_map.get(c, 0.0)))
        
    return new_labels, ranked

def cluster_pairwise_separation(X, labels, metric="euclidean", aggregate="mean"):
    """
    Compute cross-cluster separation for every pair and useful per-cluster stats.
    (Kept identical to original logic for consistency with 'separation' metric)
    """
    uniq = [c for c in sorted(set(labels)) if c != -1]
    pair_sep = {}
    # Compute all pairwise separations
    for i, h in enumerate(uniq):
        H = X[np.array(labels) == h]
        if H.shape[0] == 0:  # safety
            continue
        for f in uniq[i+1:]:
            F = X[np.array(labels) == f]
            if F.shape[0] == 0:
                continue
            D = pairwise_distances(H, F, metric=metric)
            if aggregate == "sum":
                s = float(D.sum())
            else:  # "mean" default
                s = float(D.mean())
            pair_sep[(h, f)] = s

    # Build per-cluster stats
    stats = {}
    for h in uniq:
        seps = []
        for f in uniq:
            if h == f:
                continue
            key = (h, f) if h < f else (f, h)
            if key in pair_sep:
                seps.append((f, pair_sep[key]))
        if len(seps) == 0:
            stats[h] = {'min_sep': 0., 'min_pair': None,
                        'mean_sep': 0., 'max_sep': 0., 'max_pair': None}
        else:
            f_min, v_min = min(seps, key=lambda t: t[1])
            f_max, v_max = max(seps, key=lambda t: t[1])
            v_mean = float(np.mean([v for _, v in seps]))
            stats[h] = {'min_sep': v_min, 'min_pair': f_min,
                        'mean_sep': v_mean, 'max_sep': v_max, 'max_pair': f_max}
    return pair_sep, stats


def reorder_labels_by_separation(X, labels,
                                 metric="euclidean",
                                 aggregate="mean",
                                 order_stat="min"):
    """
    Reorders cluster labels (0, 1, 2, ...) based on decreasing inter-cluster separation.
    (Kept identical to original logic for consistency with 'separation' metric)
    """
    _, stats = cluster_pairwise_separation(X, labels, metric=metric, aggregate=aggregate)
    
    # choose key per cluster for ordering
    if order_stat == "max":
        keyfun = lambda c: stats[c]['max_sep']
    elif order_stat == "mean":
        keyfun = lambda c: stats[c]['mean_sep']
    else:  # "min" default (farthest from nearest neighbor)
        keyfun = lambda c: stats[c]['min_sep']

    clusters = [c for c in sorted(set(labels)) if c != -1]
    # Sort clusters by chosen stat (descending = most separated first)
    sorted_clusters = sorted(clusters, key=keyfun, reverse=True)

    # Relabel to 0..K-1 in this order
    cluster_order = {old: new for new, old in enumerate(sorted_clusters)}
    new_labels = np.array([cluster_order[l] if l in cluster_order else -1 for l in labels])

    # Build a ranked printable list with the chosen stat and the paired cluster
    ranked = []
    for r, c in enumerate(sorted_clusters, 1):
        info = stats[c]
        if order_stat == "max":
            ranked.append((r, c, info['max_sep'], info['max_pair']))
        elif order_stat == "mean":
            ranked.append((r, c, info['mean_sep'], None))
        else:  # "min"
            ranked.append((r, c, info['min_sep'], info['min_pair']))
    return new_labels, ranked, stats


def find_optimal_k_agglomerative(X):
    """Finds a 'broad' optimal k for Agglomerative Clustering using Silhouette Score."""
    print("🔍 Optimizing k for Agglomerative Clustering (broader styles)...")
    best_k = None
    best_score = -1e9
    scores_list = []
    
    k_range = range(3, min(21, len(X) - 1))
    
    for k in k_range:
        clustering = AgglomerativeClustering(n_clusters=k)
        labels = clustering.fit_predict(X)
        if len(set(labels)) > 1:
            score = silhouette_score(X, labels)
            scores_list.append((k, score))
            # print(f"k={k} → Silhouette Score: {score:.4f}") # Comentado para evitar exceso de log
            
    if not scores_list: return None
        
    scores_list.sort(key=lambda x: x[1], reverse=True)
    best_score = scores_list[0][1]
    best_k = scores_list[0][0]
    
    # Choose a k close to the max score (within 5% tolerance) for better style breadth
    for k_val, score in scores_list:
        if k_val > 4 and score / best_score > 0.95:
            best_k = k_val
            break
            
    return max(4, best_k)

def run_agglomerative(X, names, price_df_for_display, out_file="clusters_agglomerative.png", title="Agglomerative Clustering", order_by="separation"):
    print("\n--- Running Agglomerative ---")
    
    if len(X) < 4:
         print(f"⚠️ Datos insuficientes para Agglomerative Clustering: {len(X)} muestras.")
         return None

    best_k = find_optimal_k_agglomerative(X)
    if best_k is not None:
        model = AgglomerativeClustering(n_clusters=best_k)
        labels = model.fit_predict(X)

        if order_by == "dispersion":
            # 4. Sumar las funciones de dispersión intra cluster 
            new_labels, ranked = reorder_labels_by_dispersion(X, labels)
            plot_title_suffix = f"(Ordered by Intra-Cluster Dispersion: most dispersed first)"
            # Console report for dispersion
            print("Cluster ranking by dispersion (highest average internal distance first):")
            for rank, c, disp_val in ranked:
                print(f"  Rank {rank}: Cluster {c} → dispersion {disp_val:.4f}")
        
        else: # Default is separation
            # 4. Sumar las funciones de dispersión inter cluster (mayor distancia al cluster más cercano)
            new_labels, ranked, stats = reorder_labels_by_separation(
                X, labels, metric="euclidean", aggregate="mean", order_stat="min"
            )
            plot_title_suffix = f"(Ordered by Inter-Cluster Separation: farthest-from-nearest)"
            
            # Console report for separation
            print("Cluster ranking by separation (farthest from nearest neighbor first):")
            for rank, c, sep_val, other in ranked:
                neighbor_txt = f" (nearest: {other})" if other is not None else ""
                print(f"  Rank {rank}: Cluster {c} → min-sep {sep_val:.4f}{neighbor_txt}")

            # (Optional) more detail
            print("\nSeparation stats per cluster [min / mean / max]:")
            for c in sorted(stats.keys()):
                s = stats[c]
                print(f"  Cluster {c}: min={s['min_sep']:.4f} (to {s['min_pair']}), "
                      f"mean={s['mean_sep']:.4f}, max={s['max_sep']:.4f} (to {s['max_pair']})")


        # 5. Sumar al plotting image, en el nombre de cada cluster (cluster = 1), la cantidad de elementos dentro del cluster, el precio míninimo, precio promedio y el precio máximo
        plot_clustered_images(
            X, new_labels, names, out_file,
            f"{title} {plot_title_suffix}",
            price_df_for_display=price_df_for_display
        )
        print(f"Agglomerative Clustering with k={best_k} saved to {out_file}")
        return new_labels

    return None

def run_kmeans(X, names, price_df_for_display, out_file="clusters_kmeans.png", title="K-Means Clustering"):
    print("\n--- Running K-Means ---")
    
    if len(X) < 4:
         print(f"⚠️ Datos insuficientes para K-Means Clustering: {len(X)} muestras.")
         return None

    best_k = None
    best_score = -1e9
    for k in range(2, min(12, len(X) - 1)):
        km = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = km.fit_predict(X)
        if len(set(labels)) > 1:
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_k = k
    if best_k is not None:
        model = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
        labels = model.fit_predict(X)
        
        # K-Means no se reordena por una métrica específica, pero se usan los stats
        plot_clustered_images(
            X, labels, names, out_file, 
            f"{title} (k={best_k})",
            price_df_for_display=price_df_for_display
        )
        print(f"K-Means with k={best_k} saved to {out_file}")
        return labels
    return None

def run_hdbscan(X, names, price_df_for_display, out_file="clusters_hdbscan.png", title="HDBSCAN Clustering"):
    print("\n--- Running HDBSCAN ---")
    if not HAS_HDBSCAN:
        print("HDBSCAN is not installed. Please install with 'pip install hdbscan' to use this method.")
        return None
    
    if len(X) < 4:
         print(f"⚠️ Datos insuficientes para HDBSCAN Clustering: {len(X)} muestras.")
         return None

    # HDBSCAN works best with cosine distance for high-dimensional embeddings
    dist_matrix = pairwise_distances(X, metric="cosine").astype(np.float64)

    # HDBSCAN with precomputed matrix
    model_hdbscan = hdbscan.HDBSCAN(
        min_cluster_size=3,
        min_samples=1,
        cluster_selection_epsilon=0.0,
        metric="precomputed"
    )
    labels = model_hdbscan.fit_predict(dist_matrix)

    # HDBSCAN can be reordered by the selected metric
    if CLUSTER_ORDER_BY == "dispersion":
        new_labels, ranked = reorder_labels_by_dispersion(X, labels)
        plot_title_suffix = f"(Ordered by Intra-Cluster Dispersion: most dispersed first)"
        print("Cluster ranking by dispersion (highest average internal distance first):")
        for rank, c, disp_val in ranked:
            print(f"  Rank {rank}: Cluster {c} → dispersion {disp_val:.4f}")
    
    else: # Default is separation
        new_labels, ranked, stats = reorder_labels_by_separation(
            X, labels, metric="euclidean", aggregate="mean", order_stat="min"
        )
        plot_title_suffix = f"(Ordered by Inter-Cluster Separation: farthest-from-nearest)"
        
        print("Cluster ranking by separation (farthest from nearest neighbor first):")
        for rank, c, sep_val, other in ranked:
            neighbor_txt = f" (nearest: {other})" if other is not None else ""
            print(f"  Rank {rank}: Cluster {c} → min-sep {sep_val:.4f}{neighbor_txt}")


    plot_clustered_images(
        X, new_labels, names, out_file, 
        f"{title} {plot_title_suffix}", 
        price_df_for_display=price_df_for_display, 
        show_outliers=True
    )

    print(f"HDBSCAN clustering saved to {out_file}")
    return new_labels


def build_shape_block(
    hog_matrix,
    shape_matrix,
    shape_factor=1.0,
    candidate_components=(16, 32, 48, 64),
    random_state=42,
):
    """Prepare the shape feature block by mixing HOG and geometric cues."""
    matrices = []
    if hog_matrix is not None and getattr(hog_matrix, 'size', 0) > 0:
        matrices.append(hog_matrix)
    if shape_matrix is not None and getattr(shape_matrix, 'size', 0) > 0:
        matrices.append(shape_matrix)
    if not matrices:
        raise ValueError('build_shape_block requires at least one non-empty shape matrix')

    shape_combined = np.hstack(matrices)
    scaler_shape = StandardScaler()
    shape_scaled = scaler_shape.fit_transform(shape_combined)

    best_score = float('-inf')
    best_n = 0
    best_shape = None
    candidate_components = tuple(candidate_components) if candidate_components else ()

    if shape_scaled.shape[0] > 3 and shape_scaled.shape[1] >= 2:
        for comp in candidate_components:
            if comp is None:
                continue
            n_components = min(int(comp), shape_scaled.shape[0], shape_scaled.shape[1])
            if n_components < 2:
                continue
            try:
                pca_temp = PCA(n_components=n_components)
                shape_pca_temp = pca_temp.fit_transform(shape_scaled)
            except ValueError:
                continue

            n_clusters = min(3, shape_pca_temp.shape[0] - 1)
            if n_clusters < 2:
                continue
            try:
                km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
                labels = km.fit_predict(shape_pca_temp)
            except ValueError:
                continue
            if len(set(labels)) <= 1:
                continue
            score = silhouette_score(shape_pca_temp, labels)
            if score > best_score:
                best_score = score
                best_n = n_components
                best_shape = shape_pca_temp

    used_pca = False
    if best_shape is not None:
        shape_out = best_shape
        used_pca = True
    else:
        fallback_n = min(32, shape_scaled.shape[0], shape_scaled.shape[1])
        if fallback_n >= 2 and shape_scaled.shape[1] > 2:
            try:
                pca = PCA(n_components=fallback_n)
                shape_out = pca.fit_transform(shape_scaled)
                used_pca = True
                best_n = fallback_n
            except ValueError:
                shape_out = shape_scaled
                best_n = shape_scaled.shape[1]
        else:
            shape_out = shape_scaled
            best_n = shape_scaled.shape[1]

    shape_out = shape_out * float(shape_factor)
    info = {
        'used_pca': used_pca,
        'n_components': best_n if used_pca else shape_scaled.shape[1],
        'original_dim': shape_combined.shape[1],
        'scaled_dim': shape_out.shape[1] if shape_out.ndim == 2 else 1,
    }
    return shape_out, info


# --- Main Pipeline Execution ---

filtered_img_files, price_df_filtered = load_data_and_filter(
    brands=BRANDS_TO_PROCESS, 
    brand_folders=BRAND_FOLDERS, 
    min_price=MIN_PRICE, 
    max_price=MAX_PRICE
)

if price_df_filtered.empty:
    raise SystemExit("No hay datos filtrados. Por favor, revisa las carpetas, CSVs y la configuración de marcas/precios.")

# El nombre de los archivos a procesar (solo el path completo)
img_files_for_processing = filtered_img_files 
names_for_clustering = [p.name for p in img_files_for_processing]


# --- Load Transformer-based Model for Embeddings ---

use_vit = False
vit_model = None
device = None

# Options: 'vit_base_patch16_224', 'beit_base_patch16_224', 'vit_base_patch16_224.mae'
MODEL_NAME = 'vit_base_patch16_224.mae'

if HAS_TORCH:
    try:
        # Usar la biblioteca wandb si está disponible
        print("wandb", wandb.__version__)
        print("has Result?", hasattr(p, "Result"))
    except NameError:
        pass # Ignorar si wandb no está importado/disponible

    try:
        vit_model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vit_model.to(device).eval()
        use_vit = True
        print(f"Loaded {MODEL_NAME} pretrained for embeddings.")
    except Exception as e:
        print(f"Could not load {MODEL_NAME} pretrained (falling back to classical features). Error: {e}")
        use_vit = False

# --- Extract Features ---

hog_embs = []
color_embs = []
shape_embs = []
vit_embs = []
texture_embs = []
preprocessed_images = {}


print(f"Processing {len(img_files_for_processing)} filtered images and extracting features...")
for p in img_files_for_processing:
    try:
        img = load_and_preprocess(p, target=(224, 224))
        
        # Usamos el nombre del archivo completo como clave, no solo el stem, 
        # para asegurar unicidad si imágenes de distintas marcas se llamaran igual
        preprocessed_images[p.name] = img 

        # Extracción de características
        hog_embs.append(hog_features(img))
        color_embs.append(color_hist_features(img))
        shape_embs.append(shape_properties_features(img))
        texture_embs.append(texture_gabor_features(img))
        
        if use_vit:
            vit_embs.append(vit_embedding(img, vit_model, device))

    except Exception as e:
        print(f"Error processing {p.name}: {e}")

if not hog_embs:
    raise SystemExit("No se pudieron extraer características de ninguna imagen filtrada. Terminando.")


hog_matrix = np.vstack(hog_embs)
color_matrix = np.vstack(color_embs)
shape_matrix = np.vstack(shape_embs)
texture_matrix = np.vstack(texture_embs)


vit_scaled = None  # inicializar

if use_vit and len(vit_embs) > 0:
    vit_matrix = np.vstack(vit_embs)  
    scaler_vit = StandardScaler()
    vit_scaled = scaler_vit.fit_transform(vit_matrix)
    print(f"ViT matrix shape: {vit_matrix.shape}")
else:
    print("⚠️ No ViT embeddings available, skipping ViT features.")

# -----------------------------------------------------------
# FINAL FEATURE ASSEMBLY (order matters; keep as one block)
# -----------------------------------------------------------

shape_factor = 3.0
color_factor = 1.0
texture_factor = 1.0

# 1) Scale color & texture
scaler_color = StandardScaler()
color_scaled = scaler_color.fit_transform(color_matrix)

scaler_texture = StandardScaler()
texture_scaled = scaler_color.fit_transform(texture_matrix)

# 2) Build SHAPE block (already * shape_factor inside the builder)
shape_pca, shape_info = build_shape_block(
    hog_matrix, shape_matrix, shape_factor=shape_factor,
    candidate_components=(16, 32, 48, 64),
    random_state=42
)
print(f"Shape block → used_pca={shape_info['used_pca']}, "
      f"n_components={shape_info['n_components']}, "
      f"original_dim={shape_info['original_dim']}, "
      f"scaled_dim={shape_info['scaled_dim']}")

# 3) ViT guard
if use_vit and vit_scaled is not None:
    features_vit = vit_scaled
    print(f"VIT feature matrix shape: {features_vit.shape}")
else:
    features_vit = None
    print("VIT feature matrix not available; will skip ViT clustering.")

# 4) Assemble the combined feature set
features_shape_color_texture = np.hstack([
    shape_pca,                         
    color_scaled * color_factor,
    texture_scaled * texture_factor
])
print(f"Shape+Color+Texture feature matrix shape: {features_shape_color_texture.shape}")

# --- Run clustering on VIT ---
print("\n🔹 Running clustering on VIT features...")
if features_vit is not None:
    labels_agg_VIT = run_agglomerative(
        features_vit, names_for_clustering, price_df_filtered,
        out_file="clusters_agglomerative_vit.png",
        title="Agglomerative Clustering (ViT Features)",
        order_by=CLUSTER_ORDER_BY
    )
    labels_kmeans_VIT = run_kmeans(
        features_vit, names_for_clustering, price_df_filtered,
        out_file="clusters_kmeans_vit.png",
        title="K-Means Clustering (ViT Features)"
    )
    labels_hdbscan_VIT = run_hdbscan(
        features_vit, names_for_clustering, price_df_filtered,
        out_file="clusters_hdbscan_vit.png",
        title="HDBSCAN Clustering (ViT Features)"
    )
else:
    labels_agg_VIT = labels_kmeans_VIT = labels_hdbscan_VIT = None
    print("Skipping ViT clustering because no ViT embeddings are available.")



# --- Run clustering on Shape  + Color + Texture ---
print("\n🔹 Running clustering on Shape+Color+Texture features...")

labels_agg_shape_color_texture = run_agglomerative(
    features_shape_color_texture,
    names_for_clustering,
    price_df_filtered,
    out_file="clusters_agglomerative_shape_color_texture.png",
    title="Agglomerative Clustering (Shape+Color+Texture)",
    order_by=CLUSTER_ORDER_BY
)

# Plot Dendrogram (uses the raw data, not the reordered labels)
if labels_agg_shape_color_texture is not None:
    plot_dendrogram(
        features_shape_color_texture,
        names_for_clustering,
        "dendrogram_shape_color_texture.png",
        "Agglomerative Clustering Dendrogram (Shape+Color+Texture Features)"
    )

labels_kmeans_shape_color_texture = run_kmeans(
    features_shape_color_texture,
    names_for_clustering,
    price_df_filtered,
    out_file="clusters_kmeans_shape_color_texture.png",
    title="K-Means Clustering (Shape+Color+Texture)"
)

labels_hdbscan_shape_color_texture = run_hdbscan(
    features_shape_color_texture,
    names_for_clustering,
    price_df_filtered,
    out_file="clusters_hdbscan_shape_color_texture.png",
    title="HDBSCAN Clustering (Shape+Color+Texture)"
)

print("\n✅ Clustering pipeline completed (ViT and Shape+Color+Texture) on filtered data.")

