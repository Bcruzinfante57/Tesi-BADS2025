import time, re, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import wandb
import wandb.proto.wandb_internal_pb2 as p
import joblib # Library for caching
print("wandb", wandb.__version__)
print("has Result?", hasattr(p, "Result"))

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
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

import cv2

# --- GLOBAL CONFIGURATION ---
EMBEDDING_CACHE_FILE = Path("all_brands_embeddings_cache.pkl")
TARGET_BRAND = "Fendi" # <--- CONFIGURE THE BRAND TO ANALYZE HERE
MIN_PRICE = 100
MAX_PRICE = 1000
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# IMAGEN PREPROCESSING #
_saved_first_crop = False
_saved_first_final = False


def crop_white_bg(pil_img, adaptive=True, padding=0.35): # PADDING MANTENIDO EN 0.35
    """
    Recorte de objeto robusto para fotos de productos (gafas), 
    maneja fondos blancos/grises variables, sombras suaves y bordes translúcidos.
    
    Ajustes aplicados para MEJORAR LA DETECCIÓN DE BORDES FINOS y evitar el recorte excesivo:
    1. Pre-procesamiento con Suavizado Gaussiano (Gaussian Blur) para reducir ruido de fondo.
    2. Umbral Adaptativo (C=15) más laxo para captar bordes finos.
    3. Cierre Extremo y Dilatación Reforzada para asegurar la fusión de todas las partes del objeto.
    4. **Detección de Múltiples Contornos** para asegurar que se capturen ambos lentes si están separados.
    5. Padding establecido en 0.35 (35% de margen) para un buen centrado.

    :param pil_img: PIL Image (asume fondo claro).
    :param adaptive: Booleano para usar Umbral Adaptativo (mantener en True).
    :param padding: Margen fraccional añadido alrededor del objeto (0.35 = 35%).
    :return: PIL Image recortada con fondo normalizado a blanco puro.
    """
    try:
        # Convertir a numpy array y escala de grises
        img_np = np.array(pil_img.convert("RGB"))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # === 0. PRE-PROCESAMIENTO: SUAVIZADO GAUSSIANO ===
        # Suavizar la imagen para reducir el ruido y las transiciones de sombra sutiles.
        gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # === 1. SEPARACIÓN DE FONDO: UMBRAL ADAPTATIVO ===
        # Aplicado sobre la imagen suavizada
        adaptive_mask = cv2.adaptiveThreshold(
            gray_blurred, 
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=51, 
            C=15         # C=15 para hacer la detección de objeto menos estricta
        )
        mask = adaptive_mask

        # === 2. ETAPA DE LIMPIEZA MORFOLÓGICA ===
        
        # Apertura (OPEN): Elimina ruido muy pequeño.
        kernel_small = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        # CIERRE EXTREMO: Conecta marcos rotos o partes del objeto.
        kernel_close = np.ones((25,25), np.uint8) 
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=4)
        
        # DILATACIÓN REFORZADA: Se mantiene para un último margen antes de detectar contornos.
        kernel_dilate = np.ones((10,10), np.uint8) 
        mask = cv2.dilate(mask, kernel_dilate, iterations=2) 


        # === 3. DETECTAR OBJETO PRINCIPAL (AHORA MÚLTIPLES OBJETOS) ===
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("⚠️ No se encontraron contornos — devolviendo imagen original.")
            return pil_img

        # Filtrar contornos muy pequeños (ruido residual)
        min_area = 200 # CAMBIO CLAVE: Reducido a 200 píxeles para incluir patillas finas y bordes
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        if not valid_contours:
            print("⚠️ Contornos válidos no encontrados después del filtrado — devolviendo imagen original.")
            return pil_img

        # Calcular una única caja delimitadora (bounding box) que abarque TODOS los contornos válidos
        all_points = np.concatenate(valid_contours)
        
        # Obtener la caja delimitadora (x, y, w, h) de todos los puntos
        x, y, w, h = cv2.boundingRect(all_points)

        # === 4. APLICAR PADDING (35%) ===
        pad_x = int(w * padding)
        pad_y = int(h * padding)
        
        # Calcular los límites con padding, asegurando no exceder los bordes de la imagen
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(img_np.shape[1], x + w + pad_x)
        y1 = min(img_np.shape[0], y + h + pad_y)
        
        cropped = img_np[y0:y1, x0:x1]

        if cropped.size == 0:
            print("⚠️ Imagen recortada vacía — devolviendo imagen original.")
            return pil_img

        # === 5. POST-PROCESAMIENTO: NORMALIZAR FONDO ===
        # Forzar a blanco puro cualquier píxel que sea claro (>= 250) en el área recortada.
        cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
        _, bg_mask = cv2.threshold(cropped_gray, 250, 255, cv2.THRESH_BINARY)
        bg_mask_3ch = cv2.cvtColor(bg_mask, cv2.COLOR_GRAY2BGR)
        clean_cropped = np.where(bg_mask_3ch == 255, 255, cropped)

        return Image.fromarray(clean_cropped.astype(np.uint8))
    
    except Exception as e:
        # Registro de errores
        print(f"Error crítico en crop_white_bg: {e}. Devolviendo imagen original.")
        return pil_img


def load_and_preprocess(p, target=(224, 224)):
    """Loads, crops, and centers an image for model input."""
    global _saved_first_crop, _saved_first_final

    img = Image.open(p).convert("RGB")
    img = crop_white_bg(img)

    # Save crop example
    if not _saved_first_crop:
        img.save("example_crop.jpg")
        # print("✅ Saved 'example_crop.jpg' (raw crop example).")
        _saved_first_crop = True

    # Create white background and center the object
    bg = Image.new("RGB", target, (255, 255, 255))
    img_contained = ImageOps.contain(img, target)

    paste_x = (target[0] - img_contained.width) // 2
    paste_y = (target[1] - img_contained.height) // 2
    bg.paste(img_contained, (paste_x, paste_y))

    # Save final centered example
    if not _saved_first_final:
        bg.save("example_final.jpg")
        # print("✅ Saved 'example_final.jpg' (final centered image).")
        _saved_first_final = True

    return bg

    
# --- Feature Extraction with ViT and classical methods ---
def vit_embedding(img_pil, model, device):
    """Extracts a feature vector using a pre-trained ViT model."""
        
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


def plot_clustered_images(emb_2d, labels, names, prices_list, out_file, title_str, show_outliers=False):
    """
    Creates an image grid organized by cluster. The function signature has been 
    modified to accept the prices list (prices_list) and use it for plotting.
    Price statistics (Min/Mean/Max) are added to each cluster's title.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    # Assumes 'preprocessed_images' is globally accessible.
    global preprocessed_images

    # 1. Data Adaptation
    # Map names to prices for quick lookup (uses the prices list)
    if len(names) == len(prices_list):
        price_map = dict(zip(names, prices_list))
    else:
        print("Error: Names and prices list lengths do not match. Prices will not be shown.")
        price_map = {}

    if show_outliers:
        unique_labels = sorted(list(set(labels)))
    else:
        unique_labels = sorted([l for l in list(set(labels)) if l != -1])

    n_clusters = len(unique_labels)
    if n_clusters == 0:
        print("No clusters to plot.")
        return
    
    plt.rcParams['figure.constrained_layout.use'] = False
    # Adjustment to prevent error when making subplots with a single row
    fig, axes = plt.subplots(max(1, n_clusters), 1, figsize=(15, 5 * n_clusters)) 

    if n_clusters == 1:
        axes = [axes] # Ensures it's always iterable

    colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
    if -1 in labels and show_outliers:
        colors = np.insert(colors, list(unique_labels).index(-1), [0.8, 0.8, 0.8, 1], axis=0)
    
    plt.suptitle(title_str, fontsize=20, fontweight='bold')

    for i, label in enumerate(unique_labels):
        cluster_images = [name for name, l in zip(names, labels) if l == label]
        
        # --- NEW: Get cluster prices and calculate statistics ---
        cluster_prices = [price_map[name] for name in cluster_images if name in price_map]
        stats_text = ""
        
        if cluster_prices:
            # Ensure prices are treated as numeric
            numeric_prices = np.array(cluster_prices, dtype=float) 
            
            price_min = np.min(numeric_prices)
            price_max = np.max(numeric_prices)
            price_mean = np.mean(numeric_prices)
            
            stats_text = (
                f" | Price Stats (Min/Mean/Max): "
                f"€{price_min:.0f} / €{price_mean:.0f} / €{price_max:.0f}"
            )
        # --- END NEW ---
        
        cluster_ax = axes[i]
        n_images = len(cluster_images)
        if n_images == 0: continue
        
        cols = min(10, n_images)
        img_h = 0.7 
        img_w = 1.0 / cols
        
        for j, img_name_stem in enumerate(cluster_images):
            try:
                # 2. Retrieve image
                if img_name_stem not in preprocessed_images:
                     print(f"Warning: Image stem '{img_name_stem}' not found in preprocessed_images. Skipping.")
                     continue
                    
                img = preprocessed_images[img_name_stem]
                
                # --- Text Preparation ---
                display_product_name = img_name_stem
                display_price = "N/A"
                
                # Get Price from map
                display_price_raw = price_map.get(img_name_stem)
                if display_price_raw is not None:
                    # Use the file name as the product name
                    if len(display_product_name) > 15: 
                        display_product_name = display_product_name[:12] + "..."
                        
                    display_price = f"€{float(display_price_raw):.0f}"
                
                # 3. Draw Image
                ax_img = cluster_ax.inset_axes(
                    [j * img_w, 1 - img_h, img_w, img_h],
                    transform=cluster_ax.transAxes
                )
                
                ax_img.imshow(img)
                ax_img.set_xticks([])
                ax_img.set_yticks([])

                # 4. Draw Border
                border_color = 'red' if label == -1 else colors[i]
                for spine in ax_img.spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(3)
                
                # 5. Draw Text (Name and Price)
                y_coord_name = 1 - img_h - 0.03
                y_coord_price = y_coord_name - 0.05 
                
                cluster_ax.text(
                    j * img_w + img_w / 2, 
                    y_coord_name,      
                    display_product_name,
                    transform=cluster_ax.transAxes,
                    ha='center', va='top', fontsize=8, color='black'
                )
                
                cluster_ax.text(
                    j * img_w + img_w / 2, 
                    y_coord_price,      
                    display_price,
                    transform=cluster_ax.transAxes,
                    ha='center', va='top', 
                    fontsize=11, 
                    fontweight='bold', 
                    color='black'
                )

            except Exception as e:
                # Error handling for image loading
                print(f"Error loading image stem {img_name_stem}: {e}")
        
        # --- Updated Cluster Title with Statistics ---
        cluster_ax.set_title(
            f"Cluster {label} (n={n_images}){stats_text}", 
            loc='left', 
            fontsize=14, 
            pad=10
        )
        cluster_ax.set_axis_off()

    # **FIX 2: Robust out_file handling**
    if not isinstance(out_file, str) or not any(out_file.lower().endswith(ext) for ext in ['.png', '.jpg', '.pdf']):
        safe_filename = f"clusters_output_{title_str.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        print(f"Warning: 'out_file' was an invalid type. Saving as: {safe_filename}")
        out_file = safe_filename
        
    plt.tight_layout(rect=[0, 0, 1, 0.94])
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

# --- Clustering Functions (No changes here, kept for completeness) ---
from itertools import combinations
from scipy.spatial.distance import pdist
from sklearn.metrics import pairwise_distances

def cluster_pairwise_separation(X, labels, metric="euclidean", aggregate="mean"):
    """
    Compute cross-cluster separation for every pair and useful per-cluster stats.
    Notes:
      - Outliers labeled -1 are ignored.
    """
    uniq = [c for c in sorted(set(labels)) if c != -1]
    pair_sep = {}
    for i, h in enumerate(uniq):
        H = X[np.array(labels) == h]
        if H.shape[0] == 0: continue
        for f in uniq[i+1:]:
            F = X[np.array(labels) == f]
            if F.shape[0] == 0: continue
            D = pairwise_distances(H, F, metric=metric)
            if aggregate == "sum": s = float(D.sum())
            else: s = float(D.mean())
            pair_sep[(h, f)] = s

    stats = {}
    for h in uniq:
        seps = []
        for f in uniq:
            if h == f: continue
            key = (h, f) if h < f else (f, h)
            if key in pair_sep: seps.append((f, pair_sep[key]))
        if len(seps) == 0:
            stats[h] = {'min_sep': 0., 'min_pair': None, 'mean_sep': 0., 'max_sep': 0., 'max_pair': None}
        else:
            f_min, v_min = min(seps, key=lambda t: t[1])
            f_max, v_max = max(seps, key=lambda t: t[1])
            v_mean = float(np.mean([v for _, v in seps]))
            stats[h] = {'min_sep': v_min, 'min_pair': f_min, 'mean_sep': v_mean, 'max_sep': v_max, 'max_pair': f_max}
    return pair_sep, stats


def reorder_labels_by_separation(X, labels, metric="euclidean", aggregate="mean", order_stat="min"):
    """
    Returns a remapped label array (0,1,2,...) sorted descending by the chosen stat.
    """
    _, stats = cluster_pairwise_separation(X, labels, metric=metric, aggregate=aggregate)
    if order_stat == "max": keyfun = lambda c: stats[c]['max_sep']
    elif order_stat == "mean": keyfun = lambda c: stats[c]['mean_sep']
    else: keyfun = lambda c: stats[c]['min_sep']

    clusters = [c for c in sorted(set(labels)) if c != -1]
    sorted_clusters = sorted(clusters, key=keyfun, reverse=True)

    cluster_order = {old: new for new, old in enumerate(sorted_clusters)}
    new_labels = np.array([cluster_order[l] if l in cluster_order else -1 for l in labels])

    ranked = []
    for r, c in enumerate(sorted_clusters, 1):
        info = stats[c]
        if order_stat == "max": ranked.append((r, c, info['max_sep'], info['max_pair']))
        elif order_stat == "mean": ranked.append((r, c, info['mean_sep'], None))
        else: ranked.append((r, c, info['min_sep'], info['min_pair']))
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
            print(f"k={k} → Silhouette Score: {score:.4f}")
            
    if not scores_list: return None
        
    scores_list.sort(key=lambda x: x[1], reverse=True)
    best_score = scores_list[0][1]
    best_k = scores_list[0][0]
    
    for k_val, score in scores_list:
        if k_val > 4 and score / best_score > 0.95:
            best_k = k_val
            break
            
    return max(4, best_k)

def run_agglomerative(X, names, prices, out_file="clusters_agglomerative.png", title="Agglomerative Clustering"):
    print("\n--- Running Agglomerative ---")
    best_k = find_optimal_k_agglomerative(X)
    if best_k is not None:
        model = AgglomerativeClustering(n_clusters=best_k)
        labels = model.fit_predict(X)
        new_labels, ranked, stats = reorder_labels_by_separation(
            X, labels, metric="euclidean", aggregate="mean", order_stat="min"
        )
        plot_clustered_images(
            X, new_labels, names, prices, out_file,
            f"{title} (Ordered by Separation: farthest-from-nearest)"
        )
        print("Cluster ranking by separation (farthest from nearest neighbor first):")
        for rank, c, sep_val, other in ranked:
            neighbor_txt = f" (nearest: {other})" if other is not None else ""
            print(f"  Rank {rank}: Cluster {c} → min-sep {sep_val:.4f}{neighbor_txt}")
        print("\nSeparation stats per cluster [min / mean / max]:")
        for c in sorted(stats.keys()):
            s = stats[c]
            print(f"  Cluster {c}: min={s['min_sep']:.4f} (to {s['min_pair']}), "
                  f"mean={s['mean_sep']:.4f}, max={s['max_sep']:.4f} (to {s['max_pair']})")
        return new_labels
    return None

def run_kmeans(X, names, prices, out_file="clusters_kmeans.png", title="K-Means Clustering"):
    print("\n--- Running K-Means ---")
    best_k = None
    best_score = -1e9
    for k in range(2, min(12, len(X) - 1)):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        if len(set(labels)) > 1:
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_k = k
    if best_k is not None:
        model = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
        labels = model.fit_predict(X)
        plot_clustered_images(X, labels, names, prices, out_file, f"{title} (k={best_k})")
        print(f"K-Means with k={best_k} saved to {out_file}")
        return labels
    return None

def run_hdbscan(X, names, prices, out_file="clusters_hdbscan.png", title="HDBSCAN Clustering"):
    print("\n--- Running HDBSCAN ---")
    if not HAS_HDBSCAN:
        print("HDBSCAN is not installed. Please install with 'pip install hdbscan' to use this method.")
        return None

    dist_matrix = pairwise_distances(X, metric="cosine").astype(np.float64)

    model_hdbscan = hdbscan.HDBSCAN(
        min_cluster_size=3,
        min_samples=1,
        cluster_selection_epsilon=0.0,
        metric="precomputed"
    )
    labels = model_hdbscan.fit_predict(dist_matrix)

    plot_clustered_images(X, labels, names, prices, out_file, title, show_outliers=True)

    print(f"HDBSCAN clustering saved to {out_file}")
    return labels


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


# --- 1. GENERAL CONFIGURATION --> BRANDS AND DATA ---
brands = ["Bottega Veneta", "Dolce&Gabbana", "Fendi", "Prada", "YSL"] 
brand_folders = {
    "Bottega Veneta": "images_bottega",
    "Dolce&Gabbana": "images_D&G",
    "Fendi": "images_Fendi",
    "Prada": "images_Prada",
    "YSL": "images_YSL"
}
# MIN_PRICE and MAX_PRICE are set globally at the top.

# 🔍 Loading images and CSVs for ALL brands
all_images_paths = [] 
all_prices_df_list = []

for brand in brands:
    folder = Path(brand_folders.get(brand, "")).resolve()
    if not folder.exists():
        print(f"⚠️ Folder not found for {brand}: {folder}")
        continue

    print(f"\n📂 Loading brand: {brand} → {folder}")

    brand_images_paths = sorted([p for p in folder.glob("*") if p.suffix.lower() in ALLOWED_IMAGE_SUFFIXES])
    # Store (brand, image_path) for embedding and lookup
    all_images_paths.extend([(brand, img_path) for img_path in brand_images_paths]) 
    
    csv_files = list(folder.glob("*.csv"))
    if not csv_files:
        print(f"⚠️ No CSV found for {brand}")
        continue

    df_prices = pd.read_csv(csv_files[0])

    # Normalize column names: strip spaces, lowercase, replace spaces with underscores
    df_prices.columns = df_prices.columns.str.strip().str.lower().str.replace(" ", "_")

    name_col = next((c for c in df_prices.columns if c.startswith("product")), None)
    price_col = next((c for c in df_prices.columns if "price" in c), None)
    
    if name_col is None or price_col is None:
        print(f"❌ CSV {csv_files[0].name} missing 'product' or 'price' column. Skipping brand {brand}.")
        continue 

    df_prices["brand"] = brand
    # Create standardized 'product_name' (filename stem)
    df_prices["product_name"] = df_prices[name_col].astype(str).apply(lambda x: Path(x).stem)
    
    # Clean and convert the price column to numeric in 'price_eur'
    df_prices["price_eur"] = (
        df_prices[price_col]
        .astype(str) 
        .str.replace(r"[^\d.]", "", regex=True) # Remove anything that is not a digit or dot
    )
    df_prices["price_eur"] = pd.to_numeric(df_prices["price_eur"], errors='coerce')
    df_prices.dropna(subset=['price_eur'], inplace=True)
    
    # --- KEY FIX: Use the STEM (product_name) as the universal join key ---
    df_prices["join_key"] = df_prices["product_name"] 
    # The 'product_name' column already contains the STEM: e.g., 'Prada_1'
    
    all_prices_df_list.append(df_prices)

if not all_prices_df_list:
    raise SystemExit("❌ No CSVs with price data found for the selected brands.")

price_df_full = pd.concat(all_prices_df_list, ignore_index=True) 
# The map of all image paths should be keyed by the STEM, not the full name.
all_image_paths_map = {p.stem: (b, p) for b, p in all_images_paths} # <-- KEY FIX: Map by STEM
all_image_stems = list(all_image_paths_map.keys())

print(f"\n✅ Loaded {len(all_images_paths)} images and {len(price_df_full)} full price entries across {len(brands)} brands.")


print("\n==============================================")
print("  LUXURY EYEWEAR CLUSTERING PIPELINE")
print("  Brands loaded:", ", ".join(brands))
print(f"  Target Brand for Analysis: {TARGET_BRAND}")
print(f"  Price range filter: €{MIN_PRICE} - €{MAX_PRICE}")
print("==============================================\n")

# --- 2. Caching and Feature Extraction (All Brands) ---

# Structure for caching: {stem: {vit: array, color: array, hog: array, shape: array, texture: array}}
all_embeddings = {}
run_embedding = True

if EMBEDDING_CACHE_FILE.exists():
    try:
        all_embeddings = joblib.load(EMBEDDING_CACHE_FILE)
        
        # Check if we have embeddings for ALL image stems found
        missing_stems = [stem for stem in all_image_stems if stem not in all_embeddings]
        
        if not missing_stems:
            run_embedding = False
            print(f"✨ Loaded {len(all_embeddings)} embeddings from cache: {EMBEDDING_CACHE_FILE}")
        else:
            print(f"⚠️ Cache found, but {len(missing_stems)} image stems are missing embeddings. Re-running for missing stems...")
            all_image_stems_to_process = missing_stems # Update list to process only missing ones
            # For logging purposes, we use the full list if we are reprocessing the whole thing
            if len(missing_stems) == len(all_image_stems):
                 print("Full reprocessing required.")
                 all_image_stems_to_process = all_image_stems
            
            # The next loop will use all_image_stems_to_process
            # The cache key is the stem.
            # We need the full path 'p' to load the image.
            
    except Exception as e:
        print(f"❌ Error loading cache: {e}. Re-running all embeddings.")
        all_embeddings = {}
        all_image_stems_to_process = all_image_stems
        # Keep run_embedding = True
else:
    all_image_stems_to_process = all_image_stems

# --- Load Transformer-based Model for Embeddings ---

use_vit = False
vit_model = None
device = None

# Options: 'vit_base_patch16_224', 'beit_base_patch16_224', 'vit_base_patch16_224.mae'
MODEL_NAME = 'vit_base_patch16_224.mae'

if HAS_TORCH and run_embedding:
    try:
        vit_model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vit_model.to(device).eval()
        use_vit = True
        print(f"Loaded {MODEL_NAME} pretrained for embeddings.")
    except Exception as e:
        print(f"Could not load {MODEL_NAME} pretrained (falling back to classical features).", e)
        use_vit = False

# --- Extract Features for ALL brands (if not cached) ---
new_embeddings_count = 0
preprocessed_images = {} # Will store images for the TARGET_BRAND later

if run_embedding:
    print(f"Processing {len(all_image_stems_to_process)} images and extracting features...")
    # Iterate over all image stems (or just the missing ones)
    for stem in all_image_stems_to_process:
        brand, p = all_image_paths_map[stem] # Get the path 'p' using the STEM as key
        try:
            img = load_and_preprocess(p, target=(224, 224))
            
            # Extract features
            feats = {
                'color': color_hist_features(img),
                'hog': hog_features(img),
                'shape': shape_properties_features(img),
                'texture': texture_gabor_features(img),
            }
            if use_vit:
                feats['vit'] = vit_embedding(img, vit_model, device)
            else:
                feats['vit'] = np.array([])
            
            # Cache key is the STEM
            all_embeddings[stem] = feats 
            new_embeddings_count += 1

        except Exception as e:
            print(f"Error processing {p.name} (Stem: {stem}): {e}")
            traceback.print_exc()

    # Save cache if new embeddings were created
    if new_embeddings_count > 0:
        joblib.dump(all_embeddings, EMBEDDING_CACHE_FILE)
        print(f"💾 Saved {new_embeddings_count} new embeddings to cache: {EMBEDDING_CACHE_FILE}")


# --- 3. Filter Data for TARGET_BRAND and Price Range ---

# Filter prices: target brand + price range
target_df = price_df_full[
    (price_df_full["brand"] == TARGET_BRAND) &
    (price_df_full["price_eur"] >= MIN_PRICE) &
    (price_df_full["price_eur"] <= MAX_PRICE)
].copy()

# Filter images/embeddings: only keep those that match the filtered prices and are in the embeddings map
target_names = [] # Will store the STEMs (Prada_1, Prada_2, etc.)
target_embeddings_list = []
target_prices_list = []

print(f"\n[DEBUG] Found {len(target_df)} price entries for {TARGET_BRAND} in range €{MIN_PRICE}-€{MAX_PRICE}.")

# Join target_df with all_embeddings keys
for _, row in target_df.iterrows():
    join_key = row['join_key'] # This is the STEM (e.g., Prada_1)
    price = row['price_eur']
    
    if join_key in all_embeddings:
        target_names.append(join_key)
        target_prices_list.append(price)
        target_embeddings_list.append(all_embeddings[join_key])
        
        # Load the preprocessed image for plotting later (only for the target brand)
        if join_key in all_image_paths_map:
             # Load and preprocess again, since 'img' from extraction loop is gone
             _, p = all_image_paths_map[join_key]
             preprocessed_images[join_key] = load_and_preprocess(p, target=(224, 224))
    # Note: No 'else' needed here, simply skips if no embedding found.


if not target_names:
    raise SystemExit(f"❌ No images found for Brand: {TARGET_BRAND} in price range: €{MIN_PRICE}-€{MAX_PRICE} after filtering/caching. (Check for join errors or file structure)")


print(f"\n✅ Ready for analysis: {len(target_names)} products from {TARGET_BRAND} (Price: €{MIN_PRICE}-€{MAX_PRICE}).")

# --- 4. Prepare Feature Matrices for Analysis ---

# Collect the components for the target set
vit_embs = [e['vit'] for e in target_embeddings_list]
color_embs = [e['color'] for e in target_embeddings_list]
hog_embs = [e['hog'] for e in target_embeddings_list]
shape_embs = [e['shape'] for e in target_embeddings_list]
texture_embs = [e['texture'] for e in target_embeddings_list]

# Stack matrices (safe for empty lists/arrays if only one feature is used)
vit_matrix = np.vstack(vit_embs) if any(e.size > 0 for e in vit_embs) else None
color_matrix = np.vstack(color_embs)
hog_matrix = np.vstack(hog_embs)
shape_matrix = np.vstack(shape_embs)
texture_matrix = np.vstack(texture_embs)

# --- Feature Scaling and Assembly ---

vit_scaled = None
use_vit_for_clustering = False

if vit_matrix is not None and vit_matrix.ndim == 2 and vit_matrix.shape[0] > 0 and vit_matrix.shape[1] > 1:
    scaler_vit = StandardScaler()
    vit_scaled = scaler_vit.fit_transform(vit_matrix)
    use_vit_for_clustering = True
    print(f"ViT matrix shape: {vit_matrix.shape}")
else:
    print("⚠️ ViT embeddings not available or insufficient for clustering.")
    
# Hyper-parameters for weighting
shape_factor = 4.0
color_factor = 1.0
texture_factor = 1.0

# 1) Scale color & texture
scaler_color = StandardScaler()
color_scaled = scaler_color.fit_transform(color_matrix)

scaler_texture = StandardScaler()
texture_scaled = scaler_texture.fit_transform(texture_matrix)

# 2) Build SHAPE block (always returns a valid array)
shape_pca, shape_info = build_shape_block(
    hog_matrix, shape_matrix, shape_factor=shape_factor,
    candidate_components=(16, 32, 48, 64),
    random_state=42
)
print(f"Shape block → used_pca={shape_info['used_pca']}, "
      f"n_components={shape_info['n_components']}, "
      f"original_dim={shape_info['original_dim']}, "
      f"scaled_dim={shape_info['scaled_dim']}")

# 3) Assemble the combined feature set
features_shape_color_texture = np.hstack([
    shape_pca,                         # already * shape_factor inside the builder
    color_scaled * color_factor,
    texture_scaled * texture_factor
])
print(f"Shape+Color+Texture feature matrix shape: {features_shape_color_texture.shape}")

# --- Run clustering on VIT ---
print("\n🔹 Running clustering on VIT features...")
if use_vit_for_clustering:
    labels_agg_VIT = run_agglomerative(
        vit_scaled, target_names, target_prices_list,
        out_file=f"clusters_agg_VIT_{TARGET_BRAND}.png",
        title=f"Agglomerative Clustering (ViT Embeddings) - {TARGET_BRAND} (€{MIN_PRICE}-€{MAX_PRICE})"
    )
    
    # ✅ DENDROGRAM NOW USES VIT_SCALED FEATURES
    if labels_agg_VIT is not None:
        plot_dendrogram(
            vit_scaled, # <-- USING ViT FEATURES
            target_names,
            f"dendrogram_agg_VIT_{TARGET_BRAND}.png", # <-- UPDATED FILENAME
            f"Agglomerative Clustering Dendrogram (ViT Embeddings) - {TARGET_BRAND}" # <-- UPDATED TITLE
        )
        
    labels_kmeans_VIT = run_kmeans(
        vit_scaled, target_names, target_prices_list,
        out_file=f"clusters_kmeans_VIT_{TARGET_BRAND}.png",
        title=f"K-Means Clustering (ViT Embeddings) - {TARGET_BRAND} (€{MIN_PRICE}-€{MAX_PRICE})"
    )
    labels_hdbscan_VIT = run_hdbscan(
        vit_scaled, target_names, target_prices_list,
        out_file=f"clusters_hdbscan_VIT_{TARGET_BRAND}.png",
        title=f"HDBSCAN Clustering (ViT Embeddings) - {TARGET_BRAND} (€{MIN_PRICE}-€{MAX_PRICE})"
    )
else:
    labels_agg_VIT = labels_kmeans_VIT = labels_hdbscan_VIT = None
    print("Skipping ViT clustering.")


# --- Run clustering on Shape  + Color + Texture ---
print("\n🔹 Running clustering on Shape+Color+Texture features...")

labels_agg_shape_color_texture = run_agglomerative(
    features_shape_color_texture,
    target_names,
    target_prices_list,
    out_file=f"clusters_agg_shape_color_texture_{TARGET_BRAND}.png",
    title=f"Agglomerative Clustering (Shape+Color+Texture) - {TARGET_BRAND} (€{MIN_PRICE}-€{MAX_PRICE})"
)
if labels_agg_shape_color_texture is not None:
    pass 

labels_kmeans_shape_color_texture = run_kmeans(
    features_shape_color_texture,
    target_names,
    target_prices_list,
    out_file=f"clusters_kmeans_shape_color_texture_{TARGET_BRAND}.png",
    title=f"K-Means Clustering (Shape+Color+Texture) - {TARGET_BRAND} (€{MIN_PRICE}-€{MAX_PRICE})"
)

labels_hdbscan_shape_color_texture = run_hdbscan(
    features_shape_color_texture,
    target_names,
    target_prices_list,
    out_file=f"clusters_hdbscan_shape_color_texture_{TARGET_BRAND}.png",
    title=f"HDBSCAN Clustering (Shape+Color+Texture) - {TARGET_BRAND} (€{MIN_PRICE}-€{MAX_PRICE})"
)

print("\n✅ Clustering pipeline completed.")