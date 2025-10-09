import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2


# Libraries for clustering and dimensionality reduction
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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

# --- 2 GENERAL CONFIGURATION --> BRANDS AND DATA ---

brands = ["Bottega Veneta", "Dolce&Gabbana", "Fendi", "Prada", "YSL"] 
min_price, max_price = 0, 1000

brand_folders = {
    "Bottega Veneta": "images_bottega",
    "Dolce&Gabbana": "images_D&G",
    "Fendi": "images_Fendi",
    "Prada": "images_Prada",
    "YSL": "images_YSL"
}

# 🔍 Uploading images and CSVs
all_images_paths = [] 
all_prices = []

for brand in brands:
    folder = Path(brand_folders.get(brand, "")).resolve()
    if not folder.exists():
        print(f"⚠️ Folder not found for {brand}: {folder}")
        continue

    print(f"\n📂 Loading brand: {brand} → {folder}")

    brand_images_paths = sorted([p for p in folder.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")])
    all_images_paths.extend([(brand, img_path) for img_path in brand_images_paths]) 
    
    ## DEBUG ##
    print(f"  Debug: Found {len(brand_images_paths)} image files for {brand}.")
    ## FIN DEBUG ##

    csv_files = list(folder.glob("*.csv"))
    if not csv_files:
        print(f"⚠️ No CSV found for {brand}")
        continue

    df_prices = pd.read_csv(csv_files[0])

    # Normalizar nombres de columnas: eliminar espacios, minúsculas, reemplazar espacios por guiones bajos
    df_prices.columns = df_prices.columns.str.strip().str.lower().str.replace(" ", "_")

    name_col = next((c for c in df_prices.columns if c.startswith("product")), None)
    price_col = next((c for c in df_prices.columns if "price" in c), None)
    
    if name_col is None:
        print(f"❌ CSV {csv_files[0].name} missing a column starting with 'product'. Skipping brand {brand}.")
        continue 

    if price_col is None:
        print(f"❌ CSV {csv_files[0].name} missing a column containing 'price'. Skipping brand {brand}.")
        continue 

    df_prices["brand"] = brand
    # Crea 'product_name' estandarizado (stem del nombre)
    df_prices["product_name"] = df_prices[name_col].astype(str).apply(lambda x: Path(x).stem)
    
    ## DEBUG ##
    print(f"  Debug ({brand}): Columna original de precios '{price_col}' (primeras 5):")
    print(df_prices[price_col].head())
    print(f"  Debug ({brand}): Type of original price column: {df_prices[price_col].dtype}")
    ## FIN DEBUG ##

    # Limpia y convierte la columna de precios a numérico en 'price_eur'
    # Utiliza .str.replace para asegurar que se aplica a cadenas
    df_prices["price_eur"] = (
        df_prices[price_col]
        .astype(str) 
        .str.replace(r"[^\d.]", "", regex=True) # Reemplaza cualquier cosa que no sea dígito o punto por vacío
    )
    df_prices["price_eur"] = pd.to_numeric(df_prices["price_eur"], errors='coerce')
    df_prices.dropna(subset=['price_eur'], inplace=True)

    ## DEBUG ##
    print(f"  Debug ({brand}): Columna 'product_name' después del procesamiento (primeras 5):")
    print(df_prices["product_name"].head())
    print(f"  Debug ({brand}): Columna 'price_eur' después de la limpieza (primeras 5):")
    print(df_prices['price_eur'].head())
    print(f"  Debug ({brand}): Number of valid prices after cleaning: {len(df_prices)}")
    if not df_prices['price_eur'].empty:
        print(f"  Debug ({brand}): Min/Max price in 'price_eur' DF: {df_prices['price_eur'].min()} / {df_prices['price_eur'].max()}")
        # Check how many fall in the filter range before concat
        in_range_count = df_prices[(df_prices['price_eur'] >= min_price) & (df_prices['price_eur'] <= max_price)].shape[0]
        print(f"  Debug ({brand}): Products in range {min_price}-{max_price} before concat: {in_range_count}")
    else:
        print(f"  Debug ({brand}): No valid prices found.")
    ## FIN DEBUG ##

    all_prices.append(df_prices)

if not all_prices:
    raise SystemExit("❌ No CSVs with price data found for the selected brands.")

price_df_full = pd.concat(all_prices, ignore_index=True) 
all_image_paths_list = [img_path for (_, img_path) in all_images_paths] 
print(f"\n✅ Loaded {len(all_image_paths_list)} images and {len(price_df_full)} full price entries across {len(brands)} brands.")


print("\n==============================================")
print("  LUXURY EYEWEAR CLUSTERING PIPELINE")
print("  Brands loaded:", ", ".join(brands))
print(f"  Price range filter: €{min_price} - €{max_price}")
print("==============================================\n")

# --- IMAGEN PREPROCESSING ---
_saved_first_crop = False
_saved_first_final = False

# The 'normalize_background' function has been removed as requested. 

def crop_object_with_padding(pil_img, bg_value_threshold=235, padding_ratio=0.35): 
    """
    Crops the main object with padding, using reinforced morphology to
    capture translucent frames and thin temples (pincers).
    
    NOTE: This function assumes the input image has a relatively clean, light background
    (typically white or light gray).
    
    :param pil_img: Image with white background (or normalized).
    :param bg_value_threshold: Lower threshold to detect light/translucent frames.
    :param padding_ratio: Percentage of padding (e.g., 0.35 = 35% margin).
    """
    img_np = np.array(pil_img.convert("RGB"))
    
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # ADJUSTMENT 1: Low threshold (235) to capture light/translucent color elements.
    # THRESH_BINARY_INV ensures the object is white and the background is black.
    _, mask = cv2.threshold(gray, bg_value_threshold, 255, cv2.THRESH_BINARY_INV) 
    
    # CRITICAL ADJUSTMENT 2 (Temples/Translucency): Apply a very large closing operation.
    # The large kernel (15x15) with 5 iterations forces the union of thin temples
    # or the filling of small gaps in translucent frames.
    kernel_close = np.ones((15,15), np.uint8) 
    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=5) 
    
    # ADJUSTMENT 3: Final dilation to ensure the mask is larger than the object.
    kernel_dilate = np.ones((7,7),np.uint8) 
    dilated_mask = cv2.dilate(cleaned_mask, kernel_dilate, iterations=3) 

    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("Warning: No contours found for cropping. Returning original image.")
        return pil_img 
        
    largest_cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_cnt)

    # ADJUSTMENT 4: Padding: Use padding_ratio directly (e.g., 0.35)
    pad_x = int(w * padding_ratio) 
    pad_y = int(h * padding_ratio)
    
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(img_np.shape[1], x + w + pad_x)
    x2 = min(img_np.shape[0], y + h + pad_y) 

    cropped = img_np[y0:x2, x0:x1] # Use x2 for the final y-axis limit
    
    if cropped.size == 0:
        print("Warning: Cropped image has zero size. Returning original image.")
        return pil_img
    
    # CRITICAL ADJUSTMENT 5 (Post-Processing for Pure Background):
    # This step eliminates any gray residue/shadow that may have survived the crop,
    # ensuring a pure white background for clustering.
    cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
    # If the pixel value is greater than 250, we consider it background.
    _, residual_mask = cv2.threshold(cropped_gray, 250, 255, cv2.THRESH_BINARY)
    residual_mask_3ch = cv2.cvtColor(residual_mask, cv2.COLOR_GRAY2BGR)
    
    # Where the mask is white (residue), we put pure white (255)
    clean_cropped = np.where(residual_mask_3ch == 255, 255, cropped)
    
    return Image.fromarray(clean_cropped.astype(np.uint8))

# --- PRIMARY FLOW FUNCTION (CV2 ONLY) ---

def process_and_crop_pipeline(pil_img, padding_ratio=0.35, crop_bg_threshold=235):
    """
    Applies the complete CV2-Only flow: 
    1. Robust Crop/Padding based on color threshold.
    """
    print(f"Starting CV2-Only pipeline with padding={padding_ratio} and threshold={crop_bg_threshold}")
    
    # We call crop_object_with_padding directly, as the initial normalization and 
    # rembg steps have been removed. The crop function handles post-processing.
    cropped_img = crop_object_with_padding(
        pil_img, 
        bg_value_threshold=crop_bg_threshold,
        padding_ratio=padding_ratio
    )
    
    print("Pipeline finished successfully (CV2 ONLY).")
    return cropped_img

def load_and_preprocess(p, target=(224, 224)):
    """Loads, crops, and centers an image for model input."""
    global _saved_first_crop, _saved_first_final

    img = Image.open(p).convert("RGB")

    # Removed the check for REMBG_AVAILABLE and the rembg execution logic.
    img = process_and_crop_pipeline(img)
        
    # Save crop example (raw crop)
    if not _saved_first_crop:
        img.save("example_crop.jpg")
        print("✅ Saved 'example_crop.jpg' (raw CV2 crop result example).")
        _saved_first_crop = True

    # Create white background and center the object
    bg = Image.new("RGB", target, (255, 255, 255))
    # ImageOps.contain resizes while maintaining aspect ratio and without cropping
    img_contained = ImageOps.contain(img, target)

    paste_x = (target[0] - img_contained.width) // 2
    paste_y = (target[1] - img_contained.height) // 2
    bg.paste(img_contained, (paste_x, paste_y))

    # Save final centered example
    if not _saved_first_final:
        bg.save("example_final.jpg")
        print("✅ Saved 'example_final.jpg' (final centered image).")
        _saved_first_final = True

    return bg
    
# --- Feature Extraction with ViT and classical methods ---
def vit_embedding(img_pil, model, device):
   ## """"Extracts a feature vector using a pre-trained ViT model.""""
        
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
- frequencies: List of frequencies for the filters.
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

# --- NEW PLOTTING FUNCTIONS ---

def plot_clustered_images(emb_2d, labels, names, out_file, title, price_df_for_display, show_outliers=False):
    """
    Creates a grid of images, organized by cluster, with colored borders.
    price_df_for_display: El DataFrame de precios ya filtrado que contiene info para las imágenes a plotear.
    """
    if show_outliers:
        unique_labels = sorted(list(set(labels)))
    else:
        unique_labels = sorted([l for l in list(set(labels)) if l != -1])

    n_clusters = len(unique_labels)
    if n_clusters == 0:
        print("No clusters to plot.")
        return
    
    plt.rcParams['figure.constrained_layout.use'] = False
    fig, axes = plt.subplots(n_clusters, 1, figsize=(15, 5 * n_clusters)) # ★ Ajustado el tamaño de la figura para más espacio

    if n_clusters == 1:
        axes = [axes]

    colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
    if -1 in labels and show_outliers:
        colors = np.insert(colors, list(unique_labels).index(-1), [0.8, 0.8, 0.8, 1], axis=0)
    
    plt.suptitle(title, fontsize=20, fontweight='bold')

    for i, label in enumerate(unique_labels):
        cluster_images = [name for name, l in zip(names, labels) if l == label]
        cluster_ax = axes[i]
        n_images = len(cluster_images)
        if n_images == 0: continue
        
        cols = min(10, n_images)
        # ★ Ajusta el tamaño de la imagen y el espaciado para hacer espacio al texto
        img_h = 0.7 # Altura de la imagen dentro del subplot
        img_w = 1.0 / cols
        
        for j, img_name in enumerate(cluster_images):
            try:
                img_path_obj = Path(img_name) 
                img = preprocessed_images[img_path_obj.name] 
                
                # Buscar el nombre del producto y el precio en price_df_for_display
                product_name_clean = img_path_obj.stem 
                
                # ★ Usar price_df_for_display (que ya estará filtrado)
                product_info = price_df_for_display[price_df_for_display['product_name'] == product_name_clean]
                
                display_text = ""
                if not product_info.empty:
                    display_product_name = product_info['product_name'].iloc[0].replace('_id', '').replace('_', ' ').title()
                    if len(display_product_name) > 15: 
                        display_product_name = display_product_name[:12] + "..."
                        
                    display_price = product_info['price_eur'].iloc[0]
                    display_text = f"{display_product_name}\n€{display_price:.0f}"
                else:
                    display_text = f"{product_name_clean}\nN/A" 
                
                # Ajustar el inset_axes para incluir espacio para el texto
                ax_img = cluster_ax.inset_axes(
                    [j * img_w, 1 - img_h, img_w, img_h],
                    transform=cluster_ax.transAxes
                )
                
                ax_img.imshow(img)
                ax_img.set_xticks([])
                ax_img.set_yticks([])

                if label == -1:
                    for spine in ax_img.spines.values():
                        spine.set_edgecolor('red')
                        spine.set_linewidth(3)
                else:
                    for spine in ax_img.spines.values():
                        spine.set_edgecolor(colors[i])
                        spine.set_linewidth(3)
                
                # Añadir el texto debajo de la imagen
                cluster_ax.text(
                    j * img_w + img_w / 2, 
                    1 - img_h - 0.05,      
                    display_text,
                    transform=cluster_ax.transAxes,
                    ha='center', va='top', fontsize=8, color='black'
                )

            except Exception as e:
                print(f"Error loading image {img_name}: {e}")
        
        cluster_ax.set_title(f"Cluster {label} (n={n_images})", loc='left', fontsize=14, pad=10)
        cluster_ax.set_axis_off()

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

# --- Clustering Functions ---
from itertools import combinations
from scipy.spatial.distance import pdist

def cluster_dispersion_pairwise(X, labels):
    """"Returns the dispersion (average distance between pairs) of each cluster."""
    dispersions = {}
    for cluster in set(labels):
        if cluster == -1:  #ignore outliers if any
            continue
        cluster_points = X[np.array(labels) == cluster]
        if len(cluster_points) <= 1:
            dispersions[cluster] = 0
        else:
            # pdist calculates all distances between pairs of points
            pairwise_dists = pdist(cluster_points, metric="euclidean")
            dispersions[cluster] = pairwise_dists.mean()
    return dispersions

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
            
    if not scores_list: return None #
        
    scores_list.sort(key=lambda x: x[1], reverse=True)
    best_score = scores_list[0][1]
    best_k = scores_list[0][0]
    
    for k_val, score in scores_list: 
        if k_val > 4 and score / best_score > 0.95:
            best_k = k_val
            break
            
    return max(4, best_k) 

def run_agglomerative(X, names, out_file="clusters_agglomerative.png", title="Agglomerative Clustering", price_df_for_display=None):
    print("\n--- Running Agglomerative ---")
    if len(X) <= 1: 
        print("⚠️ Not enough samples for Agglomerative Clustering. Skipping.")
        return None
    
    best_k = find_optimal_k_agglomerative(X)
    if best_k is not None:
        model = AgglomerativeClustering(n_clusters=best_k)
        labels = model.fit_predict(X)

        dispersions = cluster_dispersion_pairwise(X, labels)

        sorted_clusters = sorted([item for item in dispersions.items() if item[0] != -1], key=lambda x: x[1], reverse=True)
        
        cluster_order = {}
        current_new_label = 0
        for old_label, _ in sorted_clusters:
            cluster_order[old_label] = current_new_label
            current_new_label += 1
        if -1 in labels: 
            cluster_order[-1] = -1 

        new_labels = np.array([cluster_order.get(l, l) for l in labels])

        plot_clustered_images(X, new_labels, names, out_file, f"{title} (Ordered by Pairwise Dist.)", price_df_for_display, show_outliers=True) 

        print("Cluster ranking by avg. pairwise distance:")
        for rank, (c, d) in enumerate(sorted_clusters, 1):
            display_label = next((k for k, v in cluster_order.items() if v == c), c) 
            print(f"  Rank {rank}: Cluster {display_label} → avg pairwise dist {d:.4f}")

        return new_labels
    return None

def run_kmeans(X, names, out_file="clusters_kmeans.png", title="K-Means Clustering", price_df_for_display=None):
    print("\n--- Running K-Means ---")
    if len(X) <= 1: # ★ Añadido manejo para datos insuficientes
        print("⚠️ Not enough samples for K-Means Clustering. Skipping.")
        return None

    best_k = None
    best_score = -1e9
    
    k_range_end = min(12, len(X) - 1) # ★ Ajustado el rango para k
    if k_range_end < 2: # ★ Asegurarse de que haya al menos 2 clusters posibles
        print("⚠️ Not enough samples to form at least 2 clusters for K-Means. Skipping.")
        return None
        
    for k in range(2, k_range_end + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        if len(set(labels)) > 1: # ★ Asegurarse de que haya más de 1 cluster para Silhouette Score
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_k = k
    if best_k is not None:
        model = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
        labels = model.fit_predict(X)
        # ★ Pasar price_df_for_display a plot_clustered_images
        plot_clustered_images(X, labels, names, out_file, f"{title} (k={best_k})", price_df_for_display)
        print(f"K-Means with k={best_k} saved to {out_file}")
        return labels
    else: # ★ Manejo explícito si no se encuentra un k óptimo
        print("⚠️ K-Means: Could not determine an optimal k. Skipping.")
        return None

def run_hdbscan(X, names, out_file="clusters_hdbscan.png", title="HDBSCAN Clustering", price_df_for_display=None):
    print("\n--- Running HDBSCAN ---")
    if not HAS_HDBSCAN:
        print("HDBSCAN is not installed. Please install with 'pip install hdbscan' to use this method.")
        return None
    if len(X) < 2: # ★ Añadido manejo para datos insuficientes
        print("⚠️ Not enough samples for HDBSCAN Clustering. Skipping.")
        return None

    # # ★ Usar métrica euclidean si no es precomputada
    model_hdbscan = hdbscan.HDBSCAN(
        min_cluster_size=max(2, int(len(X) * 0.02)), # ★ Ajustar min_cluster_size
        min_samples=1, # ★ min_samples ajustado para ser más flexible
        cluster_selection_epsilon=0.0, 
        metric="euclidean" # ★ Usar euclidean directamente con X
    )
    labels = model_hdbscan.fit_predict(X) # ★ Pasar X directamente, no dist_matrix

    # ★ Pasar price_df_for_display a plot_clustered_images
    plot_clustered_images(X, labels, names, out_file, title, price_df_for_display, show_outliers=True)

    print(f"HDBSCAN clustering saved to {out_file}")
    print(f"Number of clusters found by HDBSCAN (including outliers -1): {len(set(labels))}")
    print(f"Number of outliers (-1 label): {np.sum(labels == -1)}")
    return labels


# --- Load Transformer-based Model for Embeddings ---

use_vit = False
vit_model = None
device = None

MODEL_NAME = 'vit_base_patch16_224.mae'

if HAS_TORCH:
    try:
        vit_model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vit_model.to(device).eval()
        use_vit = True
        print(f"Loaded {MODEL_NAME} pretrained for embeddings.")
    except Exception as e:
        print(f"Could not load {MODEL_NAME} pretrained (falling back to classical features).", e)
        use_vit = False

# --- Extract Features (from ALL images) ---
hog_embs_full = [] # ★ "_full" para indicar que son de todas las imágenes
color_embs_full = []
shape_embs_full = []
names_full = [] 
vit_embs_full = []
texture_embs_full = []
preprocessed_images = {} # Stores all preprocessed images for display later

print("\nProcessing ALL images and extracting features (for full Transformer context)...")
for p in all_image_paths_list: 
    try:
        img = load_and_preprocess(p, target=(224, 224))
        names_full.append(p.name) 

        preprocessed_images[p.name] = img # Store for later display

        hog_embs_full.append(hog_features(img))
        color_embs_full.append(color_hist_features(img))
        shape_embs_full.append(shape_properties_features(img))
        texture_embs_full.append(texture_gabor_features(img))
        
        if use_vit:
            vit_embs_full.append(vit_embedding(img, vit_model, device))

    except Exception as e:
        print(f"Error processing {p.name}: {e}")

# IMPORTANT: Check if any embeddings were generated before vstack
if not hog_embs_full:
    raise SystemExit("❌ No HOG embeddings generated from all images. Exiting.")

hog_matrix_full = np.vstack(hog_embs_full)
color_matrix_full = np.vstack(color_embs_full)
shape_matrix_full = np.vstack(shape_embs_full)
texture_matrix_full = np.vstack(texture_embs_full)


vit_scaled_full = None  

if use_vit and len(vit_embs_full) > 0:
    vit_matrix_full = np.vstack(vit_embs_full)  
    scaler_vit = StandardScaler()
    vit_scaled_full = scaler_vit.fit_transform(vit_matrix_full)
    print(f"ViT matrix full shape: {vit_matrix_full.shape}")
else:
    print("⚠️ No ViT embeddings available, skipping ViT features.")

# --- Filter by Price and Brand (AFTER all embeddings are created) ---
print("\nSubsetting embeddings for clustering based on price and brand filter...")

# ★ Filtra price_df_full primero para obtener los productos que cumplen los criterios
price_df_filtered = price_df_full[
    (price_df_full["price_eur"] >= min_price) & 
    (price_df_full["price_eur"] <= max_price)
].reset_index(drop=True)

# ★ Mapeo de product_name (stem) a su índice en los embeddings 'full'
name_to_idx = {Path(name_full).stem: i for i, name_full in enumerate(names_full)}

# ★ Obtener los índices de los productos que han pasado el filtro de precio y marca
indices_for_clustering = []
names_for_clustering = [] # Nombres de las imágenes que serán clusterizadas
for product_stem in price_df_filtered["product_name"].unique():
    if product_stem in name_to_idx:
        idx = name_to_idx[product_stem]
        indices_for_clustering.append(idx)
        names_for_clustering.append(names_full[idx]) # Usar el nombre de archivo completo

if not indices_for_clustering:
    raise SystemExit("❌ No images matched the price/brand filter. Cannot perform clustering. Please check your CSV data or price range/brands.")

print(f"✅ {len(indices_for_clustering)} images passed the price and brand filter for clustering.")


# ★ Seleccionar los embeddings correspondientes a las imágenes filtradas
hog_matrix = hog_matrix_full[indices_for_clustering]
color_matrix = color_matrix_full[indices_for_clustering]
shape_matrix = shape_matrix_full[indices_for_clustering]
texture_matrix = texture_matrix_full[indices_for_clustering]

vit_scaled = None
if vit_scaled_full is not None:
    vit_scaled = vit_scaled_full[indices_for_clustering]


# --- FINAL PART: Combine Features and Run Clustering (on filtered data) ---

shape_factor = 4.0   
color_factor = 1.0
texture_factor = 1.0


# Shape = HOG + geometric features
shape_combined_matrix = np.hstack([hog_matrix, shape_matrix])
print(f"Raw Shape feature matrix shape (filtered): {shape_combined_matrix.shape}")

# Standardize shape features
scaler_shape = StandardScaler()
shape_scaled = scaler_shape.fit_transform(shape_combined_matrix)

# --- PCA optimization for Shape features ---
possible_components = [16, 32, 48, 64]
best_score = -1
best_n = 0
best_shape_pca = None

print("\nOptimizing PCA components for shape features (filtered data)...")
if len(shape_scaled) > 3 and shape_scaled.shape[1] >= 2: # Ensure enough data for PCA
    for n in possible_components:
        n = min(n, shape_scaled.shape[1] - 1)
        if n < 2:
            continue
        pca_temp = PCA(n_components=n)
        shape_pca_temp = pca_temp.fit_transform(shape_scaled)

        # ★ Asegurarse de tener suficientes clusters y samples para silhouette
        if len(shape_pca_temp) > 1 and len(set(KMeans(n_clusters=min(3, len(shape_pca_temp)-1), random_state=42, n_init="auto").fit_predict(shape_pca_temp))) > 1:
            km = KMeans(n_clusters=min(3, len(shape_pca_temp)-1), random_state=42, n_init="auto").fit(shape_pca_temp)
            score = silhouette_score(shape_pca_temp, km.labels_)
            print(f"n_components={n} → Silhouette Score: {score:.4f}")

            if score > best_score:
                best_score = score
                best_n = n
                best_shape_pca = shape_pca_temp

if best_n == 0 and shape_scaled.shape[1] > 1: # ★ Ajuste de fallback
    best_n = min(32, shape_scaled.shape[1] - 1)
    if best_n >= 1: # ★ Asegurarse de que n_components sea al menos 1
        pca = PCA(n_components=best_n)
        shape_pca = pca.fit_transform(shape_scaled)
        print(f"Using fallback PCA with n_components={best_n}")
    else: # ★ Caso extremo si no se puede PCA
        shape_pca = shape_scaled
        print("Could not perform PCA with sufficient components, using original scaled shape features.")
elif best_shape_pca is not None:
    shape_pca = best_shape_pca
    print(f"✅ Best PCA n_components = {best_n} (Silhouette Score = {best_score:.4f})")
else:
    shape_pca = shape_scaled
    print("Could not perform PCA, using original scaled shape features.")


# --- Scale other features ---
scaler_color = StandardScaler()
color_scaled = scaler_color.fit_transform(color_matrix)

scaler_texture = StandardScaler()  
texture_scaled = scaler_texture.fit_transform(texture_matrix) 



# --- Build feature sets ---
if vit_scaled is not None:
    features_vit = vit_scaled
    features_shape_color_texture = np.hstack([
        shape_pca * shape_factor,
        color_scaled * color_factor,
        texture_scaled * texture_factor
    ])
else:
    features_vit = None # Explicitamente None si no hay VIT
    features_shape_color_texture = np.hstack([
        shape_pca * shape_factor,
        color_scaled * color_factor,
        texture_scaled * texture_factor
    ])

if features_vit is not None:
    print(f"VIT feature matrix shape (filtered): {features_vit.shape}")
else:
    print("VIT feature matrix not available for filtered data.")

print(f"Shape+Color+Texture feature matrix shape (filtered): {features_shape_color_texture.shape}")


# --- Run clustering on VIT (filtered data) ---
if features_vit is not None and len(features_vit) > 0: 
    print("\n🔹 Running clustering on VIT features (filtered data)...")

    labels_agg_VIT = run_agglomerative(features_vit, names_for_clustering, price_df_for_display=price_df_filtered)
    if labels_agg_VIT is not None:
        plot_dendrogram(
            features_vit,
            names_for_clustering, 
            "dendrogram_vit_filtered.png",
            "Agglomerative Clustering Dendrogram (ViT Features - Filtered)"
        )
        
    labels_kmeans_VIT = run_kmeans(features_vit, names_for_clustering, price_df_for_display=price_df_filtered)
    labels_hdbscan_VIT = run_hdbscan(features_vit, names_for_clustering, price_df_for_display=price_df_filtered)
else:
    print("\n⚠️ Skipping VIT feature clustering as no VIT embeddings are available for filtered data or there are not enough samples.")

# --- Run clustering on Shape  + Color + Texture (filtered data) ---
print("\n🔹 Running clustering on Shape+Color+Texture features (filtered data)...")

if len(features_shape_color_texture) > 0:
    labels_agg_shape_color_texture = run_agglomerative(
        features_shape_color_texture,
        names_for_clustering, 
        out_file="clusters_agglomerative_shape_color_texture_filtered.png",
        title="Agglomerative Clustering (Shape+Color+Texture - Filtered)",
        price_df_for_display=price_df_filtered 
    )
    if labels_agg_shape_color_texture is not None:
        plot_dendrogram(
            features_shape_color_texture,
            names_for_clustering, 
            "dendrogram_shape_color_texture_filtered.png",
            "Agglomerative Clustering Dendrogram (Shape+Color+Texture Features - Filtered)"
        )

    labels_kmeans_shape_color_texture = run_kmeans(
        features_shape_color_texture,
        names_for_clustering, 
        out_file="clusters_kmeans_shape_color_texture_filtered.png",
        title="K-Means Clustering (Shape+Color+Texture - Filtered)",
        price_df_for_display=price_df_filtered 
    )

    labels_hdbscan_shape_color_texture = run_hdbscan(
        features_shape_color_texture,
        names_for_clustering, 
        out_file="clusters_hdbscan_shape_color_texture_filtered.png",
        title="HDBSCAN Clustering (Shape+Color+Texture - Filtered)",
        price_df_for_display=price_df_filtered 
    )
else:
    print("\n⚠️ Skipping Shape+Color+Texture feature clustering as there are not enough samples in filtered data.")

print("\n✅ Clustering pipeline completed (ViT and Shape+Color+Texture) on filtered data.")
