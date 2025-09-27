import time, re, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

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

    # Save crop example
    if not _saved_first_crop:
        img.save("example_crop.jpg")
        print("✅ Saved 'example_crop.jpg' (raw crop example).")
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
def plot_clustered_images(emb_2d, labels, names, out_file, title, show_outliers=False):
    """
    Creates a grid of images, organized by cluster, with colored borders.
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
    fig, axes = plt.subplots(n_clusters, 1, figsize=(15, 4 * n_clusters))

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
        img_w = 1.0 / cols
        
        for j, img_name in enumerate(cluster_images):
            try:
                img = preprocessed_images[img_name]
                
                ax_img = cluster_ax.inset_axes(
                    [j * img_w, 0, img_w, 1],
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

            except Exception as e:
                print(f"Error loading image {img_name}: {e}")
        
        cluster_ax.set_title(f"Cluster {label} (n={n_images})", loc='left', fontsize=14, pad=10)
        cluster_ax.set_axis_off()

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
            
    if not scores_list: return None
        
    scores_list.sort(key=lambda x: x[1], reverse=True)
    best_score = scores_list[0][1]
    best_k = scores_list[0][0]
    
    for k_val, score in scores_list:
        if k_val > 4 and score / best_score > 0.95:
            best_k = k_val
            break
            
    return max(4, best_k)

def run_agglomerative(X, names, out_file="clusters_agglomerative.png", title="Agglomerative Clustering"):
    print("\n--- Running Agglomerative ---")
    best_k = find_optimal_k_agglomerative(X)
    if best_k is not None:
        model = AgglomerativeClustering(n_clusters=best_k)
        labels = model.fit_predict(X)

        # Calculate internal dispersion (average distance between pairs)
        dispersions = cluster_dispersion_pairwise(X, labels)

        # Sort clusters from highest to lowest dispersion
        sorted_clusters = sorted(dispersions.items(), key=lambda x: x[1], reverse=True)
        cluster_order = {old: new for new, (old, _) in enumerate(sorted_clusters)}

        # Relabel labels according to order
        new_labels = np.array([cluster_order[l] for l in labels])

       # Ordered plot
        plot_clustered_images(X, new_labels, names, out_file, f"{title} (Ordered by Pairwise Dist.)")

        print("Cluster ranking by avg. pairwise distance:")
        for rank, (c, d) in enumerate(sorted_clusters, 1):
            print(f"  Rank {rank}: Cluster {c} → avg pairwise dist {d:.4f}")

        return new_labels
    return None

def run_kmeans(X, names, out_file="clusters_kmeans.png", title="K-Means Clustering"):
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
        plot_clustered_images(X, labels, names, out_file, f"{title} (k={best_k})")
        print(f"K-Means with k={best_k} saved to {out_file}")
        return labels
    return None

def run_hdbscan(X, names, out_file="clusters_hdbscan.png", title="HDBSCAN Clustering"):
    print("\n--- Running HDBSCAN ---")
    if not HAS_HDBSCAN:
        print("HDBSCAN is not installed. Please install with 'pip install hdbscan' to use this method.")
        return None

    # # Precompute cosine distance matrix in float64
    dist_matrix = pairwise_distances(X, metric="cosine").astype(np.float64)

    # HDBSCAN with precomputed matrix
    model_hdbscan = hdbscan.HDBSCAN(
        min_cluster_size=3,
        min_samples=1,
        cluster_selection_epsilon=0.0,
        metric="precomputed"
    )
    labels = model_hdbscan.fit_predict(dist_matrix)

    # For visualization we continue using X, not dist_matrix
    plot_clustered_images(X, labels, names, out_file, title, show_outliers=True)

    print(f"HDBSCAN clustering saved to {out_file}")
    return labels


# --- Main Pipeline ---
IMAGES_DIR = Path("images_bottega")
if not IMAGES_DIR.exists():
    IMAGES_DIR = Path("imagenes_bottega")
    if not IMAGES_DIR.exists():
        raise SystemExit(f"No se encontró la carpeta {IMAGES_DIR.resolve()} — crea la carpeta y guarda las imágenes allí.")

img_files = sorted([p for p in IMAGES_DIR.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")])
if len(img_files) == 0:
    raise SystemExit("No se encontraron imágenes en images_bottega — agrega imágenes y vuelve a correr el script.")


# --- Load Transformer-based Model for Embeddings ---

use_vit = False
vit_model = None
device = None

# Options: 'vit_base_patch16_224', 'beit_base_patch16_224', 'vit_base_patch16_224.mae'
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

# --- Extract Features ---
hog_embs = []
color_embs = []
shape_embs = []
names = []
vit_embs = []
texture_embs = []
preprocessed_images = {}


print("Processing images and extracting features...")
for p in img_files:
    try:
        img = load_and_preprocess(p, target=(224, 224))
        names.append(p.name)

        preprocessed_images[p.name] = img

        hog_embs.append(hog_features(img))
        color_embs.append(color_hist_features(img))
        shape_embs.append(shape_properties_features(img))
        texture_embs.append(texture_gabor_features(img))
        
        if use_vit:
            vit_embs.append(vit_embedding(img, vit_model, device))

    except Exception as e:
        print(f"Error processing {p.name}: {e}")

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

# --- FINAL PART: Combine Features and Run Clustering ---

shape_factor = 4.0   
color_factor = 1.0
texture_factor = 1.0


# Shape = HOG + geometric features
shape_combined_matrix = np.hstack([hog_matrix, shape_matrix])
print(f"Raw Shape feature matrix shape: {shape_combined_matrix.shape}")

# Standardize shape features
scaler_shape = StandardScaler()
shape_scaled = scaler_shape.fit_transform(shape_combined_matrix)

# --- PCA optimization for Shape features ---
possible_components = [16, 32, 48, 64]
best_score = -1
best_n = 0
best_shape_pca = None

print("\nOptimizing PCA components for shape features...")
if len(shape_scaled) > 3:
    for n in possible_components:
        n = min(n, shape_scaled.shape[1])
        if n < 2:
            continue
        pca_temp = PCA(n_components=n)
        shape_pca_temp = pca_temp.fit_transform(shape_scaled)

        if len(shape_pca_temp) > 1 and len(set(KMeans(n_clusters=3, random_state=42, n_init="auto").fit_predict(shape_pca_temp))) > 1:
            km = KMeans(n_clusters=3, random_state=42, n_init="auto").fit(shape_pca_temp)
            score = silhouette_score(shape_pca_temp, km.labels_)
            print(f"n_components={n} → Silhouette Score: {score:.4f}")

            if score > best_score:
                best_score = score
                best_n = n
                best_shape_pca = shape_pca_temp

if best_n == 0 and shape_scaled.shape[1] > 2:
    best_n = min(32, shape_scaled.shape[1])
    pca = PCA(n_components=best_n)
    shape_pca = pca.fit_transform(shape_scaled)
    print(f"Using fallback PCA with n_components={best_n}")
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
features_vit = vit_scaled
if vit_scaled is not None:
    features_vit = np.hstack([vit_scaled])
    features_shape_color_texture = np.hstack([
        shape_pca * shape_factor,
        color_scaled * color_factor,
        texture_scaled * texture_factor
    ])
else:
    features_vit = None
    features_shape_color_texture = np.hstack([
        shape_pca * shape_factor,
        color_scaled * color_factor,
        texture_scaled * texture_factor
    ])

print(f"VIT feature matrix shape: {features_vit.shape}")

print(f"Shape+Color+Texture feature matrix shape: {features_shape_color_texture.shape}")


# --- Run clustering on VIT ---
print("\n🔹 Running clustering on VIT features...")
labels_agg_VIT = run_agglomerative(features_vit, names)
if labels_agg_VIT is not None:
    plot_dendrogram(
        features_vit,
        names,
        "dendrogram_vit.png",
        "Agglomerative Clustering Dendrogram (ViT Features)"
    )
    
labels_kmeans_VIT = run_kmeans(features_vit, names)
labels_hdbscan_VIT = run_hdbscan(features_vit, names)


# --- Run clustering on Shape  + Color + Texture ---
print("\n🔹 Running clustering on Shape+Color+Texture features...")

labels_agg_shape_color_texture = run_agglomerative(
    features_shape_color_texture,
    names,
    out_file="clusters_agglomerative_shape_color_texture.png",
    title="Agglomerative Clustering (Shape+Color+Texture)"
)
if labels_agg_shape_color_texture is not None:
    plot_dendrogram(
        features_shape_color_texture,
        names,
        "dendrogram_shape_color_texture.png",
        "Agglomerative Clustering Dendrogram (Shape+Color+Texture Features)"
    )

labels_kmeans_shape_color_texture = run_kmeans(
    features_shape_color_texture,
    names,
    out_file="clusters_kmeans_shape_color_texture.png",
    title="K-Means Clustering (Shape+Color+Texture)"
)

labels_hdbscan_shape_color_texture = run_hdbscan(
    features_shape_color_texture,
    names,
    out_file="clusters_hdbscan_shape_color_texture.png",
    title="HDBSCAN Clustering (Shape+Color+Texture)"
)

print("\n✅ Clustering pipeline completed (ViT and Shape+Color+Texture).")