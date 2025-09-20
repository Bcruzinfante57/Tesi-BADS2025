import time, re, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Libraries for clustering and dimensionality reduction
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
from PIL import Image, ImageOps

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

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# --- Utilities for Image Processing ---
_saved_first_crop = False
def crop_white_bg(pil_img, white_thresh=235):
    """Crops white background from a PIL image using a more aggressive threshold."""
    img = pil_img.convert("RGB")
    arr = np.array(img)
    mask = np.any(arr < white_thresh, axis=2)
    if not mask.any():
        return img
    
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    
    y0 = max(0, y0 - 5)
    y1 = min(img.height, y1 + 5)
    x0 = max(0, x0 - 5)
    x1 = min(img.width, x1 + 5)
    
    return img.crop((x0, y0, x1, y1))

def load_and_preprocess(p, target=(224, 224)):
    """Loads, crops, and centers an image for model input."""
    global _saved_first_crop
    img = Image.open(p).convert("RGB")
    
    img = crop_white_bg(img)
    
    if not _saved_first_crop:
        img.save("crop_example_1.jpg")
        print("Saved 'crop_example_1.jpg' to check the cropping quality.")
        _saved_first_crop = True

    bg = Image.new("RGB", target, (255, 255, 255))
    img_contained = ImageOps.contain(img, target)
    
    paste_x = (target[0] - img_contained.width) // 2
    paste_y = (target[1] - img_contained.height) // 2
    bg.paste(img_contained, (paste_x, paste_y))
    return bg
    
# --- Feature Extraction with ViT and classical methods ---
_saved_grayscale_example = False

def vit_embedding(img_pil, model, device):
    """Extracts a feature vector using a pretrained ViT model."""
    global _saved_grayscale_example
    
    if not _saved_grayscale_example:
        img_pil.convert('L').save("grayscale_example.jpg")
        print("Saved 'grayscale_example.jpg' to check the grayscale transformation.")
        _saved_grayscale_example = True
        
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
    
    feats[7:] = feats[7:] / np.linalg.norm(feats[7:] + 1e-8)
    
    return feats

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
                # ⭐ MODIFICACIÓN CLAVE: Cargar la imagen preprocesada en lugar de la original ⭐
                # Necesitas volver a preprocesar aquí para que los bordes de la trama coincidan con el gráfico
                img = load_and_preprocess(IMAGES_DIR / img_name, target=(224, 224))
                
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

def run_agglomerative(X, names):
    """Runs Agglomerative Clustering with optimal k."""
    print("\n--- Running Agglomerative ---")
    best_k = find_optimal_k_agglomerative(X)
    if best_k is not None:
        model = AgglomerativeClustering(n_clusters=best_k)
        labels = model.fit_predict(X)
        plot_clustered_images(X, labels, names, f"clusters_agglomerative.png", f"Agglomerative Clustering (Optimal k={best_k})")
        print(f"Agglomerative Clustering with optimal k={best_k} saved to clusters_agglomerative.png")
        return labels
    return None

def run_kmeans(X, names):
    print("\n--- Running K-Means ---")
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
        plot_clustered_images(X, labels, names, f"clusters_kmeans.png", f"K-Means Clustering (k={best_k})")
        return labels
    return None

def run_hdbscan(X, names):
    print("\n--- Running HDBSCAN ---")
    if not HAS_HDBSCAN:
        print("HDBSCAN is not installed. Please install with 'pip install hdbscan' to use this method.")
        return None
    
    model_hdbscan = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=1, cluster_selection_epsilon=0.5)
    labels = model_hdbscan.fit_predict(X)
    
    plot_clustered_images(X, labels, names, f"clusters_hdbscan.png", "HDBSCAN Clustering", show_outliers=True)
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

print("Processing images and extracting features...")
for p in img_files:
    try:
        img = load_and_preprocess(p, target=(224, 224))
        names.append(p.name)

        hog_embs.append(hog_features(img))
        color_embs.append(color_hist_features(img))
        shape_embs.append(shape_properties_features(img))
        
        if use_vit:
            vit_embs.append(vit_embedding(img, vit_model, device))

    except Exception as e:
        print(f"Error processing {p.name}: {e}")

hog_matrix = np.vstack(hog_embs)
color_matrix = np.vstack(color_embs)
shape_matrix = np.vstack(shape_embs)

# --- Combine Features and Run Clustering ---
shape_combined_matrix = np.hstack([hog_matrix, shape_matrix])
print(f"Combined Shape feature matrix shape: {shape_combined_matrix.shape}")

scaler_shape = StandardScaler()
shape_scaled = scaler_shape.fit_transform(shape_combined_matrix)

possible_components = [16, 32, 48, 64]
best_score = -1
best_n = 0
best_shape_pca = None

print("\nOptimizing PCA components for shape features...")
if len(shape_scaled) > 3:
    for n in possible_components:
        n = min(n, shape_scaled.shape[1])
        if n < 2: continue
        pca_temp = PCA(n_components=n)
        shape_pca_temp = pca_temp.fit_transform(shape_scaled)
        
        if len(shape_pca_temp) > 1 and len(set(KMeans(n_clusters=3, random_state=42, n_init='auto').fit_predict(shape_pca_temp))) > 1:
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
    

# Run clustering on Shape Features
labels_agg = run_agglomerative(shape_pca, names)
if labels_agg is not None:
    plot_dendrogram(shape_pca, names, "dendrogram_agglomerative.png", "Agglomerative Clustering Dendrogram (Shape Features)")

labels_kmeans = run_kmeans(shape_pca, names)
labels_hdbscan = run_hdbscan(shape_pca, names)

# If ViT embeddings are used, combine them with classical features
if use_vit and vit_embs:
    try:
        vit_matrix = np.vstack(vit_embs)
        print(f"Transformer Embedding shape: {vit_matrix.shape}")
        
        scaler_vit = StandardScaler()
        vit_scaled = scaler_vit.fit_transform(vit_matrix)
        
        combined_features = np.hstack([shape_pca, vit_scaled])
        print(f"Combined feature matrix shape: {combined_features.shape}")
        
        # Run clustering on combined features
        labels_agg_combined = run_agglomerative(combined_features, names)
        if labels_agg_combined is not None:
            plot_dendrogram(combined_features, names, "dendrogram_agglomerative_combined.png", "Agglomerative Clustering Dendrogram (Combined Features)")
        
        labels_kmeans_combined = run_kmeans(combined_features, names)
        labels_hdbscan_combined = run_hdbscan(combined_features, names)

    except Exception as e:
        print(f"Error combining ViT features: {e}")
