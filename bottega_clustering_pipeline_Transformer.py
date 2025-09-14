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

# Optional libraries
try:
    import timm, torch, torchvision
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False
    
try:
    import hdbscan
    HAS_HDBSCAN = True
except Exception:
    HAS_HDBSCAN = False

try:
    from skimage.feature import hog
    from skimage.color import rgb2gray
    from skimage.measure import regionprops, moments, moments_central, moments_hu
    # Library for dendrogram
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    HAS_SKIMAGE = True
    HAS_SCIPY = True
except Exception:
    HAS_SKIMAGE = False
    HAS_SCIPY = False

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
    
    # Add a small buffer to the crop box to ensure no part of the object is cut off
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
def vit_embedding(img_pil, model, device):
    """Extracts a feature vector using a pretrained ViT model."""
    import torchvision.transforms as T
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
        return np.array([])
    
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
    
    # Disable tight_layout to handle large number of axes
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
        if n_images == 0:
            continue
        
        rows = 1
        cols = min(10, n_images)
        
        img_h = 1.0 / n_clusters
        img_w = 1.0 / cols
        
        for j, img_name in enumerate(cluster_images):
            try:
                img_path = IMAGES_DIR / img_name
                img = Image.open(img_path)
                
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
def run_agglomerative(X, names):
    """
    Runs Agglomerative Clustering and generates plots for k=2, 3, and 4.
    """
    print("\n--- Running Agglomerative ---")
    
    # Generate the linkage matrix first
    linked_matrix = linkage(X, method='ward')
    
    # Plot for k=2
    labels_2 = fcluster(linked_matrix, 2, criterion='maxclust')
    plot_clustered_images(X, labels_2, names, "clusters_agglomerative_k2.png", "Agglomerative Clustering (k=2)")
    print(f"Agglomerative Clustering with k=2 saved to clusters_agglomerative_k2.png")
    
    # Plot for k=3
    labels_3 = fcluster(linked_matrix, 3, criterion='maxclust')
    plot_clustered_images(X, labels_3, names, "clusters_agglomerative_k3.png", "Agglomerative Clustering (k=3)")
    print(f"Agglomerative Clustering with k=3 saved to clusters_agglomerative_k3.png")

    # Plot for k=4
    labels_4 = fcluster(linked_matrix, 4, criterion='maxclust')
    plot_clustered_images(X, labels_4, names, "clusters_agglomerative_k4.png", "Agglomerative Clustering (k=4)")
    print(f"Agglomerative Clustering with k=4 saved to clusters_agglomerative_k4.png")

    return labels_3 # Return labels for the most balanced cluster, k=3

# --- Old clustering functions with auto k removed ---
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
    
    model_hdbscan = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=2, cluster_selection_epsilon=0.5)
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

# Load ViT model
use_vit = False
vit_model = None
device = None
if HAS_TORCH:
    try:
        vit_model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vit_model.to(device).eval()
        use_vit = True
        print("Loaded ViT pretrained for embeddings.")
    except Exception as e:
        print("Could not load ViT pretrained (falling back to classical features).", e)
        use_vit = False

# Extract all features first
hog_embs = []
color_embs = []
shape_embs = []
names = []

print("Processing images and extracting features...")
for p in img_files:
    try:
        img = load_and_preprocess(p, target=(224, 224))
        names.append(p.name)

        hog_embs.append(hog_features(img))
        color_embs.append(color_hist_features(img))
        shape_embs.append(shape_properties_features(img))
    except Exception as e:
        print(f"Error processing {p.name}: {e}")

hog_matrix = np.vstack(hog_embs)
color_matrix = np.vstack(color_embs)
shape_matrix = np.vstack(shape_embs)

# Combine shape features (HOG + Hu Moments + Geometric properties)
shape_combined_matrix = np.hstack([hog_matrix, shape_matrix])

# Standardize and PCA for combined Shape features
scaler_shape = StandardScaler()
shape_scaled = scaler_shape.fit_transform(shape_combined_matrix)
pca_shape = PCA(n_components=min(64, shape_scaled.shape[1]))
shape_pca = pca_shape.fit_transform(shape_scaled)
print(f"Combined Shape PCA shape: {shape_pca.shape}")

# Standardize and PCA for Color
scaler_color = StandardScaler()
color_scaled = scaler_color.fit_transform(color_matrix)
pca_color = PCA(n_components=min(64, color_scaled.shape[1]))
color_pca = pca_color.fit_transform(color_scaled)
print(f"Color PCA shape: {color_pca.shape}")

# --- EXECUTE ALL CLUSTERING METHODS ---
run_kmeans(shape_pca, names)
run_agglomerative(shape_pca, names)
run_hdbscan(shape_pca, names)

# Generate Dendrogram for Agglomerative Clustering
if HAS_SCIPY:
    plot_dendrogram(shape_pca, names, "dendrogram_agglomerative.png", "Agglomerative Clustering Dendrogram (Shape-based)")