import time, re, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Libraries for clustering and dimensionality reduction
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
from PIL import Image, ImageOps

# Libraries for autoencoder
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import transforms
    from torch.utils.data import Dataset, DataLoader
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
    img = Image.open(p).convert("RGB")
    img = crop_white_bg(img)
    bg = Image.new("RGB", target, (255, 255, 255))
    img_contained = ImageOps.contain(img, target)
    paste_x = (target[0] - img_contained.width) // 2
    paste_y = (target[1] - img_contained.height) // 2
    bg.paste(img_contained, (paste_x, paste_y))
    return bg

# --- NEW: Training and Embedding with DeepGrayAutoencoder ---

# Dataset para el entrenamiento
class ShapeImageDataset(Dataset):
    def __init__(self, folders, transform_gray):
        self.paths = []
        for folder in folders:
            folder_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            self.paths.extend(folder_paths)
        self.transform_gray = transform_gray

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        img_np = np.array(img)
        mask = (img_np != 255).any(axis=2)
        coords = np.argwhere(mask)
        if coords.size > 0:
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0)
            img = img.crop((x0, y0, x1, y1))
        img = img.resize((128, 128))
        gray_img = img.convert("L")
        return self.transform_gray(gray_img)

# El autoencoder completo
class DeepGrayEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.ReLU(),   # 64x64
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),  # 32x32
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(), # 16x16
            nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(),# 8x8
            nn.Conv2d(256, 512, 3, 2, 1), nn.ReLU(),# 4x4
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
    def forward(self, x):
        return self.encoder(x)

class DeepGrayDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(512, 4*4*512)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, 2, 1, 1), nn.ReLU(), # 8x8
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1), nn.ReLU(),# 16x16
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1), nn.ReLU(), # 32x32
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1), nn.ReLU(), # 64x64
            nn.ConvTranspose2d(32, 1, 3, 2, 1, 1), nn.Sigmoid() # 128x128
        )
    def forward(self, x):
        x = self.fc(x).view(-1, 512, 4, 4)
        return self.decoder(x)

class DeepGrayAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = DeepGrayEncoder()
        self.decoder = DeepGrayDecoder()
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

def train_autoencoder(folders, num_epochs=50, learning_rate=1e-3, output_path="autoencoder_forma.pth"):
    if not HAS_TORCH:
        print("PyTorch no está instalado. No se puede entrenar el autoencoder.")
        return None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando el dispositivo: {device}")
    
    transform_gray = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    dataset = ShapeImageDataset(folders, transform_gray)
    if len(dataset) == 0:
        print(f"No se encontraron imágenes en las carpetas de entrenamiento: {folders}")
        return None
    
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    model = DeepGrayAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"🚀 Iniciando entrenamiento con {len(dataset)} imágenes...")
    for epoch in range(num_epochs):
        for data in loader:
            img_gray = data.to(device)
            reconstructed = model(img_gray)
            loss = criterion(reconstructed, img_gray)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Época [{epoch+1}/{num_epochs}], Pérdida: {loss.item():.4f}')
    
    torch.save(model.encoder.state_dict(), output_path)
    print(f"✅ Entrenamiento completado. Modelo del encoder guardado en {output_path}")
    return output_path

def get_deep_encoder_features(img_path, model, transform, device):
    """Extracts a feature vector using a pretrained DeepGrayEncoder model."""
    if not HAS_TORCH:
        return np.array([])
    img = load_and_preprocess(img_path, target=(128, 128))
    img_gray = img.convert("L")
    x = transform(img_gray).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(x).cpu().numpy().squeeze()
        return feat

# --- Plotting Functions ---
def plot_clustered_images(emb_2d, labels, names, out_file, title, show_outliers=False):
    """Creates a grid of images, organized by cluster, with colored borders."""
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
                ax_img = cluster_ax.inset_axes([j * img_w, 0, img_w, 1], transform=cluster_ax.transAxes)
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
    """Plots a dendrogram for hierarchical clustering."""
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
def run_kmeans(X, names):
    print("\n--- Running K-Means ---")
    best_k, best_score = None, -1e9
    for k in range(2, min(12, len(X) - 1)):
        km = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = km.fit_predict(X)
        if len(set(labels)) > 1:
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score, best_k = score, k
    if best_k is not None:
        model = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
        labels = model.fit_predict(X)
        plot_clustered_images(X, labels, names, f"clusters_kmeans.png", f"K-Means Clustering (k={best_k})")
        return labels
    return None

def run_agglomerative(X, names):
    print("\n--- Running Agglomerative ---")
    best_k, best_score = None, -1e9
    for k in range(2, min(12, len(X) - 1)):
        agg = AgglomerativeClustering(n_clusters=k)
        labels = agg.fit_predict(X)
        if len(set(labels)) > 1:
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score, best_k = score, k
    if best_k is not None:
        model = AgglomerativeClustering(n_clusters=best_k)
        labels = model.fit_predict(X)
        plot_clustered_images(X, labels, names, f"clusters_agglomerative.png", f"Agglomerative Clustering (k={best_k})")
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

# --- MAIN PIPELINE ---
if __name__ == '__main__':
    TRAINING_FOLDERS = ["images_ysl","images_D&G","images_bottega"]  # <-- Ponga aquí los nombres de sus carpetas de entrenamiento
    IMAGES_DIR = Path("images_bottega")          # Carpeta para el clustering

    # 1. Entrenar el autoencoder y guardar el modelo
    MODEL_PATH = train_autoencoder(TRAINING_FOLDERS)

    if MODEL_PATH:
        # 2. Cargar el encoder entrenado para extraer embeddings
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        deep_encoder = DeepGrayEncoder().to(device)
        deep_encoder.load_state_dict(torch.load(MODEL_PATH))
        deep_encoder.eval()
        transform_gray = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # Obtener las rutas de las imágenes para el clustering
        img_files = sorted([p for p in IMAGES_DIR.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")])
        if len(img_files) == 0:
            raise SystemExit(f"No se encontraron imágenes en {IMAGES_DIR} — agrega imágenes y vuelve a correr el script.")

        # 3. Extraer los embeddings de forma
        shape_embs = []
        names = []
        print("\nProcessing images and extracting features with trained encoder...")
        for p in img_files:
            try:
                names.append(p.name)
                shape_embs.append(get_deep_encoder_features(p, deep_encoder, transform_gray, device))
            except Exception as e:
                print(f"Error processing {p.name}: {e}")

        shape_matrix = np.vstack(shape_embs)
        
        # Estandarizar y reducir dimensionalidad
        scaler_shape = StandardScaler()
        shape_scaled = scaler_shape.fit_transform(shape_matrix)
        pca_shape = PCA(n_components=min(64, shape_scaled.shape[1]))
        shape_pca = pca_shape.fit_transform(shape_scaled)
        print(f"Shape PCA shape: {shape_pca.shape}")

        # 4. Ejecutar todos los métodos de clustering y generar los gráficos
        run_kmeans(shape_pca, names)
        run_agglomerative(shape_pca, names)
        run_hdbscan(shape_pca, names)
        
        # 5. Generar el dendrograma
        if HAS_SCIPY:
            plot_dendrogram(shape_pca, names, "dendrogram_agglomerative.png", "Agglomerative Clustering Dendrogram (Shape-based)")