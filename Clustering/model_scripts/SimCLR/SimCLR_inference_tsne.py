import os
import glob
import torch
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE  # Replaced PCA with t-SNE
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
import csv
import plotly.express as px
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import random  # For random image selection
import shutil  # For copying images

# Define the SimCLR model (renamed from MoCoModel)
class SimCLRModel(nn.Module):
    def __init__(self, base_model, projection_dim=128):
        super(SimCLRModel, self).__init__()
        self.encoder = base_model  # ResNet-50 without the final layer
        self.projection_head = nn.Sequential(
            nn.Linear(2048, 512),  # First layer (arbitrary hidden size, e.g., 512)
            nn.ReLU(),
            nn.Linear(512, 128)  # Second layer to projection_dim
        )

    def forward(self, x):
        h = self.encoder(x)  # Feature representation (2048-dim)
        z = self.projection_head(h)  # Projection for contrastive loss (128-dim)
        return h, z

# Feature extraction function (unchanged)
def extract_features(encoder, image_dir, batch_size=16, device='cuda', num_workers=4):
    encoder.eval()
    transform = T.Compose([
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")) +
                         glob.glob(os.path.join(image_dir, "*.jpeg")) +
                         glob.glob(os.path.join(image_dir, "*.png")))
    if not image_paths:
        raise FileNotFoundError(f"No image files found in {image_dir}")

    class ImageDataset(torch.utils.data.Dataset):
        def __init__(self, image_paths, transform):
            self.image_paths = image_paths
            self.transform = transform

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, index):
            path = self.image_paths[index]
            image = Image.open(path).convert("RGB")
            image = self.transform(image)
            return image, path

    dataset = ImageDataset(image_paths, transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=num_workers)

    all_features = []
    all_paths = []

    for images, paths in tqdm(data_loader, desc="Extracting features", unit="batch"):
        images = images.to(device)
        with torch.no_grad():
            features, _ = encoder(images)  # Use the feature vector (h), not the projection (z)
        features = features.cpu().numpy()
        all_features.append(features)
        all_paths.extend(paths)

    all_features = np.concatenate(all_features, axis=0)
    return all_paths, all_features

# Clustering function (unchanged)
def cluster_features(features, num_clusters=100, random_state=42):
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state, verbose=1)
    cluster_labels = kmeans.fit_predict(features)
    return kmeans, cluster_labels

# Save clusters to CSV (updated output name)
def save_clusters_to_csv(image_paths, cluster_labels, output_csv="simclr_clusters.csv"):
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_path", "cluster_label"])
        for path, label in zip(image_paths, cluster_labels):
            writer.writerow([path, label])
    print(f"Clustering results saved to {output_csv}")

# Print cluster distribution (unchanged)
def print_cluster_distribution(cluster_labels):
    unique, counts = np.unique(cluster_labels, return_counts=True)
    total = len(cluster_labels)
    print("Cluster distribution:")
    for cluster, count in zip(unique, counts):
        percentage = count / total * 100
        print(f"Cluster {cluster}: {percentage:.2f}% images ({count} out of {total})")

# Save one random image per cluster (new function)
def save_sample_images(image_paths, cluster_labels, sample_folder="sample_images"):
    if not os.path.exists(sample_folder):
        os.makedirs(sample_folder)
    cluster_dict = {}
    for path, label in zip(image_paths, cluster_labels):
        if label not in cluster_dict:
            cluster_dict[label] = []
        cluster_dict[label].append(path)
    for label, paths in cluster_dict.items():
        random_image_path = random.choice(paths)
        image_name = os.path.basename(random_image_path)
        new_image_name = f"cluster_{label}_{image_name}"
        shutil.copy(random_image_path, os.path.join(sample_folder, new_image_name))
    print(f"One random image per cluster saved to {sample_folder}")

# Visualize clusters with t-SNE (replaced PCA with t-SNE)
def visualize_clusters(features, cluster_labels, image_paths):
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)

    df = pd.DataFrame({
        "t-SNE 1": features_2d[:, 0],
        "t-SNE 2": features_2d[:, 1],
        "Cluster": cluster_labels,
        "Image Path": image_paths
    })

    fig = px.scatter(
        df, x="t-SNE 1", y="t-SNE 2",
        color=df["Cluster"].astype(str),
        hover_data=["Image Path"],
        title="Interactive t-SNE Visualization of SimCLR Image Clusters",
        labels={"color": "Cluster"}
    )

    fig.update_layout(legend_title_text="Cluster Label")
    fig.write_html("simclr_interactive_tsne_plot.html")
    print("Interactive t-SNE plot saved as simclr_interactive_tsne_plot.html")

# Main function (updated for SimCLR)
def main():
    # Parameters to adjust
    checkpoint_path = "/ceph/project/P4-concept-drift/YOLOv8-Anton/jens/SimCLR_checkpoints/checkpoints_v4/simclr_epoch_80.pt"  # Path to SimCLR checkpoint
    image_dir = "/ceph/project/P4-concept-drift/YOLOv8-Anton/data/cropped_images/test"  # Directory containing images
    num_clusters = 25  # Number of clusters for K-Means
    batch_size = 128  # Batch size for feature extraction
    num_workers = 4  # Number of workers for data loading
    output_csv = "simclr_clusters.csv"  # Output CSV file
    output_html = "simclr_interactive_tsne_plot.html"  # Output HTML visualization

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize the base model
    base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    base_model.fc = nn.Identity()  # Remove the final classification layer

    # Initialize the SimCLR model
    model = SimCLRModel(base_model).to(device)

    # Load the checkpoint
    print(f"Loading SimCLR checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # SimCLR typically saves the entire model state under 'model_state_dict'
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    # Remove 'module.' prefix if present (e.g., from DataParallel training)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()
    print("SimCLR model loaded successfully")

    # Feature extraction
    print("Starting feature extraction...")
    image_paths, features = extract_features(model, image_dir, batch_size=batch_size,
                                             device=device, num_workers=num_workers)

    # Clustering
    print("Starting clustering...")
    kmeans_model, cluster_labels = cluster_features(features, num_clusters=num_clusters)

    # Save clusters to CSV
    save_clusters_to_csv(image_paths, cluster_labels, output_csv=output_csv)

    # Print cluster distribution
    print_cluster_distribution(cluster_labels)

    # Save one random image per cluster
    save_sample_images(image_paths, cluster_labels)

    # Visualize clusters with t-SNE
    print("Visualizing clusters with t-SNE...")
    visualize_clusters(features, cluster_labels, image_paths)

if __name__ == "__main__":
    main()