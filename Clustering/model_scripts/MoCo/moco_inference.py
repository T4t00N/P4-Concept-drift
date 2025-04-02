import os
import glob
import torch
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
import csv
import plotly.express as px
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F


# Define the MoCo model (simplified from the training script)
class MoCoModel(nn.Module):
    def __init__(self, base_model, projection_dim=128):
        super(MoCoModel, self).__init__()
        self.encoder = base_model  # ResNet-50 without the final layer
        self.projection_head = nn.Linear(2048, projection_dim)  # Single linear layer as in MoCo

    def forward(self, x):
        h = self.encoder(x)  # Feature representation (2048-dim)
        z = self.projection_head(h)  # Projection for contrastive loss (128-dim)
        return h, z


# Feature extraction function
def extract_features(encoder, image_dir, batch_size=16, device='cuda', num_workers=4):
    encoder.eval()
    transform = T.Compose([
        T.Resize(256),
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
            features, _ = encoder(images)  # Use the feature vector, not the projection
        features = features.cpu().numpy()
        all_features.append(features)
        all_paths.extend(paths)

    all_features = np.concatenate(all_features, axis=0)
    return all_paths, all_features


# Clustering function
def cluster_features(features, num_clusters=100, random_state=42):
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state, verbose=1)
    cluster_labels = kmeans.fit_predict(features)
    return kmeans, cluster_labels


# Save clusters to CSV
def save_clusters_to_csv(image_paths, cluster_labels, output_csv="moco_clusters.csv"):
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_path", "cluster_label"])
        for path, label in zip(image_paths, cluster_labels):
            writer.writerow([path, label])
    print(f"Clustering results saved to {output_csv}")


# Print cluster distribution
def print_cluster_distribution(cluster_labels):
    unique, counts = np.unique(cluster_labels, return_counts=True)
    total = len(cluster_labels)
    print("Cluster distribution:")
    for cluster, count in zip(unique, counts):
        percentage = count / total * 100
        print(f"Cluster {cluster}: {percentage:.2f}% images ({count} out of {total})")


# Visualize clusters with PCA
def visualize_clusters(features, cluster_labels, image_paths):
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)

    df = pd.DataFrame({
        "PCA 1": features_2d[:, 0],
        "PCA 2": features_2d[:, 1],
        "Cluster": cluster_labels,
        "Image Path": image_paths
    })

    fig = px.scatter(
        df, x="PCA 1", y="PCA 2",
        color=df["Cluster"].astype(str),
        hover_data=["Image Path"],
        title="Interactive PCA Visualization of MoCo Image Clusters",
        labels={"color": "Cluster"}
    )

    fig.update_layout(legend_title_text="Cluster Label")
    fig.write_html("moco_interactive_PCA_plot.html")
    print("Interactive PCA plot saved as moco_interactive_PCA_plot.html")


# Main function
def main():
    # Parameters to adjust
    checkpoint_path = "moco_checkpoints/moco_epoch_15.pt"  # Path to MoCo checkpoint
    image_dir = "/ceph/project/P4-concept-drift/YOLOv8-Anton/data/cropped_images/test"  # Directory containing images
    num_clusters = 50  # Number of clusters for K-Means
    batch_size = 200  # Batch size for feature extraction
    num_workers = 4  # Number of workers for data loading
    output_csv = "moco_clusters.csv"  # Output CSV file
    output_html = "moco_interactive_PCA_plot.html"  # Output HTML visualization

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize the base model
    base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    base_model.fc = nn.Identity()  # Remove the final classification layer

    # Initialize the MoCo model
    model = MoCoModel(base_model).to(device)

    # Load the checkpoint
    print(f"Loading MoCo checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Check if checkpoint has query or key model state dict and load accordingly
    if 'query_model_state_dict' in checkpoint:
        state_dict = checkpoint['query_model_state_dict']
    else:
        state_dict = checkpoint['model_state_dict']

    # If the checkpoint was saved with DataParallel, remove 'module.' prefix
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()
    print("MoCo model loaded successfully")

    # Feature extraction
    print("Starting feature extraction...")
    image_paths, features = extract_features(model, image_dir, batch_size=batch_size,
                                             device=device, num_workers=num_workers)

    # Clustering
    print("Starting clustering...")
    kmeans_model, labels = cluster_features(features, num_clusters=num_clusters)

    # Save clusters to CSV
    save_clusters_to_csv(image_paths, labels, output_csv=output_csv)

    # Print cluster distribution
    print_cluster_distribution(labels)

    # Visualize clusters with PCA
    print("Visualizing clusters with PCA...")
    visualize_clusters(features, labels, image_paths)


if __name__ == "__main__":
    main()