import os
import glob
import torch
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
import csv
import plotly.express as px
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import random
import shutil

# Define the MoCo model
class MoCoModel(nn.Module):
    def __init__(self, base_model, projection_dim=128):
        super(MoCoModel, self).__init__()
        self.encoder = base_model
        self.projection_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        return h, z

# Feature extraction function with February filter
def extract_features(encoder, image_dir, batch_size=16, device='cuda', num_workers=4, prefix='202102'):
    encoder.eval()
    transform = T.Compose([
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Gather all image paths
    image_paths = sorted(
        glob.glob(os.path.join(image_dir, "*.jpg")) +
        glob.glob(os.path.join(image_dir, "*.jpeg")) +
        glob.glob(os.path.join(image_dir, "*.png"))
    )
    # Filter only February images (prefix '202102')
    image_paths = [p for p in image_paths if os.path.basename(p).startswith(prefix)]
    if not image_paths:
        raise FileNotFoundError(f"No February images found in {image_dir} with prefix {prefix}")

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
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers
    )

    all_features = []
    all_paths = []

    for images, paths in tqdm(data_loader, desc="Extracting features", unit="batch"):
        images = images.to(device)
        with torch.no_grad():
            features, _ = encoder(images)
        features = features.cpu().numpy()
        all_features.append(features)
        all_paths.extend(paths)

    all_features = np.concatenate(all_features, axis=0)
    return all_paths, all_features

# Clustering function (3 clusters)
def cluster_features(features, num_clusters=3, random_state=42):
    kmeans = KMeans(
        n_clusters=num_clusters,
        init="k-means++",
        n_init=20,
        random_state=random_state,
        verbose=1
    )
    cluster_labels = kmeans.fit_predict(features)
    return kmeans, cluster_labels

# Save clusters to CSV
def save_clusters_to_csv(image_paths, cluster_labels, output_csv="moco_clusters_02.csv"):
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

# Save sample images
def save_sample_images(image_paths, cluster_labels, sample_folder="sample_images_02"):
    if not os.path.exists(sample_folder):
        os.makedirs(sample_folder)

    cluster_dict = {}
    for path, label in zip(image_paths, cluster_labels):
        cluster_dict.setdefault(label, []).append(path)

    for label, paths in cluster_dict.items():
        random_image_path = random.choice(paths)
        image_name = os.path.basename(random_image_path)
        new_image_name = f"cluster_{label}_{image_name}"
        shutil.copy(random_image_path, os.path.join(sample_folder, new_image_name))
    print(f"One random image per cluster saved to {sample_folder}")

# Visualize clusters with t-SNE
def visualize_clusters(features, cluster_labels, image_paths):
    tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
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
        title="Interactive t-SNE Visualization of MoCo Image Clusters",
        labels={"color": "Cluster"}
    )

    fig.update_layout(legend_title_text="Cluster Label")
    fig.write_html("moco_interactive_tsne_plot.html")
    print("Interactive t-SNE plot saved as moco_interactive_tsne_plot.html")

# Main function
def main():
    checkpoint_path = "moco_checkpoints/checkpoints_v1/moco_epoch_85.pt"
    image_dir = "/ceph/project/P4-concept-drift/YOLOv8-Anton/data/cropped_images/train"
    num_clusters = 3
    batch_size = 128
    num_workers = 4
    output_csv = "moco_clusters_02.csv"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    base_model.fc = nn.Identity()

    model = MoCoModel(base_model).to(device)

    print(f"Loading MoCo checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'query_model_state_dict' in checkpoint:
        state_dict = checkpoint['query_model_state_dict']
    else:
        state_dict = checkpoint['model_state_dict']

    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()
    print("MoCo model loaded successfully")

    print("Starting feature extraction for February images...")
    image_paths, features = extract_features(
        model,
        image_dir,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
        prefix='202102'
    )
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.clip(norm, a_min=1e-9, a_max=None)

    print("Starting KMeans clustering...")
    kmeans_model, cluster_labels = cluster_features(features, num_clusters=num_clusters)

    save_clusters_to_csv(image_paths, cluster_labels, output_csv=output_csv)
    print_cluster_distribution(cluster_labels)
    save_sample_images(image_paths, cluster_labels)

    print("Visualizing clusters with t-SNE...")
    visualize_clusters(features, cluster_labels, image_paths)

if __name__ == "__main__":
    main()
