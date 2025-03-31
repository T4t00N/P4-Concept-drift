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
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import plotly.express as px
import pandas as pd


# Define the NT-Xent loss for SimCLR
def nt_xent_loss(z, tau=0.5):
    """
    Compute the NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.
    z: Projections from the model (shape: [2*B, projection_dim])
    tau: Temperature parameter
    """
    B = z.size(0) // 2  # Batch size before concatenation
    z = F.normalize(z, p=2, dim=1)  # L2-normalize the projections
    S = torch.matmul(z, z.T) / tau  # Similarity matrix
    labels = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)  # Positive pair indices
    mask = torch.eye(2 * B, dtype=torch.bool).to(z.device)  # Mask to exclude self-similarity
    S_masked = S.masked_fill(mask, -1e9)  # Set diagonal to a large negative value
    denom = torch.logsumexp(S_masked, dim=1)  # Log-sum-exp over all negatives
    pos_sim = S[torch.arange(2 * B), labels]  # Similarity with positive pairs
    loss = -pos_sim + denom
    return loss.mean()


class SimCLRModel(nn.Module):
    def __init__(self, base_model, projection_dim=128):
        super(SimCLRModel, self).__init__()
        self.encoder = base_model
        self.projection_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x)  # Feature representation (2048-dim)
        z = self.projection_head(h)  # Projection for contrastive loss (128-dim)
        return h, z


# Dataset for SimCLR training
class SimCLRDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        image = Image.open(path).convert("RGB")
        view1 = self.transform(image)
        view2 = self.transform(image)
        return view1, view2


# Modified extract_features to accept an encoder
def extract_features(encoder, image_dir, batch_size=16, device='cuda', num_workers=20):
    """
    Extract features using the provided encoder (e.g., fine-tuned ResNet-50).
    """
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

    for images, paths in tqdm(data_loader, desc="Extracting features, sir", unit="batch"):
        images = images.to(device)
        with torch.no_grad():
            features = encoder(images)
        features = features.cpu().numpy()
        all_features.append(features)
        all_paths.extend(paths)

    all_features = np.concatenate(all_features, axis=0)
    return all_paths, all_features

def cluster_features(features, num_clusters=100, random_state=42):
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state, verbose=1)
    cluster_labels = kmeans.fit_predict(features)
    return kmeans, cluster_labels


def save_clusters_to_csv(image_paths, cluster_labels, output_csv="clusters.csv"):
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_path", "cluster_label"])
        for path, label in zip(image_paths, cluster_labels):
            writer.writerow([path, label])
    print(f"Clustering results saved to {output_csv}, sir.")


def print_cluster_distribution(cluster_labels):
    unique, counts = np.unique(cluster_labels, return_counts=True)
    total = len(cluster_labels)
    print("Cluster distribution, sir:")
    for cluster, count in zip(unique, counts):
        percentage = count / total * 100
        print(f"Cluster {cluster}: {percentage:.2f}% images ({count} out of {total})")

def visualize_clusters(features, cluster_labels, image_paths):
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)

    # Prepare data for interactive plot
    df = pd.DataFrame({
        "PCA 1": features_2d[:, 0],
        "PCA 2": features_2d[:, 1],
        "Cluster": cluster_labels,
        "Image Path": image_paths
    })

    # Interactive plot
    fig = px.scatter(
        df, x="PCA 1", y="PCA 2",
        color=df["Cluster"].astype(str),
        hover_data=["Image Path"],
        title="Interactive PCA Visualization of Image Clusters",
        labels={"color": "Cluster"}
    )

    fig.update_layout(legend_title_text="Cluster Label")
    fig.write_html("interactive_PCA_plot.html")
    print("Interactive PCA plot saved as interactive_PCA_plot.html")

def main():
    image_dir = "/ceph/project/P4-concept-drift/YOLOv8-Anton/data/cropped_images/val"  # Updated to 'val'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Define SimCLR augmentations (no flipping or rotation for static images)
    simclr_transform = T.Compose([
        T.RandomResizedCrop(224, scale=(0.2, 1.0)),
        T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0, hue=0)], p=0.8),
        T.RandomApply([T.GaussianBlur(kernel_size=23)], p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")) +
                         glob.glob(os.path.join(image_dir, "*.jpeg")) +
                         glob.glob(os.path.join(image_dir, "*.png")))
    if not image_paths:
        raise FileNotFoundError(f"No image files found in {image_dir}")

    simclr_dataset = SimCLRDataset(image_paths, simclr_transform)
    simclr_loader = torch.utils.data.DataLoader(simclr_dataset, batch_size=300,
                                                shuffle=True, num_workers=20)

    base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    base_model.fc = nn.Identity()
    model = SimCLRModel(base_model).to(device)
    if device == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

    # SimCLR training loop
    num_epochs = 100
    print("Starting SimCLR training, sir...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for views1, views2 in tqdm(simclr_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"):
            views = torch.cat([views1, views2], dim=0).to(device)
            optimizer.zero_grad()
            _, z = model(views)
            loss = nt_xent_loss(z)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(simclr_loader)
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}, sir")

        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }
            checkpoint_path = f"simclr_epoch_{epoch + 1}.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}, sir.")

    encoder = model.encoder if not isinstance(model, nn.DataParallel) else model.module.encoder

    print("Starting feature extraction with fine-tuned encoder, sir...")
    image_paths, features = extract_features(encoder, image_dir, batch_size=200,
                                             device=device, num_workers=10)

    print("Starting clustering")
    kmeans_model, labels = cluster_features(features, num_clusters=50)

    save_clusters_to_csv(image_paths, labels, output_csv="clusters.csv")

    print_cluster_distribution(labels)
    print("Visualizing clusters with PCA")
    visualize_clusters(features, labels, image_paths)

    # Example: Print the first 10 images and their cluster labels
    print("Sample clustering results")
    for path, label in zip(image_paths[:10], labels[:10]):
        print(f"Image: {os.path.basename(path)} -> Cluster: {label}")

if __name__ == "__main__":
    main()
