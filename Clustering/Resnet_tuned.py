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


def extract_features(image_dir, batch_size=16, device='cuda', num_workers=4):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model.to(device)

    if device == 'cuda' and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model.eval()

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
    image_paths = sorted(image_paths)

    if len(image_paths) == 0:
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
            if self.transform:
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
            features = model(images)
        features = features.cpu().numpy()
        all_features.append(features)
        all_paths.extend(paths)

    all_features = np.concatenate(all_features, axis=0)
    return all_paths, all_features

def cluster_features(features, num_clusters=100, random_state=42):
    """
    Cluster feature vectors into 'num_clusters' using K-Means.
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state, verbose=1)
    cluster_labels = kmeans.fit_predict(features)
    return kmeans, cluster_labels

def save_clusters_to_csv(image_paths, cluster_labels, output_csv="clusters.csv"):
    """
    Save the clustering results to a CSV file.
    """
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_path", "cluster_label"])
        for path, label in zip(image_paths, cluster_labels):
            writer.writerow([path, label])
    print(f"Clustering results saved to {output_csv}, sir.")

def print_cluster_distribution(cluster_labels):
    """
    Print the distribution of images in each cluster.
    """
    unique, counts = np.unique(cluster_labels, return_counts=True)
    total = len(cluster_labels)
    print("Cluster distribution, sir:")
    for cluster, count in zip(unique, counts):
        percentage = count / total * 100
        print(f"Cluster {cluster}: {percentage:.2f}% images ({count} out of {total})")

def visualize_clusters(features, cluster_labels):
    """
    Use PCA to reduce the feature dimensions to 2 and visualize the clusters.
    """
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1],
                          c=cluster_labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label="Cluster Label")
    plt.title("PCA Visualization of Image Features")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.savefig('PCA plot')

def main():

    image_dir = "/ceph/project/P4-concept-drift/YOLOv8-Anton/data/cropped_images/train"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Starting feature extraction, sir...")
    image_paths, features = extract_features(image_dir, batch_size=200, device=device, num_workers=10)

    print("Starting clustering, sir...")
    kmeans_model, labels = cluster_features(features, num_clusters=50)

    save_clusters_to_csv(image_paths, labels, output_csv="clusters.csv")

    print_cluster_distribution(labels)

    print("Visualizing clusters with PCA")
    visualize_clusters(features, labels)

    print("Sample clustering results")
    for path, label in zip(image_paths[:10], labels[:10]):
        print(f"Image: {os.path.basename(path)} -> Cluster: {label}")


if __name__ == "__main__":
    main()
