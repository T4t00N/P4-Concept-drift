import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

# Custom Dataset for your single-channel images
class ImageDataset(Dataset):
    def __init__(self, image_paths):  # image_paths: list of file paths to your images
        self.image_paths = image_paths
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # Ensure single-channel
            transforms.Resize((64, 64)),  # Resize to 64x64 (adjust as needed)
            transforms.ToTensor(),  # Convert to tensor and normalize to [0, 1]
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx])
        img = self.transform(img)
        return img

# Example: Load your 1M images (replace with your actual data loading logic)
# image_paths = ["path/to/image1.png", "path/to/image2.png", ...]  # Your image file list
# dataset = ImageDataset(image_paths)
# dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)

import torch.nn as nn

# Define the Convolutional Autoencoder
class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=32):  # Latent space size (adjustable)
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 16x16 -> 8x8
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, latent_dim)  # 8x8x64 -> latent_dim
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (64, 8, 8)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # 32x32 -> 64x64
            nn.Sigmoid()  # Output in [0, 1]
        )

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent

# Training the Autoencoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvAutoencoder(latent_dim=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Example training loop (replace dataloader with yours)
def train_autoencoder(model, dataloader, epochs=20):
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

# train_autoencoder(model, dataloader)

from sklearn.cluster import KMeans

# Extract latent features
def get_latent_features(model, dataloader):
    model.eval()
    latent_features = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            _, latent = model(batch)
            latent_features.append(latent.cpu().numpy())
    return np.concatenate(latent_features)

# latent_features = get_latent_features(model, dataloader)

# Initialize clusters with K-Means
n_clusters = 10  # Start with 10, adjust later
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
# cluster_labels = kmeans.fit_predict(latent_features)
# centroids = kmeans.cluster_centers_

# DEC Implementation (simplified)
class DEC(nn.Module):
    def __init__(self, encoder, n_clusters, latent_dim):
        super(DEC, self).__init__()
        self.encoder = encoder
        self.cluster_centers = nn.Parameter(torch.tensor(centroids, dtype=torch.float32).to(device))
        self.alpha = 1.0  # Degrees of freedom for Student's t-distribution

    def forward(self, x):
        latent = self.encoder(x)
        # Compute soft assignments (Student's t-distribution)
        q = 1.0 / (1.0 + torch.sum((latent.unsqueeze(1) - self.cluster_centers) ** 2, dim=2) / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = q / q.sum(dim=1, keepdim=True)
        return q

# DEC training
dec_model = DEC(model.encoder, n_clusters, latent_dim=32).to(device)
optimizer = torch.optim.SGD(dec_model.parameters(), lr=0.01, momentum=0.9)

def target_distribution(q):
    p = q ** 2 / q.sum(dim=0)
    return (p.t() / p.sum(dim=1)).t()

def train_dec(dec_model, dataloader, epochs=50):
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            q = dec_model(batch)
            p = target_distribution(q.detach())  # Target distribution
            loss = nn.KLDivLoss(reduction='batchmean')(q.log(), p)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

# train_dec(dec_model, dataloader)