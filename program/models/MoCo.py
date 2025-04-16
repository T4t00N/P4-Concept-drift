import os
import glob
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import wandb

# MoCo loss function (unchanged)
def moco_loss(q, k, queue, tau=0.07):  # Updated tau to 0.07
    B = q.size(0)
    pos_logits = (q * k).sum(dim=1) / tau  # [B]
    neg_logits = torch.matmul(q, queue.T) / tau  # [B, K]
    logits = torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1)  # [B, 1 + K]
    labels = torch.zeros(B, dtype=torch.long).to(q.device)
    loss = F.cross_entropy(logits, labels)
    return loss

# Update key encoder with momentum (unchanged)
def update_key_model(query_model, key_model, m=0.999):
    if isinstance(query_model, nn.DataParallel):
        query_model = query_model.module
    if isinstance(key_model, nn.DataParallel):
        key_model = key_model.module
    for param_q, param_k in zip(query_model.parameters(), key_model.parameters()):
        param_k.data = m * param_k.data + (1 - m) * param_q.data

# Model definition (modified projection head)
class SimCLRModel(nn.Module):  # Name retained for compatibility, but now MoCo v1-style
    def __init__(self, base_model, projection_dim=128):
        super(SimCLRModel, self).__init__()
        self.encoder = base_model
        self.projection_head = nn.Sequential(
            nn.Linear(2048, 512),  # First layer (arbitrary hidden size, e.g., 512)
            nn.ReLU(),
            nn.Linear(512, 128)  # Second layer to projection_dim
        ) # Single linear layer

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        return h, z

# Dataset (unchanged)
class SimCLRDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform, step=1):
        self.image_paths = image_paths[::step]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        image = Image.open(path).convert("RGB")
        view1 = self.transform(image)
        view2 = self.transform(image)
        return view1, view2

def main():
    # Initialize WandB for MoCo (updated config)
    wandb.init(project="MoCo", config={
        "learning_rate": 0.001,          # Updated to match MoCo v1
        "epochs": 100,
        "batch_size": 256,
        "optimizer": "SGD",             # Updated to SGD
        "model": "ResNet50 with MoCo",
        "queue_size": 65536,
        "momentum": 0.999,
        "tau": 0.07,                    # Updated to 0.07
    })

    image_dir = "/ceph/project/P4-concept-drift/YOLOv8-Anton/data/cropped_images/train"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Augmentations (updated to match MoCo v1)
    simclr_transform = T.Compose([
        T.RandomResizedCrop(224, scale=(0.2, 1.0)),
        T.RandomApply([T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5), # Simulate sensor blur
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load image paths (unchanged)
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")) +
                         glob.glob(os.path.join(image_dir, "*.jpeg")) +
                         glob.glob(os.path.join(image_dir, "*.png")))
    if not image_paths:
        raise FileNotFoundError(f"No image files found in {image_dir}")

    # Create dataset and loader (unchanged)
    simclr_dataset = SimCLRDataset(image_paths, simclr_transform, step=5)
    simclr_loader = torch.utils.data.DataLoader(simclr_dataset, batch_size=wandb.config.batch_size,
                                                shuffle=True, num_workers=16)
    print(f"Selected {len(simclr_dataset)} images for training.")

    # Initialize models (unchanged except projection head)
    base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    base_model.fc = nn.Identity()
    query_model = SimCLRModel(base_model).to(device)
    key_model = SimCLRModel(base_model).to(device)
    key_model.load_state_dict(query_model.state_dict())  # Copy weights to key_model

    # Handle multi-GPU (unchanged)
    if device == 'cuda' and torch.cuda.device_count() > 1:
        query_model = nn.DataParallel(query_model)
        key_model = nn.DataParallel(key_model)

    # Set key_model to eval mode (unchanged)
    key_model.eval()

    # Optimizer for query_model only (changed to SGD)
    optimizer = torch.optim.SGD(query_model.parameters(),
                                lr=wandb.config.learning_rate,
                                momentum=0.9,
                                weight_decay=1e-4)

    # Initialize queue (unchanged)
    K = 65536  # Queue size
    projection_dim = 128
    queue = torch.randn(K, projection_dim).to(device)
    queue = F.normalize(queue, dim=1)

    # Training loop (updated tau)
    num_epochs = wandb.config.epochs
    tau = wandb.config.tau  # Uses 0.07 from config
    print("Starting MoCo training")
    for epoch in range(num_epochs):
        query_model.train()
        total_loss = 0
        for view1, view2 in tqdm(simclr_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"):
            view1 = view1.to(device)
            view2 = view2.to(device)
            B = view1.size(0)

            optimizer.zero_grad()
            _, q = query_model(view1)  # Query representations
            with torch.no_grad():
                _, k = key_model(view2)  # Key representations
            q = F.normalize(q, dim=1)
            k = F.normalize(k, dim=1)

            # Compute MoCo loss
            loss = moco_loss(q, k, queue, tau)
            loss.backward()
            optimizer.step()

            # Update key_model (unchanged)
            update_key_model(query_model, key_model, m=0.999)

            # Update queue (unchanged)
            queue = torch.cat([queue, k.detach()], dim=0)
            if queue.size(0) > K:
                queue = queue[-K:, :]  # Keep last K elements

            total_loss += loss.item()

        avg_loss = total_loss / len(simclr_loader)
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "loss": avg_loss})

        # Save checkpoint every 5 epochs (unchanged)
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'query_model_state_dict': query_model.state_dict(),
                'key_model_state_dict': key_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }
            checkpoint_path = f"moco_checkpoints/moco_epoch_{epoch + 1}.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}.")

    wandb.finish()

if __name__ == "__main__":
    main()
