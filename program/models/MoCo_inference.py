import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F

# MoCo model definition (same as before)
class MoCoModel(torch.nn.Module):
    def __init__(self, base_model, projection_dim=128):
        super(MoCoModel, self).__init__()
        self.encoder = base_model
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(2048, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, projection_dim)  # Projection head to output 128-dimensional vector
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        return h, z  # Returning both the original and projected feature vectors

# Load the pre-trained model
def load_moco_model(model_path, device):
    base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    base_model.fc = torch.nn.Identity()  # Removing the final classification layer
    model = MoCoModel(base_model).to(device)
    
    # Load the checkpoint
    # print(f"Loading MoCo checkpoint from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

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
    # print("MoCo model loaded successfully")
    return model

# Preprocess image to match the model's input requirements
def preprocess_image(image_path):
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

# Extract feature vector using the trained MoCo model
def extract_feature_vector(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    _, feature_vector = model(image_tensor)  # Get the feature vector (z) from the projection head
    return feature_vector

# Main function to load model, preprocess image, and extract feature vector
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = 'C:\DAKI AAU\DAKI_4_semester\P4-Concept-drift\program\models\moco_epoch_100.pt'  # Path to your MoCo checkpoint file
    image_path = 'C:/DAKI AAU/DAKI_4_semester/P4-Concept-drift/program/models/20210405330234759.jpg'  # Path to the image you want to process
    

    
    # Load the model
    model = load_moco_model(model_path, device)
    
    # Preprocess the image
    image_tensor = preprocess_image(image_path)
    
    # Extract the feature vector
    feature_vector = extract_feature_vector(model, image_tensor, device)
    
    print(f"Extracted feature vector (dim {feature_vector.shape[1]}): {feature_vector}")
    return feature_vector

if __name__ == "__main__":
    fv = main()
