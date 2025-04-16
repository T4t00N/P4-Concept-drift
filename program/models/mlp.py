import torch
import torch.nn as nn
import torch.nn.functional as F
import MoCo_inference as fv
import os
import glob

class MLP(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=512, num_experts=3):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_experts)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        hidden = self.fc2(x)           # Hidden representation
        logits = self.fc3(hidden)              # Raw logits output
        return logits                     # Return logits directly

def feature_extractor(model_path, image_path, device):

    # Load the model
    model = fv.load_moco_model(model_path, device)
    
    # Preprocess the image
    image_tensor = fv.preprocess_image(image_path)
    
    # Extract the feature vector
    feature_vector = fv.extract_feature_vector(model, image_tensor, device)

    return feature_vector

def get_image_paths(root_folder):
    image_paths = []
    image_paths.extend(glob.glob(os.path.join(root_folder, '**', '*.jpg'), recursive=True))

    return image_paths



def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = '/ceph/project/P4-concept-drift/YOLOv8-Anton/data/moco_epoch_100.pt'  # Path to your MoCo checkpoint file
    # image_path = 'C:/DAKI AAU/DAKI_4_semester/P4-Concept-drift/program/models/20210405330234759.jpg'  # Path to the image you want to process

    root_folder = '/ceph/project/P4-concept-drift/YOLOv8-Anton/data/cropped_images/test'

    image_paths = get_image_paths(root_folder)

    
    temp = 0.1
    for image_path in image_paths[1000:]:

        feature_vector = feature_extractor(model_path, image_path, device) # Batch of 1 feature vectors
        feature_vector = feature_vector.to(device)
        model = MLP().to(device)
        logits = model(feature_vector)
        weights = torch.softmax(logits/temp, dim=1)  # Optional: convert to weight distribution
        print("Weights shape:", weights)

if __name__ == "__main__":
    main()
