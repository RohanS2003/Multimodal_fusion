import torch
import torch.nn as nn
from torchvision import models

# Load Pre-trained ResNet-50
class ResNet50FaceModel(nn.Module):
    def __init__(self, embedding_size=128):
        super(ResNet50FaceModel, self).__init__()
        base_model = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(base_model.children())[:-1])  # Remove FC layer
        self.fc = nn.Linear(base_model.fc.in_features, embedding_size)   # Add embedding layer

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Triplet Loss Function
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = (anchor - positive).pow(2).sum(1)  # Squared L2 distance
        neg_dist = (anchor - negative).pow(2).sum(1)
        loss = torch.relu(pos_dist - neg_dist + self.margin).mean()
        return loss

# Example Model Initialization
model = ResNet50FaceModel(embedding_size=128)
criterion = TripletLoss()
