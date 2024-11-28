from torch.nn import TransformerEncoder, TransformerEncoderLayer

class DualBranchReID(nn.Module):
    def __init__(self, feature_dim=512, hidden_dim=256, num_layers=2):
        super(DualBranchReID, self).__init__()
        # Global Branch
        self.global_branch = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        # Local Branch
        self.local_branch = TransformerEncoder(
            TransformerEncoderLayer(d_model=feature_dim, nhead=8, dim_feedforward=hidden_dim), 
            num_layers=num_layers
        )
        self.fc = nn.Linear(feature_dim, feature_dim)

    def forward(self, global_features, local_features):
        # Global Features
        global_features = self.global_branch(global_features)
        global_features = torch.flatten(global_features, 1)
        
        # Local Features
        local_features = self.local_branch(local_features)
        local_features = torch.mean(local_features, dim=1)

        # Combined Output
        combined_features = torch.cat((global_features, local_features), dim=1)
        combined_features = self.fc(combined_features)
        return combined_features
