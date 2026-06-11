import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split

# --- Settings ---
DATA_PATH = "./processed_data"
MODEL_SAVE_PATH = "drone_catcher_model.pth"
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 20

# --- 1. The Dataset Loader ---
class AudioDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load .npy file
        spec = np.load(self.file_paths[idx])
        # Ensure it is 1 channel (1, Height, Width)
        spec = spec[np.newaxis, ...] 
        return torch.tensor(spec, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

def load_data():
    # Use glob to find all .npy files in the subfolders
    drone_files = glob.glob(os.path.join(DATA_PATH, "drone", "*.npy"))
    bg_files = glob.glob(os.path.join(DATA_PATH, "background", "*.npy"))

    print(f"Found {len(drone_files)} Drone samples")
    print(f"Found {len(bg_files)} Background samples")

    if len(drone_files) == 0 or len(bg_files) == 0:
        raise ValueError("CRITICAL ERROR: Data missing. Did you run master_setup.py?")

    # Label 1 = Drone, Label 0 = Background
    files = drone_files + bg_files
    labels = [1] * len(drone_files) + [0] * len(bg_files)

    return train_test_split(files, labels, test_size=0.2, random_state=42)

# --- 2. The Neural Network (CNN) ---
class DroneNet(nn.Module):
    def __init__(self):
        super(DroneNet, self).__init__()
        # 3 Convolutional Layers to find patterns in the spectrogram
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        
        # Adaptive pooling ensures it works regardless of slight size variations
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4)) 
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 2) # Output: [Background, Drone]

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        
        x = self.global_pool(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- 3. Training Loop ---
def train():
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Load Data
    train_files, val_files, train_labels, val_labels = load_data()
    train_loader = DataLoader(AudioDataset(train_files, train_labels), batch_size=BATCH_SIZE, shuffle=True)

    # Init Model
    model = DroneNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print("--- Starting Training ---")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | Accuracy: {acc:.2f}%")

    # Save the trained brain
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"--- Model Saved to {MODEL_SAVE_PATH} ---")

if __name__ == "__main__":
    # Install sklearn if missing
    try:
        import sklearn
    except ImportError:
        os.system("pip install scikit-learn")
        
    train()