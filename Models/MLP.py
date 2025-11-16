# protbert_mlp_simple.py
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ======= 1. Charger embeddings et labels =======
def load_embeddings_and_labels(folder_path):
    labels_list = np.load(os.path.join(folder_path, "labels.npy"), allow_pickle=True)
    seq_files = sorted([f for f in os.listdir(folder_path) if f.startswith("seq_") and f.endswith(".npy")])
    
    X_list, y_list = [], []
    for i, f in enumerate(seq_files):
        emb = np.load(os.path.join(folder_path, f))
        labels = labels_list[i]
        X_list.append(emb)
        y_list.append(labels)
    
    X_all = np.vstack(X_list)
    y_all = np.hstack(y_list)
    return X_all, y_all

train_folder = "../data/embeddings_train"
val_folder   = "../data/embeddings_valid"

X_train, y_train = load_embeddings_and_labels(train_folder)
X_val, y_val     = load_embeddings_and_labels(val_folder)

# Normalisation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)

# ======= 2. Dataset PyTorch =======
class ProtDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(ProtDataset(X_train, y_train), batch_size=128, shuffle=True)
val_loader   = DataLoader(ProtDataset(X_val, y_val), batch_size=128)

# ======= 3. Définir le MLP =======
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, output_dim=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.model(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleMLP(input_dim=X_train.shape[1]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ======= 4. Entraînement =======
epochs = 10
for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            preds = outputs.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(yb.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    acc = accuracy_score(all_labels, all_preds)
    print(f"Epoch {epoch+1}/{epochs} - Val Accuracy: {acc:.4f}")

# ======= 5. Sauvegarde du modèle =======
torch.save(model.state_dict(), "mlp_simple_protbert.pth")
print("Modèle sauvegardé dans mlp_simple_protbert.pth")
