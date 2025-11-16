import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
import seaborn as sns
from torch.nn.utils.rnn import pad_sequence

# ======================
# DATASET
# ======================
class ProtBERTDataset(Dataset):
    """Dataset pour embeddings ProtBERT + labels H/E/C"""
    def __init__(self, embeddings_dir):
        self.embeddings, self.labels = [], []
        labels_path = os.path.join(embeddings_dir, "labels.npy")
        labels_list = np.load(labels_path, allow_pickle=True)

        for i, lab in enumerate(labels_list):
            emb_path = os.path.join(embeddings_dir, f"seq_{i}.npy")
            if not os.path.exists(emb_path):
                continue
            emb = np.load(emb_path)
            if emb.shape[0] != len(lab):
                print(f"Séquence {i} ignorée (embedding={emb.shape[0]}, labels={len(lab)})")
                continue
            self.embeddings.append(torch.tensor(emb, dtype=torch.float32))
            self.labels.append(torch.tensor(lab, dtype=torch.long))

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx], self.embeddings[idx].shape[0]

# ======================
# COLLATE
# ======================
def collate_batch(batch):
    sequences, labels, lengths = zip(*batch)
    padded_seqs = pad_sequence(sequences, batch_first=True)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    return padded_seqs, padded_labels, torch.tensor(lengths)

# ======================
# TRANSFORMER
# ======================
class ProtBERT_Transformer(nn.Module):
    """Transformer simple pour classification H/E/C par résidu"""
    def __init__(self, input_dim=1024, hidden_dim=256, num_layers=2, num_heads=4, output_dim=3, dropout=0.1):
        super().__init__()
        self.embedding_projection = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim*2,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, lengths):
        mask = (torch.arange(x.size(1), device=x.device)[None, :] >= lengths[:, None])
        x = self.embedding_projection(x)
        x = self.encoder(x, src_key_padding_mask=mask)
        logits = self.classifier(x)
        return logits

# ======================
# TRAIN / EVAL
# ======================
def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for X, y, lengths in tqdm(dataloader, leave=False):
        X, y, lengths = X.to(device), y.to(device), lengths.to(device)
        optimizer.zero_grad()
        logits = model(X, lengths)
        loss = loss_fn(logits.view(-1, logits.shape[-1]), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    all_true, all_pred = [], []
    total_loss = 0
    for X, y, lengths in tqdm(dataloader, leave=False):
        X, y, lengths = X.to(device), y.to(device), lengths.to(device)
        logits = model(X, lengths)
        loss = loss_fn(logits.view(-1, logits.shape[-1]), y.view(-1))
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        mask = y != -100
        all_true.extend(y[mask].cpu().numpy())
        all_pred.extend(preds[mask].cpu().numpy())
    return total_loss / len(dataloader), np.array(all_true), np.array(all_pred)

# ======================
# MAIN
# ======================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Appareil:", device)

    # Chargement datasets
    train_loader = DataLoader(ProtBERTDataset("data/embeddings_train"), batch_size=2, shuffle=True, collate_fn=collate_batch)
    valid_loader = DataLoader(ProtBERTDataset("data/embeddings_valid"), batch_size=2, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(ProtBERTDataset("data/embeddings_test"), batch_size=2, shuffle=False, collate_fn=collate_batch)

    # Modèle
    model = ProtBERT_Transformer()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    n_epochs = 8
    for epoch in range(n_epochs):
        print(f"\n=== Époque {epoch+1}/{n_epochs} ===")
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, y_val_true, y_val_pred = evaluate(model, valid_loader, loss_fn, device)

        # Métriques
        acc = accuracy_score(y_val_true, y_val_pred)
        bal_acc = balanced_accuracy_score(y_val_true, y_val_pred)
        macro_f1 = f1_score(y_val_true, y_val_pred, average='macro')
        q3 = np.mean(y_val_true == y_val_pred)
        print(f"Train loss={train_loss:.4f}, Val loss={val_loss:.4f}")
        print(f"Accuracy={acc:.4f}, Balanced={bal_acc:.4f}, Macro F1={macro_f1:.4f}, Q3={q3:.4f}")

    # Évaluation finale
    _, y_test_true, y_test_pred = evaluate(model, test_loader, loss_fn, device)
    cm = confusion_matrix(y_test_true, y_test_pred)
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=["H","E","C"], yticklabels=["H","E","C"])
    plt.show()

    torch.save(model.state_dict(), "protbert_transformer.pt")
    print("Modèle sauvegardé")

if __name__ == "__main__":
    main()
