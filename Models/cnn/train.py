import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# ==============================================================================
# REMPLACEZ CES LIGNES PAR VOS IMPORTS RÉELS si vous séparez les fichiers
# Sinon, assurez-vous que les classes ProteinDataset et ProteinCNN sont définies ci-dessus
# ==============================================================================
from encoding import ProteinDataset, SS_TO_INT
from model import ProteinCNN, INPUT_DIM, OUTPUT_DIM

# Définition des Constantes (à synchroniser avec vos modules)
SS_TO_INT = {'H': 0, 'E': 1, 'C': 2, '<pad>': 3}
OUTPUT_DIM = len(SS_TO_INT) # 4
PAD_INDEX = SS_TO_INT['<pad>'] 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Le reste du code suppose que ProteinDataset et ProteinCNN sont accessibles.

# ==============================================================================
# --- 1. Définition de la Fonction d'Entraînement ---
# ==============================================================================

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    """
    Boucle d'entraînement et d'évaluation du modèle.
    """
    model.to(DEVICE)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # --- PHASE D'ENTRAÎNEMENT ---
        model.train()
        train_loss = 0.0
        train_correct_predictions = 0
        train_total_residues = 0

        for X_batch, Y_batch, L_batch in train_loader:
            
            # 1. Préparation des données et envoi au device (CPU/GPU)
            X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)
            
            # 2. Forward pass (Prédiction)
            optimizer.zero_grad()
            outputs = model(X_batch) # Shape: (Batch, L_max, D_output=4)

            # 3. Calcul de la Loss
            # La Loss est calculée position par position.
            # outputs doit être transformé en (Batch * L_max, D_output)
            # Y_batch doit être transformé en (Batch * L_max)
            loss = criterion(
                outputs.contiguous().view(-1, OUTPUT_DIM), 
                Y_batch.contiguous().view(-1)
            )

            # 4. Backward pass et optimisation
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)

            # Optionnel: Calcul de la Q3 (Précision) pour les résidus non-padding
            # Nous simplifions ici au calcul de Q3 global.
            _, predicted = torch.max(outputs.data, 2)
            mask = (Y_batch != PAD_INDEX)
            
            train_total_residues += mask.sum().item()
            train_correct_predictions += ((predicted == Y_batch) & mask).sum().item()


        train_loss /= len(train_loader.dataset)
        train_accuracy = (train_correct_predictions / train_total_residues) * 100 if train_total_residues > 0 else 0

        # --- PHASE DE VALIDATION ---
        model.eval()
        val_loss = 0.0
        val_correct_predictions = 0
        val_total_residues = 0

        with torch.no_grad():
            for X_batch, Y_batch, L_batch in val_loader:
                X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)
                
                outputs = model(X_batch)
                
                loss = criterion(
                    outputs.contiguous().view(-1, OUTPUT_DIM), 
                    Y_batch.contiguous().view(-1)
                )
                val_loss += loss.item() * X_batch.size(0)

                _, predicted = torch.max(outputs.data, 2)
                mask = (Y_batch != PAD_INDEX)
                
                val_total_residues += mask.sum().item()
                val_correct_predictions += ((predicted == Y_batch) & mask).sum().item()

        val_loss /= len(val_loader.dataset)
        val_accuracy = (val_correct_predictions / val_total_residues) * 100 if val_total_residues > 0 else 0

        print(f'Epoch {epoch+1}/{num_epochs} | '
              f'Train Loss: {train_loss:.4f} | Train Acc (Q3): {train_accuracy:.2f}% | '
              f'Val Loss: {val_loss:.4f} | Val Acc (Q3): {val_accuracy:.2f}%')

        # Sauvegarde du meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("Modèle sauvegardé (meilleure perte de validation).")


# ==============================================================================
# --- 2. Configuration et Lancement (Exemple) ---
# ==============================================================================

if __name__ == '__main__':
    # REMPLACER AVEC VOS CHEMINS DE FICHIERS CONSOLIDÉS
    TRAIN_FILE = 'training_30_dataset.json'  
    VAL_FILE = 'validation_dataset.json'
    
    # Hypothèses tirées de vos tests (vous devez synchroniser L_max pour les 3 loaders)
    L_MAX = 505
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 10 

    # --- Chargement des Données ---
    # Note: On force L_max pour que les 3 jeux aient la même dimension
    train_dataset = ProteinDataset(TRAIN_FILE, max_len=L_MAX)
    val_dataset = ProteinDataset(VAL_FILE, max_len=L_MAX)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Modèle, Loss et Optimiseur ---
    model = ProteinCNN()

    # Fonction de Perte: CrossEntropyLoss (standard pour la classification multi-classe)
    # L'argument 'ignore_index=PAD_INDEX' est CRUCIAL: il ignore les positions 
    # de padding (label 3) dans le calcul de la perte, comme recommandé.
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_INDEX) 
    
    # Optimiseur: Adam (standard pour le Deep Learning)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\nDébut de l'entraînement sur le device: {DEVICE}")
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=NUM_EPOCHS)