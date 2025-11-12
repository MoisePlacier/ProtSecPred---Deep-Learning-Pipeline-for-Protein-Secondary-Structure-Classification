import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Assurez-vous que les classes et constantes (ProteinDataset, ProteinCNN, PAD_INDEX) 
# sont accessibles dans ce script.
# Import des classes et constantes de 'dataset.py'
from encoding import ProteinDataset, SS_TO_INT, INPUT_DIM

# Import de la classe de modèle depuis 'model.py'
from model import ProteinCNN

# Dimensions et constantes (à synchroniser)
SS_TO_INT = {'H': 0, 'E': 1, 'C': 2, '<pad>': 3}
PAD_INDEX = SS_TO_INT['<pad>'] 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
L_MAX = 505 # La longueur maximale utilisée pour l'entraînement

def evaluate_model(model, test_loader, model_path):
    """
    Charge le modèle entraîné et évalue sa précision Q3 sur le jeu de test.
    """
    # 1. Chargement du meilleur modèle
    try:
        model.load_state_dict(torch.load(model_path))
        print(f"\nModèle chargé depuis: {model_path}")
    except FileNotFoundError:
        print(f"\nErreur: Le fichier modèle '{model_path}' n'a pas été trouvé. Assurez-vous que l'entraînement a bien sauvegardé 'best_model.pth'.")
        return 0.0

    model.to(DEVICE)
    model.eval() # Met le modèle en mode évaluation (désactive Dropout)
    
    total_correct_predictions = 0
    total_residues = 0

    with torch.no_grad():
        for X_batch, Y_batch, L_batch in test_loader:
            X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)
            
            # Forward pass
            outputs = model(X_batch) 
            
            # Trouver la classe prédite (l'indice avec la probabilité maximale)
            # outputs est (Batch, L_max, D_output=4)
            _, predicted = torch.max(outputs.data, 2)

            # Créer un masque pour ignorer les positions de padding (PAD_INDEX = 3)
            mask = (Y_batch != PAD_INDEX)
            
            # Calcul des prédictions correctes et du total des résidus
            total_residues += mask.sum().item()
            total_correct_predictions += ((predicted == Y_batch) & mask).sum().item()

    # Calcul de la précision Q3
    q3_accuracy = (total_correct_predictions / total_residues) * 100 if total_residues > 0 else 0.0
    
    return q3_accuracy, total_correct_predictions, total_residues


# --- Exécution de l'Évaluation ---

if __name__ == '__main__':
    # Définir le chemin vers le jeu de test et le modèle sauvegardé
    TEST_FILE = 'testing_dataset.json'  
    MODEL_PATH = 'best_model.pth'

    # Création du Dataset et du DataLoader de Test
    test_dataset = ProteinDataset(TEST_FILE, max_len=L_MAX)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False) # Shuffle=False pour le test
    
    # Instanciation du modèle (pour charger les poids)
    model = ProteinCNN() 

    # Lancement de l'évaluation
    q3, correct, total = evaluate_model(model, test_loader, MODEL_PATH)

    print("\n========================================")
    print("      RÉSULTATS DE L'ÉVALUATION FINALE")
    print("========================================")
    print(f"Précision Q3 sur le jeu de test: {q3:.2f}%")
    print(f"Résidus correctement prédits: {correct} / {total}")
    print("========================================\n")