### Architecture CNN ###

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. Définition des Constantes d'Architecture ---

# Dimensions basées sur la Phase 1 (Encodage OHE)
INPUT_DIM = 21   # 20 AA + <pad>
OUTPUT_DIM = 4   # 3 classes (H, E, C) + <pad>

# Hyperparamètres inspirés de la littérature
KERNEL_SIZE_1 = 5  # Fenêtre de 5 acides aminés (AA) pour le contexte local
FILTERS_1 = 128
FILTERS_2 = 256
FILTERS_3 = 512    # Utilisation d'un nombre croissant de filtres

DROPOUT_RATE = 0.5 # Pour prévenir le surapprentissage


# --- 2. Définition de la Classe du Modèle CNN 1D ---

class ProteinCNN(nn.Module):
    """
    Réseau Neuronal Convolutif (CNN) 1D pour l'extraction de caractéristiques.
    Ceci sert de base pour le modèle CNN-SVM (le SVM remplacerait la couche finale).
    """
    def __init__(self, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM):
        super(ProteinCNN, self).__init__()
        
        # Le modèle sera une séquence de blocs Conv -> ReLU -> Pooling/Dropout

        # Couche 1 : Détection des motifs locaux (K=5)
        # Input: (Batch, D_input=21, L_max)
        self.conv1 = nn.Conv1d(
            in_channels=input_dim, 
            out_channels=FILTERS_1, 
            kernel_size=KERNEL_SIZE_1, 
            padding='same'  # 'same' permet de maintenir la longueur L_max
        )
        self.dropout1 = nn.Dropout(DROPOUT_RATE)

        # Couche 2 : Augmentation du champ réceptif et complexité des features
        self.conv2 = nn.Conv1d(
            in_channels=FILTERS_1, 
            out_channels=FILTERS_2, 
            kernel_size=KERNEL_SIZE_1, 
            padding='same'
        )
        self.dropout2 = nn.Dropout(DROPOUT_RATE)
        
        # Couche 3 : Extraction de caractéristiques de haut niveau
        self.conv3 = nn.Conv1d(
            in_channels=FILTERS_2, 
            out_channels=FILTERS_3, 
            kernel_size=KERNEL_SIZE_1, 
            padding='same'
        )
        # Note: Pas de pooling explicite car la tâche est positionnelle (PSSP)

        # Couche de classification finale (Remplace temporairement le SVM)
        # Nous utilisons un Conv1D de taille 1 (équivalent à une couche dense positionnelle)
        # pour obtenir l'output final (probabilité pour chaque classe).
        self.output_layer = nn.Conv1d(
            in_channels=FILTERS_3, 
            out_channels=output_dim, 
            kernel_size=1
        )
        
        
    def forward(self, x):
        # Transposition nécessaire : Conv1D attend (Batch, Channels, Length)
        # L'input vient de la Phase 1 en tant que (Batch, Length, Channels=21)
        x = x.transpose(1, 2)  
        
        # Bloc 1
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)
        
        # Bloc 2
        x = F.relu(self.conv2(x))
        x = self.dropout2(x)
        
        # Bloc 3
        x = F.relu(self.conv3(x))
        
        # Couche de sortie (Classification position par position)
        # Output: (Batch, D_output=4, L_max)
        x = self.output_layer(x)
        
        # Transposition inverse pour la fonction de perte : (Batch, L_max, D_output)
        x = x.transpose(1, 2)
        
        # Nous ne passons pas par SoftMax ici car CrossEntropyLoss de PyTorch
        # l'inclut pour des raisons de stabilité numérique (log-sum-exp trick).
        return x

# --- 3. Exemple d'Instanciation et Test ---

if __name__ == '__main__':
    # Simuler un batch de données (tiré de la Phase 1)
    # Exemple: 16 séquences de longueur max 200, avec 21 features OHE
    BATCH_SIZE = 16
    L_MAX = 200
    dummy_input = torch.randn(BATCH_SIZE, L_MAX, INPUT_DIM) 
    
    # Création du modèle
    model = ProteinCNN()
    
    # Passage de l'input dans le modèle
    output = model(dummy_input)
    
    print("--- Modèle CNN 1D ---")
    print(f"Dimension de l'input simulé: {dummy_input.shape}")
    print(f"Dimension de l'output du modèle: {output.shape}")
    print(f" => (Batch={BATCH_SIZE}, L_max={L_MAX}, D_output={OUTPUT_DIM})")

    # Vérification du nombre de paramètres (une bonne pratique)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total de paramètres entraînables: {num_params}")