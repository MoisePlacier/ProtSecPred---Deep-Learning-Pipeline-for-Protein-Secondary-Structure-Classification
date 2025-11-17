import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


# 20 acides aminés standards + 1 pour le padding
AA_ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_INT = {aa: i for i, aa in enumerate(AA_ALPHABET)}
AA_TO_INT['<pad>'] = 20 # 21ème dimension pour le padding

# Les 8 classes DSSP (H, G, I sont des hélices; E, B sont des feuillets; T, S, L sont des boucles)
# L est souvent utilisé pour Loop/Irregular/Coil
DSSP_8_CLASSES = 'HGEBITSL'
# Mapping vers les 3 classes standard: H (Helix), E (Sheet), C (Coil)
DSSP_3_MAP = {
    'H': 'H',  # Alpha Helix
    'G': 'H',  # 3-10 Helix
    'I': 'H',  # Pi Helix
    'E': 'E',  # Beta Strand
    'B': 'E',  # Beta Bridge
    'T': 'C',  # Turn
    'S': 'C',  # Bend
    'L': 'C',  # Loop / Irregular / Coil (often noted as '-')
    '-': 'C',  # Pour gérer les éventuels 'manques' qui sont considérés comme Coil
}

SS_3_CLASSES = 'HEC'
SS_TO_INT = {ss: i for i, ss in enumerate(SS_3_CLASSES)}
SS_TO_INT['<pad>'] = 3 # 4ème dimension pour le padding (masque de perte)

# Dimensions OHE
INPUT_DIM = len(AA_ALPHABET) + 1 # 21 (20 AA + <pad>)
OUTPUT_DIM = len(SS_3_CLASSES) + 1 # 4 (3 SS + <pad>)


# --- 2. Fonction d'Encodage (One-Hot Encoding) ---

def one_hot_encode(sequence, char_to_int_map, dim, padding_char='<pad>', dtype=torch.float32):
    """
    Convertit une séquence de caractères en un tenseur One-Hot Encoded.
    """
    seq_len = len(sequence)
    # Tenseur de zéros (L x D)
    one_hot = torch.zeros(seq_len, dim, dtype=dtype)
    
    for i, char in enumerate(sequence):
        # Utiliser l'index du caractère, ou l'index du padding si non trouvé (erreur)
        # Pour les AA et SS, nous faisons une vérification simple
        idx = char_to_int_map.get(char, char_to_int_map[padding_char])
        if idx != char_to_int_map[padding_char] or char == padding_char:
            one_hot[i, idx] = 1.0
            
    return one_hot


class ProteinDataset(Dataset):
    def __init__(self, data_file_path, max_len=None):
        with open(data_file_path, 'r') as f:
            self.data = json.load(f)
        
        # Trouver la longueur maximale si non spécifiée (pour le padding)
        if max_len is None:
            self.max_len = max([len(record['primary_sequence']) for record in self.data])
        else:
            self.max_len = max_len
        
        print(f"Dataset loaded. Max sequence length (L_max): {self.max_len}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        
        # 1. Préparation de la Séquence Primaire (Input X)
        primary_seq = record['primary_sequence']
        L = len(primary_seq)
        
        # 2. Préparation de la Structure Secondaire (Label Y)
        raw_ss_seq = record['secondary_structure']
        
        # Mappage des 8 classes DSSP aux 3 classes (H, E, C)
        ss_3_seq = ''.join([DSSP_3_MAP.get(ss, 'C') for ss in raw_ss_seq])
        
        # 3. One-Hot Encoding (OHE)
        # Entrée X: Séquence AA
        X_ohe = one_hot_encode(primary_seq, AA_TO_INT, INPUT_DIM)
        
        # Sortie Y: Structure Secondaire (peut être laissé en int pour CrossEntropyLoss)
        # Mais on va OHE ici pour la cohérence, et on utilisera la version int pour la loss
        Y_ohe = one_hot_encode(ss_3_seq, SS_TO_INT, OUTPUT_DIM)
        
        # Version Label (Int) pour la Loss
        Y_int = torch.tensor([SS_TO_INT.get(ss, SS_TO_INT['<pad>']) for ss in ss_3_seq], dtype=torch.long)

        
        # 4. Padding (Remplissage)
        pad_size = self.max_len - L
        
        # Pad Input X: (L x 21) -> (L_max x 21)
        X_padded = torch.nn.functional.pad(X_ohe.transpose(0, 1), (0, pad_size), 'constant', 0).transpose(0, 1)

        # Pad Output Y (Int): (L) -> (L_max)
        # Nous remplissons les labels d'une valeur spéciale (l'index du padding) pour le masque
        Y_padded = torch.nn.functional.pad(Y_int, (0, pad_size), 'constant', SS_TO_INT['<pad>'])


        return X_padded, Y_padded, L # L est retourné pour le dé-padding après prédiction



if __name__ == '__main__':
    file_path = 'testing_dataset.json' 

    # Création du Dataset et du DataLoader
    test_dataset = ProteinDataset(file_path)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    
    # Inspection du premier batch
    X_batch, Y_batch, L_batch = next(iter(test_dataloader))
    
    print("\n--- Inspection des Tenseurs ---")
    print(f"Batch Size: {X_batch.shape[0]}")
    print(f"Longueur Max (L_max): {test_dataset.max_len}")
    
    # Dimensions du Tenseur d'Entrée (Input X)
    # (Batch Size, L_max, D_input) => (32, L_max, 21)
    print(f"Dimension Tenseur X (Input CNN): {X_batch.shape}")
    print(f"   => (Batch, L_max, D_input(21))")

    # Dimension du Tenseur de Label (Output Y)
    # (Batch Size, L_max) => (32, L_max) - Long est nécessaire pour CrossEntropyLoss
    print(f"Dimension Tenseur Y (Labels int): {Y_batch.shape}") 
    print(f"   => (Batch, L_max) - (Indices H/E/C ou <pad>=3)")

    first_aa_vector = X_batch[0, 0, :]
    print(f"\nExemple de vecteur OHE (premier AA): {first_aa_vector[:len(AA_ALPHABET)]}")
    print(f"Index de la classe d'output du premier AA: {Y_batch[0, 0]}")
    
    # Note: Dans PyTorch, les CNN 1D s'attendent souvent à (Batch, D_input, L_max)
    # Il faudra transposer X_batch lors de l'appel du modèle: X_batch.transpose(1, 2)
