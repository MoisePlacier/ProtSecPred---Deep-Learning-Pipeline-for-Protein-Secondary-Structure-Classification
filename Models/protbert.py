# protbert.py
# Objectif : Exemple complet de pipeline ProtBERT pour prédire la structure secondaire

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# MINI DATASET 

# Séquences d'acides aminés et structures secondaires correspondantes
# H = hélice, E = feuillet, C = boucle
dataset = [
    {"sequence": "ACDEFGHIK", "labels": "CCCHHHHHH"},
    {"sequence": "LMNPQRSTV", "labels": "HHHHCCCCE"},
    {"sequence": "WYACDGHIK", "labels": "EEEHHHCCC"},
]

# CHARGER PROTBERT
print("Chargement de ProtBERT...")
tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
model = AutoModel.from_pretrained("Rostlab/prot_bert")
model.eval()  # mode évaluation (pas d'entraînement du modèle de langage)


# FONCTION D’EMBEDDING
def embed_sequence(seq):
    """Retourne les embeddings (vecteurs) d'une séquence avec ProtBERT"""
    seq = " ".join(list(seq))  # ProtBERT attend des acides aminés séparés par des espaces
    tokens = tokenizer(seq, return_tensors="pt")
    with torch.no_grad():
        output = model(**tokens)
    emb = output.last_hidden_state.squeeze(0)[1:-1]  # on retire les tokens spéciaux [CLS], [SEP]
    return emb.numpy()  # (longueur de séquence, 1024)

# CRÉER X et y

X_list, y_list = [], []
print("Génération des embeddings ProtBERT...")

for sample in dataset:
    emb = embed_sequence(sample["sequence"])
    labels = list(sample["labels"])
    # On ajoute les embeddings et leurs labels correspondants
    X_list.append(emb)
    y_list.extend(labels)

# Concaténer toutes les séquences (par position)
X = np.vstack(X_list)
y = np.array(y_list)
print(f"Embeddings générés : X = {X.shape}, y = {y.shape}")


# ENTRAÎNEMENT D'UN MODELE SIMPLE
print("Entraînement d’un RandomForest...")
clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
clf.fit(X, y)
print("Modèle entraîné")


# PRÉDICTION SUR UNE NOUVELLE SÉQUENCE
test_seq = "ACDFGHIKL"
print(f"🔹 Prédiction sur nouvelle séquence : {test_seq}")

test_emb = embed_sequence(test_seq)
pred = clf.predict(test_emb)
print("Structure secondaire prédite :", "".join(pred))


# ÉVALUATION RAPIDE SUR LE TRAIN SET
y_pred_train = clf.predict(X)
acc = accuracy_score(y, y_pred_train)
print(f"Accuracy (train) = {acc:.2f}")
