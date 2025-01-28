import torch
from sklearn.metrics import classification_report, confusion_matrix
from dataLoader import get_dataloader  
from CNN_GRUModel import CNNGRUModel  
from train import label_map
import numpy as np
import os

# Configurazioni principali
CONFIG = {
    "sequence_dir": "sequences/test_file",  # Directory contenente le sequenze
    "label_file": "sequences/test_labels.json",  # File dei label
    "label_map": label_map,
    "batch_size": 256,
    "num_classes": 15,  # Numero di classi nel dataset
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "model_save_path": "models/cnn_gru_model.pth"  # Percorso del modello salvato
}

# Caricamento del dataloader di test
test_loader = get_dataloader(
    CONFIG["sequence_dir"],
    CONFIG["label_file"],
    CONFIG["label_map"],
    batch_size=CONFIG["batch_size"],
    num_workers=4,
)

# Inizializzazione del modello
model = CNNGRUModel(
    input_channels=9,
    num_joints=25,
    cnn_out_channels=64,
    gru_hidden_size=128,
    num_classes=len(CONFIG["label_map"])
).to(CONFIG["device"])

# Caricamento del modello salvato
if os.path.exists(CONFIG["model_save_path"]):
    model.load_state_dict(torch.load(CONFIG["model_save_path"], weights_only = False))
    print("Modello caricato correttamente!")
else:
    raise FileNotFoundError(f"Modello non trovato: {CONFIG['model_save_path']}")

# Valutazione del modello
def evaluate_model():
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(CONFIG["device"]), labels.to(CONFIG["device"])

            # Forward pass
            outputs = model(sequences)

            # Previsioni
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    # Calcolo delle metriche
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=[f"Class {i}" for i in range(CONFIG["num_classes"])]))

    # Matrice di confusione
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

if __name__ == "__main__":
    evaluate_model()
