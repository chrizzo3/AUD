import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from CNN_GRUModel import CNNGRUModel
from dataLoader import get_dataloader

#TODO chaing labels file and train model

# Configurazione
CONFIG = {
    "sequence_dir": "sequences/train_file",
    "train_label_file": "sequences/train_labels.json",
    "val_sequence_dir": "sequences/val_file",
    "val_label_file": "sequences/val_labels.json",
    "label_map": {
        6: 0,   # "Pick up"
        7: 1,   # "Throw"
        24: 2,  # "Kicking something"
        25: 3,  # "Reach into pocket"
        31: 4,  # "Point to something"
        93: 5,  # "Shake fist"
        42: 6,  # "Staggering"
        43: 7,  # "Falling down"
        50: 8,  # "Punch/slap"
        51: 9,  # "Kicking"
        52: 10, # "Pushing"
        57: 11, # "Touch pocket"
        106: 12,# "Hit with object"
        107: 13,# "Wield knife"
        110: 14 # "Shoot with gun"
    },
    "epochs": 30,
    "batch_size": 256,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
     "feature_size": 9,
    "model_save_path": "models/cnn_gru_model.pth",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "patience": 5
}

# Carica il modello
model = CNNGRUModel(
    input_channels=9,
    num_joints=25,
    cnn_out_channels=64,
    gru_hidden_size=128,
    num_classes=len(CONFIG["label_map"])
).to(CONFIG["device"])

# Loss e ottimizzatore
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)

# Dataloader
train_loader = get_dataloader(
    CONFIG["sequence_dir"], CONFIG["train_label_file"], CONFIG["label_map"], batch_size=CONFIG["batch_size"]
)
val_loader = get_dataloader(
    CONFIG["val_sequence_dir"], CONFIG["val_label_file"], CONFIG["label_map"], batch_size=CONFIG["batch_size"]
)

# Training Loop
def train_model():
    best_loss = float('inf')
    for epoch in range(CONFIG["epochs"]):
        # Training Phase
        model.train()
        train_loss = 0.0
        patience_counter = 0
        all_train_labels = []
        all_train_preds = []

        with tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{CONFIG['epochs']}") as pbar:
            for sequences, labels in pbar:
                sequences, labels = sequences.to(CONFIG["device"]), labels.to(CONFIG["device"])

                # Forward pass
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                train_loss += loss.item()

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Collect predictions
                preds = torch.argmax(outputs, dim=1)
                all_train_labels.extend(labels.cpu().numpy())
                all_train_preds.extend(preds.cpu().numpy())

                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        # Validation Phase
        model.eval()
        val_loss = 0.0
        all_val_labels = []
        all_val_preds = []

        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(CONFIG["device"]), labels.to(CONFIG["device"])

                outputs = model(sequences)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(preds.cpu().numpy())

        # Calcola metriche
        train_accuracy = accuracy_score(all_train_labels, all_train_preds)
        train_f1 = f1_score(all_train_labels, all_train_preds, average='weighted')
        val_accuracy = accuracy_score(all_val_labels, all_val_preds)
        val_f1 = f1_score(all_val_labels, all_val_preds, average='weighted')

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}:")
        print(f"  Training Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.4f}, F1-Score: {train_f1:.4f}")
        print(f"  Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}, F1-Score: {val_f1:.4f}")

        scheduler.step(avg_val_loss)

        # Salva il modello migliore
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            os.makedirs(os.path.dirname(CONFIG["model_save_path"]), exist_ok=True)
            torch.save(model.state_dict(), CONFIG["model_save_path"])
            print(f"Model saved at epoch {epoch+1} with loss {best_loss:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= CONFIG["patience"]:
            print("Early stopping triggered. Stopping training")
            break

# Esegui il training
if __name__ == "__main__":
    train_model()


