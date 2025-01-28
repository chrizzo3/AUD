import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import shutil

def augment_rotation(skeleton, angle):
    """Ruota uno scheletro di un dato angolo (in gradi) attorno all'asse Y."""
    angle_rad = np.radians(angle)
    rotation_matrix = np.array([
        [np.cos(angle_rad), 0, np.sin(angle_rad)],
        [0, 1, 0],
        [-np.sin(angle_rad), 0, np.cos(angle_rad)]
    ])
    skeleton[:, :3] = np.dot(skeleton[:, :3], rotation_matrix.T)
    return skeleton

def normalize_skeleton(skeleton):
    """Normalizza i joint rispetto al centro del bacino."""
    pelvis = skeleton[0, :3]  # Assumendo che il bacino sia il primo joint
    skeleton[:, :3] -= pelvis
    return skeleton

def process_skeleton_file(file_path, output_dir: Path, angles, labels_dict):
    """Applica rotazione e normalizzazione a un file .npy."""
    skeleton_data = np.load(file_path)
    sequences = []

    for angle in angles:
        rotated_sequence = []
        for frame in skeleton_data:
            normalized_frame = []
            for body in frame:
                if body.sum() == 0:  # Salta corpi vuoti
                    normalized_frame.append(body)
                    continue
                rotated_body = augment_rotation(body.copy(), angle)
                normalized_body = normalize_skeleton(rotated_body)
                normalized_frame.append(normalized_body)
            rotated_sequence.append(np.stack(normalized_frame))
        sequences.append(np.stack(rotated_sequence))

    # Salva le sequenze rotazionali
    base_filename = Path(file_path).stem
    for i, sequence in enumerate(sequences):
        output_path = Path(output_dir) / f"{base_filename}_rot{angles[i]}.npy"
        np.save(output_path, sequence)
        # Add the new file to the labels dictionary
        labels_dict[f"{base_filename}_rot{angles[i]}.npy"] = labels_dict.get(f"{base_filename}.npy", "unknown")

def process_all_files(input_dir, output_dir, angles, labels_file):
    """Processa tutti i file .npy in una directory (incluso nelle sottocartelle)."""
    os.makedirs(output_dir, exist_ok=True)
    input_files = list(Path(input_dir).rglob("*.npy"))  # Usa rglob per la ricerca ricorsiva

    # Load existing labels
    with open(labels_file, 'r') as f:
        labels_dict = json.load(f)

    for file_path in tqdm(input_files, desc="Processing skeleton files"):
        process_skeleton_file(file_path, output_dir, angles, labels_dict)

    # Save updated labels
    with open(labels_file, 'w') as f:
        json.dump(labels_dict, f, indent=4)

def split_data():
    """Divide i dati in training, validation e test set."""
    all_files = []
    for subdir in ["1_skeleton", "2_skeletons", "more_than_2_skeletons", "augmented"]:
        subdir_path = Path("sequences") / subdir
        if subdir_path.exists():
            all_files += list(subdir_path.glob("*.npy"))

    # Split combined files into train, validation, and test sets
    train_files, temp_files = train_test_split(all_files, train_size=0.7, random_state=42)
    val_files, test_files = train_test_split(temp_files, train_size=0.5, random_state=42)

    # Sposta i file nelle directory corrispondenti
    print("Moving training files...")
    for train_file in tqdm(train_files, desc="Training"):
        dest_path = Path("sequences") / "new_train" / train_file.name
        os.makedirs(dest_path.parent, exist_ok=True)
        shutil.move(str(train_file), dest_path)

    print("Moving validation files...")
    for val_file in tqdm(val_files, desc="Validation"):
        dest_path = Path("sequences") / "new_val" / val_file.name
        os.makedirs(dest_path.parent, exist_ok=True)
        shutil.move(str(val_file), dest_path)

    print("Moving test files...")
    for test_file in tqdm(test_files, desc="Test"):
        dest_path = Path("sequences") / "new_test" / test_file.name
        os.makedirs(dest_path.parent, exist_ok=True)
        shutil.move(str(test_file), dest_path)

    shutil.rmtree("sequences/1_skeleton")
    shutil.rmtree("sequences/2_skeletons")
    shutil.rmtree("sequences/more_than_2_skeletons")
    shutil.rmtree(CONFIG["output_dir"])
    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
    print(f"Testing files: {len(test_files)}")

    # Generate separate label files for train, val, and test sets
    generate_label_files(train_files, val_files, test_files, CONFIG["labels_file"])

def generate_label_files(train_files, val_files, test_files, labels_file):
    """Generate separate JSON files for train, val, and test labels."""
    with open(labels_file, 'r') as f:
        labels_dict = json.load(f)

    train_labels = {file.name: labels_dict[file.name] for file in train_files}
    val_labels = {file.name: labels_dict[file.name] for file in val_files}
    test_labels = {file.name: labels_dict[file.name] for file in test_files}

    with open("sequences/train_labels.json", 'w') as f:
        json.dump(train_labels, f, indent=4)

    with open("sequences/val_labels.json", 'w') as f:
        json.dump(val_labels, f, indent=4)

    with open("sequences/test_labels.json", 'w') as f:
        json.dump(test_labels, f, indent=4)

if __name__ == "__main__":
    CONFIG = {
        "input_dir": "sequences",  # Directory con file originali
        "output_dir": "sequences/augmented",  # Directory per file processati
        "angles": [0, 15, -15, 30, -30],  # Angoli di rotazione
        "labels_file": "sequences/labels.json"  # File dei label
    }

    process_all_files(CONFIG["input_dir"], CONFIG["output_dir"], CONFIG["angles"], CONFIG["labels_file"])
    split_data()