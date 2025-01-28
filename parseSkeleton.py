import os
import zipfile
import json
from tqdm import tqdm  # Per la barra di progresso

# Path delle cartelle zip
zip_paths = [
    r"C:/Users/chris/Desktop/Università/FIA/ProgettoFIABuono/nturgbd_skeletons_s001_to_s017.zip",
    r"C:/Users/chris/Desktop/Università/FIA/ProgettoFIABuono/nturgbd_skeletons_s018_to_s032.zip"
]

# Pose rilevanti (label delle azioni) e nomi associati
relevant_actions = {
    6: "Pick up", 7: "Throw", 24: "Kicking something", 25: "Reach into pocket",
    31: "Point to something", 93: "Shake fist", 42: "Staggering", 43: "Falling down",
    50: "Punch/slap", 51: "Kicking", 52: "Pushing", 57: "Touch pocket",
    106: "Hit with object", 107: "Wield knife", 110: "Shoot with gun"
}

# Directory temporanea per estrarre i file
extract_dir = "extracted_skeletons"

# Directory per i file JSON
output_dir = "processed_skeletons"
os.makedirs(output_dir, exist_ok=True)


def extract_and_filter_files(zip_paths, relevant_actions):
    """Estrae e filtra i file dai .zip in base alle pose rilevanti."""
    os.makedirs(extract_dir, exist_ok=True)
    extracted_files = []

    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path, 'r') as z:
            for file_name in tqdm(z.namelist(), desc=f"Estrazione da {os.path.basename(zip_path)}", ncols=100):
                if file_name.endswith('/'):
                    continue

                # Estrai l'etichetta della classe dal nome del file
                action_label = int(file_name.split('A')[-1].split('.')[0])
                if action_label in relevant_actions:
                    z.extract(file_name, extract_dir)
                    extracted_files.append((os.path.join(extract_dir, file_name), action_label))

    return extracted_files


def parse_skeleton_file(file_path, action_label, action_name):
    """Parsa un file .skeleton e restituisce i dati strutturati."""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    data = {
        "filename": os.path.basename(file_path),
        "action_label": action_label,
        "action_name": action_name,
        "frames": []
    }

    idx = 0
    num_frames = int(lines[idx].strip())
    idx += 1

    for _ in range(num_frames):
        frame_data = {
            "frame_index": len(data["frames"]) + 1,
            "num_bodies": int(lines[idx].strip()),
            "bodies": []
        }
        idx += 1

        for _ in range(frame_data["num_bodies"]):
            body = {
                "body_id": lines[idx].strip().split()[0],
                "joints": []
            }
            idx += 1

            num_joints = int(lines[idx].strip())
            idx += 1

            for _ in range(num_joints):
                joint_data = list(map(float, lines[idx].strip().split()))
                body["joints"].append({
                    "3D_position": {"x": joint_data[0], "y": joint_data[1], "z": joint_data[2]},
                    "orientation": {
                        "w": joint_data[7],
                        "x": joint_data[8],
                        "y": joint_data[9],
                        "z": joint_data[10]
                    },
                    "tracking_state": int(joint_data[11])
                })
                idx += 1

            frame_data["bodies"].append(body)

        data["frames"].append(frame_data)

    return data


def process_files(file_info):
    """Processa i file estratti e li salva in formato JSON."""
    for file_path, action_label in tqdm(file_info, desc="Parsing file", ncols=100):
        action_name = relevant_actions[action_label]
        parsed_data = parse_skeleton_file(file_path, action_label, action_name)
        if parsed_data:
            output_path = os.path.join(output_dir, os.path.basename(file_path).replace('.skeleton', '.json'))
            with open(output_path, 'w') as json_file:
                json.dump(parsed_data, json_file, indent=4)


if __name__ == "__main__":
    # Estrazione e filtro dei file
    extracted_files = extract_and_filter_files(zip_paths, relevant_actions)

    # Parsing e salvataggio in JSON
    process_files(extracted_files)

    print(f"Processamento completato! File salvati nella directory: {output_dir}")
