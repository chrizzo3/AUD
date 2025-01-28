import os
import json
import numpy as np
import cv2
import torch
import mediapipe as mp
from math import atan2, degrees
from CNN_GRUModel import CNNGRUModel  # Ensure this import matches your model definition

# Constants
num_classes = 15
sequence_length = 30
num_joints = 25
max_skeletons = 2
feature_size = 9  # Adjust based on your feature size

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

joint_order_map = [
    "SPINEBASE", "SPINEMID", "NECK", "HEAD",
    "SHOULDERLEFT", "ELBOWLEFT", "WRISTLEFT", "HANDLEFT",
    "SHOULDERRIGHT", "ELBOWRIGHT", "WRISTRIGHT", "HANDRIGHT",
    "HIPLEFT", "KNEELEFT", "ANKLELEFT", "FOOTLEFT",
    "HIPRIGHT", "KNEERIGHT", "ANKLERIGHT", "FOOTRIGHT",
    "SPINESHOULDER", "HANDTIPLEFT", "THUMBLEFT", "HANDTIPRIGHT", "THUMBRIGHT"
]

mediapipe_to_custom_map = {
    0: "SPINEBASE",  # Pelvis
    1: "SPINESHOULDER",  # Lower Spine
    2: "NECK",  # Neck
    3: "HEAD",  # Head
    11: "SHOULDERLEFT",  # Left Shoulder
    13: "ELBOWLEFT",  # Left Elbow
    15: "WRISTLEFT",  # Left Wrist
    21: "HANDTIPLEFT",  # Left Hand Tip
    17: "HANDLEFT",  # Left Hand
    19: "THUMBLEFT",  # Left Thumb
    12: "SHOULDERRIGHT",  # Right Shoulder
    14: "ELBOWRIGHT",  # Right Elbow
    16: "WRISTRIGHT",  # Right Wrist
    22: "HANDTIPRIGHT",  # Right Hand Tip
    18: "HANDRIGHT",  # Right Hand
    20: "THUMBRIGHT",  # Right Thumb
    23: "HIPLEFT",  # Left Hip
    25: "KNEELEFT",  # Left Knee
    27: "ANKLELEFT",  # Left Ankle
    31: "FOOTLEFT",  # Left Foot
    24: "HIPRIGHT",  # Right Hip
    26: "KNEERIGHT",  # Right Knee
    28: "ANKLERIGHT",  # Right Ankle
    32: "FOOTRIGHT"  # Right Foot
}

action_labels = {
    0: "Pick up",
    1: "Throw",
    2: "Kicking something",
    3: "Reach into pocket",
    4: "Point to something",
    5: "Shake fist",
    6: "Staggering",
    7: "Falling down",
    8: "Punch/slap",
    9: "Kicking",
    10: "Pushing",
    11: "Touch pocket",
    12: "Hit with object",
    13: "Wield knife",
    14: "Shoot with gun"
}

def reorder_joints(skeleton):
    """Riordina i joints dello scheletro seguendo il mapping specifico."""
    reordered_skeleton = []
    for target_joint in joint_order_map:
        joint_idx = next((k for k, v in mediapipe_to_custom_map.items() if v == target_joint), None)
        if joint_idx is not None and joint_idx < len(skeleton):
            reordered_skeleton.append(skeleton[joint_idx])
        else:
            reordered_skeleton.append({"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.0})
    return reordered_skeleton

def calculate_joint_angle(joint1, joint2, joint3):
    vector1 = np.array([joint1['3D_position']['x'] - joint2['3D_position']['x'], 
                        joint1['3D_position']['y'] - joint2['3D_position']['y'], 
                        joint1['3D_position']['z'] - joint2['3D_position']['z']])
    vector2 = np.array([joint3['3D_position']['x'] - joint2['3D_position']['x'], 
                        joint3['3D_position']['y'] - joint2['3D_position']['y'], 
                        joint3['3D_position']['z'] - joint2['3D_position']['z']])
    
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    if norm1 == 0 or norm2 == 0:
        return 0.0

    dot_product = np.dot(vector1, vector2)
    angle = degrees(atan2(np.linalg.norm(np.cross(vector1, vector2)), dot_product))
    return angle

def calculate_joint_rotation(joints):
    rotations = []
    for joint in joints:
        rotation = joint.get('rotation', [0, 0, 0])
        if len(rotation) < 3:
            rotation = rotation + [0] * (3 - len(rotation))
        rotations.append({"rx": rotation[0], "ry": rotation[1], "rz": rotation[2]})
    return rotations

def extract_skeletons_with_rotation(video_path, output_json_path):
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose(model_complexity=2, enable_segmentation=False, min_detection_confidence=0.5)

    skeleton_data = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        frame_bodies = []
        if results.pose_landmarks:
            joints = [
                {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
                for lm in results.pose_landmarks.landmark
            ]

            # Reorder joints
            joints = reorder_joints(joints)

            rotations = calculate_joint_rotation(joints)

            joints_data = [
                {
                    "3D_position": {"x": joint["x"], "y": joint["y"], "z": joint["z"]},
                    "orientation": {"w": 1, "x": 0, "y": 0, "z": 0},  # Placeholder for actual orientation data
                    "tracking_state": 2 if joint["visibility"] > 0.5 else 1
                }
                for joint, rotation in zip(joints, rotations)
            ]

            frame_bodies.append({"joints": joints_data})

        skeleton_data.append({
            "frame_index": frame_idx + 1,
            "num_bodies": len(frame_bodies),
            "bodies": frame_bodies
        })
        frame_idx += 1

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow('MediaPipe Pose', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    pose.close()
    cv2.destroyAllWindows()

    skeleton_data_dict = {"frames": skeleton_data}

    with open(output_json_path, "w") as f:
        json.dump(skeleton_data_dict, f, indent=4)

    print(f"Scheletri con rotazioni salvati in: {output_json_path}")
    return skeleton_data_dict

def create_sequences(skeleton_data, sequence_length, num_joints, max_skeletons):
    sequences = []
    num_frames = len(skeleton_data["frames"])
    feature_size = 9  # Changed from 18 to 9

    for i in range(0, num_frames - sequence_length + 1, sequence_length):
        sequence = np.zeros((sequence_length, max_skeletons, num_joints, feature_size))
        for t, frame_data in enumerate(skeleton_data["frames"][i:i + sequence_length]):
            for skel_idx, body in enumerate(frame_data['bodies'][:max_skeletons]):
                joints = body['joints']
                for joint_idx, joint in enumerate(joints[:num_joints]):
                    # Position (0-2)
                    sequence[t, skel_idx, joint_idx, 0:3] = [
                        joint['3D_position']['x'],
                        joint['3D_position']['y'],
                        joint['3D_position']['z']
                    ]
                    # Orientation (3-6)
                    sequence[t, skel_idx, joint_idx, 3:7] = [
                        joint['orientation']['w'],
                        joint['orientation']['x'],
                        joint['orientation']['y'],
                        joint['orientation']['z']
                    ]
                    # State (7)
                    sequence[t, skel_idx, joint_idx, 7] = joint['tracking_state']
                    # Confidence (8)
                    sequence[t, skel_idx, joint_idx, 8] = joint.get('confidence', 1.0)

        sequences.append(sequence)
    return sequences

def rotate_skeleton(skeleton, angle):
    """Rotate the skeleton by a given angle around the y-axis."""
    rotation_matrix = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])
    rotated_skeleton = []
    for joint in skeleton:
        if isinstance(joint, dict) and '3D_position' in joint:
            position = np.array([joint['3D_position']['x'], joint['3D_position']['y'], joint['3D_position']['z']])
            rotated_position = rotation_matrix.dot(position)
            rotated_joint = joint.copy()
            rotated_joint['3D_position']['x'] = rotated_position[0]
            rotated_joint['3D_position']['y'] = rotated_position[1]
            rotated_joint['3D_position']['z'] = rotated_position[2]
            rotated_skeleton.append(rotated_joint)
        else:
            rotated_skeleton.append(joint)
    return rotated_skeleton

def predict_actions(sequences, model):
    predictions = []
    model.eval()
    with torch.no_grad():
        for sequence in sequences:
            all_predictions = []
            best_prediction = None
            best_confidence = -1
            best_skeleton = None
            for angle in [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]:
                rotated_sequence = np.array([rotate_skeleton(frame, angle) for frame in sequence])
                input_tensor = torch.tensor(rotated_sequence, dtype=torch.float32).unsqueeze(0)
                output = model(input_tensor)
                confidence, predicted_class = torch.max(output, dim=1)
                all_predictions.append((predicted_class.item(), confidence.item(), angle))
                if confidence.item() > best_confidence:
                    best_confidence = confidence.item()
                    best_prediction = predicted_class.item()
                    best_skeleton = angle
            predictions.append((all_predictions, best_prediction, best_confidence, best_skeleton))
    return predictions

def process_and_infer_with_overlay(video_path, output_video_path, output_json_path):
    print("Estrazione degli scheletri...")
    skeleton_data = extract_skeletons_with_rotation(video_path, output_json_path)

    print("Creazione delle sequenze...")
    sequences = create_sequences(skeleton_data, sequence_length, num_joints, max_skeletons)

    print("Caricamento del modello...")
    model = CNNGRUModel(
        input_channels=9, 
        num_joints=25, 
        cnn_out_channels=64, 
        gru_hidden_size=128, 
        num_classes=num_classes
    )
    model_path = "models/cnn_gru_model.pth"  # Adjust the path to your model
    model.load_state_dict(torch.load(model_path))
    model.eval()

    print("Inferenza sulle sequenze...")
    predictions = predict_actions(sequences, model)

    

    for all_preds, best_pred, best_conf, best_skel in predictions:
        print("All predictions:")
        for pred, conf, skel in all_preds:
            print(f"Predicted action: {action_labels.get(pred, 'Non Dangerous Action')} with confidence: {conf:.2f} from skeleton: {skel}")
        print(f"Best prediction: {action_labels.get(best_pred, 'Non Dangerous Action')} with confidence: {best_conf:.2f} from skeleton: {best_skel}")

    print("Sovrapposizione delle azioni sul video...")
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    frame_idx = 0
    seq_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx // sequence_length == seq_idx and seq_idx < len(predictions):
            pred_info = predictions[seq_idx]
            action_id = pred_info['prediction']
            confidence = pred_info['confidence']
            action_name = action_labels.get(action_id, "Unknown Action")
            cv2.putText(
                frame, 
                f"Action: {action_name})", 
                (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )

        out.write(frame)
        frame_idx += 1
        if frame_idx % sequence_length == 0:
            seq_idx += 1

    cap.release()
    out.release()
    print(f"Video con sovraimpressione salvato in: {output_video_path}")
    os.startfile(output_video_path)

# Percorsi
video_path = "video/punching.mp4"
output_video_path = "output_with_predictions.mp4"
output_json_path  = "skeleton_with_rotation.json"

# Esegui la pipeline con sovraimpressione
process_and_infer_with_overlay(video_path, output_video_path, output_json_path)