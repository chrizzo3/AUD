import json
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class SkeletonProcessor:
    def __init__(
        self,
        json_dir: str,
        sequence_dir: str,
        sequence_length: int = 30,
        num_joints: int = 25,
        max_bodies: int = 2,
        feature_size: int = 9,  # x, y, z, orientation (w,x,y,z), tracking_state, confidence
        train_ratio: float = 0.8
    ):
        self.json_dir = Path(json_dir)
        self.sequence_dir = Path(sequence_dir)
        self.sequence_length = sequence_length
        self.num_joints = num_joints
        self.max_bodies = max_bodies
        self.feature_size = feature_size
        self.train_ratio = train_ratio
        
        # Create output directories
        self.sequence_dir.mkdir(parents=True, exist_ok=True)
        (self.sequence_dir / "1_skeleton").mkdir(exist_ok=True)
        (self.sequence_dir / "2_skeletons").mkdir(exist_ok=True)
        (self.sequence_dir / "more_than_2_skeletons").mkdir(exist_ok=True)
        
        # Statistics
        self.stats = {
            'processed_files': 0,
            'skipped_files': 0,
            'errors': 0,
            'body_counts': {},
            'zero_body_files' : []
        }

    def load_json(self, file_path: Path) -> Optional[Dict]:
        """Load and validate a JSON file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            if not isinstance(data, dict) or 'frames' not in data:
                logging.warning(f"Invalid JSON structure in {file_path}")
                return None
            return data
        except json.JSONDecodeError:
            logging.error(f"Failed to decode JSON file: {file_path}")
            return None
        except Exception as e:
            logging.error(f"Error reading {file_path}: {str(e)}")
            return None

    def interpolate_missing_frames(self, sequence: np.ndarray) -> np.ndarray:
        """Interpolate missing frames using linear interpolation."""
        for body_idx in range(sequence.shape[1]):
            for joint_idx in range(sequence.shape[2]):
                # Find frames where tracking state is 0 (not tracked)
                missing_frames = sequence[:, body_idx, joint_idx, -1] == 0
                if not np.any(missing_frames):
                    continue
                
                # Interpolate each feature
                for feature_idx in range(self.feature_size - 1):  # Exclude tracking state
                    valid_frames = ~missing_frames
                    if np.any(valid_frames):
                        sequence[missing_frames, body_idx, joint_idx, feature_idx] = np.interp(
                            np.where(missing_frames)[0],
                            np.where(valid_frames)[0],
                            sequence[valid_frames, body_idx, joint_idx, feature_idx]
                        )
        return sequence

    def create_sequence(self, frames: List[Dict]) -> np.ndarray:
        sequence = np.zeros((
            self.sequence_length,
            self.max_bodies,
            self.num_joints,
            self.feature_size
        ), dtype=np.float32)
        
        for i, frame in enumerate(frames):
            if i >= self.sequence_length:
                break  # Ensure we do not exceed the sequence length
            for b, body in enumerate(frame.get('bodies', [])[:self.max_bodies]):
                for j, joint in enumerate(body.get('joints', [])[:self.num_joints]):
                    # Position (0-2)
                    sequence[i, b, j, 0:3] = [
                        joint['3D_position']['x'],
                        joint['3D_position']['y'],
                        joint['3D_position']['z']
                    ]
                    # Orientation (3-6)
                    sequence[i, b, j, 3:7] = [
                        joint['orientation']['w'],
                        joint['orientation']['x'],
                        joint['orientation']['y'],
                        joint['orientation']['z']
                    ]
                    # State (7)
                    sequence[i, b, j, 7] = joint['tracking_state']
                    # Confidence (8)
                    sequence[i, b, j, 8] = joint.get('confidence', 1.0)
        
        return sequence
    
    def get_output_folder(self, num_bodies: int) -> Path:
        """Determine output folder based on number of bodies."""
        if num_bodies == 1:
            folder = "1_skeleton"
        elif num_bodies == 2:
            folder = "2_skeletons"
        else:
            folder = "more_than_2_skeletons"
        
        return self.sequence_dir / folder

    def process_single_file(self, json_file: Path):
        result = {'processed_files': 0, 'skipped_files': 0, 'errors': 0, 'body_counts': {}, 'zero_body_files': [], 'label':{}}
        try:
            data = self.load_json(json_file)
            if data is None:
                result['skipped_files'] += 1
                return result

            num_bodies = max(len(frame.get('bodies', [])) for frame in data['frames'])
            result['body_counts'][num_bodies] = result['body_counts'].get(num_bodies, 0) + 1

            if num_bodies == 0:
                result['zero_body_files'].append(json_file.name)
                return result

            result['label'][json_file.stem + ".npy"] = data.get('action_label', 'unknown')

            sequence = self.create_sequence(data['frames'])
            output_folder = self.get_output_folder(num_bodies)
            output_path = output_folder / f"{json_file.stem}.npy"
            np.save(output_path, sequence)

            result['processed_files'] += 1
        except Exception as e:
            logging.error(f"Error processing {json_file}: {str(e)}")
            result['errors'] += 1
        return result

    def process_files(self):
        #Processa tutti i file JSON in parallelo.
        with Manager() as manager:
            label_dict = {}
            json_files = list(self.json_dir.glob("*.json"))

            with ProcessPoolExecutor() as executor:
                all_results =  list(tqdm(executor.map(self.process_single_file, json_files), total=len(json_files), desc="Processing skeleton files"))
            
            for result in all_results:
                self.stats['processed_files'] += result['processed_files']
                self.stats['skipped_files'] += result['skipped_files']
                self.stats['errors'] += result['errors']
                for num_bodies, count in result['body_counts'].items():
                    self.stats['body_counts'][num_bodies] = self.stats['body_counts'].get(num_bodies, 0) + count
                self.stats['zero_body_files'].extend(result['zero_body_files'])
                label_dict.update(result['label'])

            with open(self.sequence_dir / "labels.json", "w") as f:
                json.dump(label_dict, f, indent=4)
        
    def print_statistics(self):
        """Print processing statistics."""
        logging.info("\nProcessing Statistics:")
        logging.info(f"Processed files: {self.stats['processed_files']}")
        logging.info(f"Skipped files: {self.stats['skipped_files']}")
        logging.info(f"Errors encountered: {self.stats['errors']}")
        logging.info("\nBody count distribution:")
        for num_bodies, count in sorted(self.stats['body_counts'].items()):
            logging.info(f"{num_bodies} bodies: {count} files")
        
        # Print files with zero bodies
        if self.stats['zero_body_files']:
            logging.info("\nFiles with zero bodies:")
            for filename in self.stats['zero_body_files']:
                logging.info(filename)
    
    

if __name__ == "__main__":
    # Configuration
    CONFIG = {
        'json_dir': "processed_skeletons",
        'sequence_dir': "sequences",
        'sequence_length': 30,
        'num_joints': 25,
        'max_bodies': 2,
        'feature_size': 9
    }
    
    # Initialize and run processor
    processor = SkeletonProcessor(**CONFIG)
    processor.process_files()
    processor.print_statistics()