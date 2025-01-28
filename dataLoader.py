import os
import json
import numpy as np
from typing import Optional, Callable, Tuple, Dict, List
import math
from torch.utils.data import Dataset, DataLoader
import logging
import torch

class SkeletonSequenceDataset(Dataset):
    def __init__(
        self,
        sequence_dir: str,
        label_file: str,
        label_map: Dict[int, int],
        sequence_length: int = 30,
        max_skeletons: int = 4,
        transform: Optional[Callable] = None,
        pad_mode: str = 'zero',
        logger: Optional[logging.Logger] = None
    ):
        super().__init__()
        self.sequence_dir = sequence_dir
        self.sequence_length = sequence_length
        self.label_map = label_map
        self.max_skeletons = max_skeletons
        self.transform = transform
        self.pad_mode = pad_mode
        self.logger = logger or logging.getLogger(__name__)
        
        # Validate inputs
        if not os.path.exists(sequence_dir):
            raise ValueError(f"Sequence directory does not exist: {sequence_dir}")
        if not os.path.exists(label_file):
            raise ValueError(f"Label file does not exist: {label_file}")

        # Load labels and sequence files
        self.labels = self._load_labels(label_file)
        self.sequence_files = self._get_sequence_files()
        self.sequence_paths = self._map_sequence_paths()
    
    def _load_labels(self, label_file: str) -> Dict:
        """Load and validate the label file."""
        try:
            with open(label_file, 'r') as f:
                labels = json.load(f)
            
            if not isinstance(labels, dict):
                raise ValueError("Label file must contain a dictionary")
            return labels
            
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in label file: {label_file}")
    
    def _get_sequence_files(self) -> List[str]:
        """Get all valid sequence files with labels."""
        all_sequences = []
        for root, _, files in os.walk(self.sequence_dir):
            for file in files:
                if file.endswith('.npy') and file in self.labels:
                    all_sequences.append(file)
                    
        if not all_sequences:
            raise ValueError("No labeled sequences found in the dataset")
            
        self.logger.info(f"Found {len(all_sequences)} labeled sequences")
        return sorted(all_sequences)  # Sort for reproducibility
    
    def _map_sequence_paths(self) -> Dict[str, str]:
        """Create a mapping of sequence filenames to their full paths."""
        sequence_paths = {}
        for root, _, files in os.walk(self.sequence_dir):
            for file in self.sequence_files:
                if file in files:
                    sequence_paths[file] = os.path.join(root, file)
        return sequence_paths
    
    def _pad_skeletons(self, frame: np.ndarray) -> np.ndarray:
        """Pad or truncate the number of skeletons in a frame."""
        num_skeletons = frame.shape[0]
        
        if num_skeletons == self.max_skeletons:
            return frame
            
        if num_skeletons > self.max_skeletons:
            return frame[:self.max_skeletons]
            
        pad_shape = (self.max_skeletons - num_skeletons,) + frame.shape[1:]
        padding = np.zeros(pad_shape)
        return np.concatenate((frame, padding), axis=0)
    
    def _pad_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """Pad or truncate sequence to target length, handling variable skeleton counts."""
        curr_length = sequence.shape[0]
        
        if curr_length > self.sequence_length:
            sequence = sequence[:self.sequence_length]
        elif curr_length < self.sequence_length:
            pad_length = self.sequence_length - curr_length
            pad_shape = (pad_length,) + sequence.shape[1:]
            
            if self.pad_mode == 'zero':
                padding = np.zeros(pad_shape)
            elif self.pad_mode == 'replicate':
                padding = np.repeat(sequence[-1:], pad_length, axis=0)
            elif self.pad_mode == 'reflect':
                padding = sequence[-(pad_length + 1):-1][::-1]
            else:
                raise ValueError(f"Unknown padding mode: {self.pad_mode}")
                
            sequence = np.concatenate((sequence, padding), axis=0)
        
        padded_frames = []
        for frame in sequence:
            padded_frame = self._pad_skeletons(frame)
            padded_frames.append(padded_frame)
            
        return np.stack(padded_frames)
    
    def __len__(self) -> int:
        return len(self.sequence_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence_file = self.sequence_files[idx]
        
        try:
            sequence = np.load(self.sequence_paths[sequence_file])
            sequence = self._pad_sequence(sequence)
            
            if self.transform is not None:
                sequence = self.transform(sequence)
            
            label = self.labels[sequence_file]

            label = self.label_map.get(label, -1)

            if label == -1:
                self.logger.error(f"Invalid label {label} for sequence {sequence_file}")
            
            # Convert to PyTorch tensors
            sequence_tensor = torch.tensor(sequence, dtype=torch.float32)
            label_tensor = torch.tensor(label, dtype=torch.long)
            
            return sequence_tensor, label_tensor
        
        except Exception as e:
            self.logger.error(f"Error processing sequence {sequence_file}: {str(e)}")
            raise

def get_dataloader(sequence_dir, label_file, label_map, batch_size=32, num_workers=4):
    dataset = SkeletonSequenceDataset(sequence_dir, label_file,label_map, sequence_length=30, max_skeletons=2, )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("SkeletonDataset")
    
    # Create dataset
    dataset = SkeletonSequenceDataset(
        sequence_dir="sequences",
        label_file="sequences/labels.json",
        label_map ={
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
        sequence_length=30,
        max_skeletons=4,
        logger=logger
    )
    
    batch_size = 32
    total_sequences = len(dataset)
    total_batches = math.ceil(total_sequences / batch_size)
    
    print(f"\nDataset Statistics:")
    print(f"Total sequences: {total_sequences}")
    print(f"Batch size: {batch_size}")
    print(f"Total number of batches: {total_batches}")
    print("\n" + "="*50 + "\n")
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Test iteration
    for batch_idx, (sequences, labels) in enumerate(dataloader):
        # Print first 3 batches
        if batch_idx < 3:
            print(f"Batch {batch_idx}/{total_batches-1}")
            print(f"Sequences shape: {sequences.shape}")
            print(f"Number of labels: {len(labels)}")
            print(f"First few labels: {labels[:5]}...")
            print("\n" + "-"*50 + "\n")
            
        # Print last 2 batches
        elif batch_idx >= total_batches - 2:
            print(f"Batch {batch_idx}/{total_batches-1}")
            print(f"Sequences shape: {sequences.shape}")
            print(f"Number of labels: {len(labels)}")
            print(f"First few labels: {labels[:5]}...")
            print("\n" + "-"*50 + "\n")
            
        # Print a progress indicator every 50 batches
        elif batch_idx % 50 == 0:
            print(f"Processing batch {batch_idx}/{total_batches-1}")