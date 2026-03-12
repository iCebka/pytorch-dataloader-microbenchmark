import bisect
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import List, Dict, Any

FLOAT32_BYTES = 4
n = 60 # Terminal styling

# Groups the dataset-reading parameters in a single configuration object.
@dataclass
class DatasetConfig:
    data_dir: str
    sample_size_floats: int = 1024

# Map-style PyTorch dataset over a directory of raw float32 binary shards
# Each sample is interpreted as a fixed-size block of sample_size_floats
# consecutive float32 values
class BinaryShardedDataset(Dataset):
    def __init__(self, config: DatasetConfig):
        self.data_dir = config.data_dir
        self.sample_size_floats = config.sample_size_floats
        self.sample_size_bytes = self.sample_size_floats * FLOAT32_BYTES

        if self.sample_size_floats <= 0:
            raise ValueError("sample_size_floats must be positive")
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")
        
        self.file_records: List[str] = []
        self.cumulative_samples: List[int] = []
        self.total_samples = 0
        
        # Scan the directory and build a global sample index across all shards
        for f in sorted(os.listdir(self.data_dir)):
            if not f.endswith(".bin"):
                continue

            path = os.path.join(self.data_dir, f)
            size_bytes = os.path.getsize(path)

            # Only full samples are considered; incomplete tail bytes are ignored
            # This should be changed in the future to consider or notify this issues to the user
            samples_in_file = size_bytes // self.sample_size_bytes

            if samples_in_file > 0:
                self.file_records.append(path)
                self.total_samples += samples_in_file
                self.cumulative_samples.append(self.total_samples)
        
        if self.total_samples == 0:
            raise ValueError(
                f"No .bin files found"
            )
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(f"Sample index outt of range: {idx}")
        
        # Locate which shard contains the requested global sample index using binary search
        file_idx = bisect.bisect_right(self.cumulative_samples, idx)

        # Convert global index to local index inside the selected file
        if file_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_samples[file_idx - 1]
        
        file_path = self.file_records[file_idx]
        offset_bytes = local_idx * self.sample_size_bytes

        # Perform random-access read for the requested sample
        with open(file_path, "rb") as f:
            f.seek(offset_bytes)
            raw_bytes = f.read(self.sample_size_bytes)
        
        if len(raw_bytes) != self.sample_size_bytes:
            raise IOError(
                f"Short read in file {file_path} "
                f"Expected {self.sample_size_bytes} bytes, got {len(raw_bytes)}"
            )
        
        array = np.frombuffer(raw_bytes, dtype=np.float32)
        return torch.from_numpy(np.copy(array))
    
    def describe(self) -> Dict[str, Any]:
        return {
            "data_dir": self.data_dir,
            "sample_size_floats": self.sample_size_floats,
            "sample_size_bytes": self.sample_size_bytes,
            "num_files": len(self.file_records),
            "total_samples": self.total_samples,
            "files": list(self.file_records),
        }

def create_dataloader(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 0,
    sample_size_floats: int = 1024,
    shuffle: bool = False,
    #pin_memory: bool = False
):
    dataset = BinaryShardedDataset(
        DatasetConfig(data_dir=data_dir, sample_size_floats=sample_size_floats)
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        #pin_memory=pin_memory,
    )

if __name__ == "__main__":

    print("=" * n)
    print("BinaryShardedDataset Test")
    print("=" * n)

    config = DatasetConfig(
        data_dir="./synthetic_data",
        sample_size_floats=1024
    )

    dataset = BinaryShardedDataset(config)
    info = dataset.describe()

    print("Dataset summary")
    print("-" * n)
    print(f"Directory           : {info['data_dir']}")
    print(f"Sample size (floats): {info['sample_size_floats']}")
    print(f"Sample size (bytes) : {info['sample_size_bytes']}")
    print(f"Mapped files        : {info['num_files']}")
    print(f"Total samples       : {info['total_samples']}")

    print("-" * n)
    print("Mapped files")
    for i, path in enumerate(info["files"][:]):
        print(f"{i}: {path}")

    sample_idx = min(500, len(dataset) - 1)
    sample = dataset[sample_idx]

    print("-" * n)
    print(f"Preview sample at index {sample_idx}")
    print(f"Tensor shape : {tuple(sample.shape)}")
    print(f"Tensor dtype : {sample.dtype}")
    print(f"First values : {sample[:8]}")

    print("-" * n)
    print("DataLoader test")
    loader = create_dataloader(
        data_dir="./synthetic_data",
        batch_size=64,
        num_workers=1,
        sample_size_floats=1024,
        shuffle=False,
    )

    for i, batch in enumerate(loader):
        if i >= 3:
            break
        print(f"Batch {i}")
        print(f"  shape: {tuple(batch.shape)}")
        print(f"  dtype: {batch.dtype}")
        print(f"  mean : {batch.mean().item():.6f}")

    print("=" * n)