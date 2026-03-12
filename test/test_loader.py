import os
import torch
import numpy as np
import bisect
import time 

from torch.utils.data import Dataset, DataLoader

"""
Personalized Pytorch dataset class to read generated binary files
"""
class BinaryShardedDataset(Dataset):

    def __init__(self, data_dir: str, sample_size_floats: int = 1024):
        # data_dir: dataset directory
        # sample_size_floats: how many float32 compose a sample. Ej: 1024 floats = 4KB per sample

        self.data_dir = data_dir
        self.sample_size_floats = sample_size_floats
        self.sample_size_bytes = sample_size_floats * 4 # float32 = 4 bytes

        self.file_records = [] # (path_file, # samples)
        self.cumulative_samples = [] # For binary search execution
        self.total_samples = 0

        # Scanning the directory to set up the index map
        # Not included in the timing
        for f in sorted(os.listdir(data_dir)):
            if f.endswith('.bin'):
                path = os.path.join(data_dir, f)
                size_bytes = os.path.getsize(path)

                # Calculate how many complete samples could be stored in this file 
                # non-complete are ignored 
                samples_in_file = size_bytes // self.sample_size_bytes

                if samples_in_file > 0:
                    self.file_records.append(path)
                    self.total_samples += samples_in_file
                    self.cumulative_samples.append(self.total_samples)
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        # Find a file given an idx
        file_idx = bisect.bisect_right(self.cumulative_samples, idx)

        # Calculate the local index inside the file
        if file_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_samples[file_idx - 1]

        # Compute byte offset
        file_path = self.file_records[file_idx]
        offset_bytes = local_idx * self.sample_size_bytes

        # Key operation: Here the I/O operation is performed!
        with open(file_path, 'rb') as f:
            f.seek(offset_bytes) # Direct access to the needed byte
            raw_bytes = f.read(self.sample_size_bytes)
        
        # Raw bytes to Pytorch Tensor
        array = np.frombuffer(raw_bytes, dtype=np.float32)
        tensor = torch.from_numpy(np.copy(array))

        return tensor

def create_dataloader(
        data_dir: str,
        batch_size: int=64,
        num_workers: int=0
):
    dataset = BinaryShardedDataset(data_dir=data_dir)

    loader = DataLoader(
        dataset,
        batch_size=batch_size, # Fixed in our example
        shuffle=False, # True in shuffle could provoke non-determinism in throughput results
        num_workers=num_workers,
        #pin_memory=True # Usual in ML worklows when transfering to GPU
    )

    return loader

if __name__ == "__main__":
    
    n = 50

    DATA_DIR = "./synthetic_data"
    SAMPLE_SIZE = 1024 # Each sample is a vector of 1024 floats (4KB)

    print("-"*n)
    print("Testing Dataset class")
    print("-"*n)

    print(f"Instantiating Dataset from {DATA_DIR}")
    dataset = BinaryShardedDataset(data_dir=DATA_DIR, sample_size_floats=SAMPLE_SIZE)

    print(f"-> Total samples: {len(dataset)}")
    print(f"-> Mapped files: {len(dataset.file_records)}")
    
    for i, f in enumerate(dataset.file_records[:]):
        print(f"  {i}: {f}")
    # Random access with get
    
    print("-"*n)
    print(f"Accessing an individual sample (index=500)")
    sample = dataset[500]
    print(f"-> Tensor shape: {sample.shape}")
    print(f"-> Data type: {sample.dtype}")
    print(f"-> Values: {sample[:4]}")

    # DataLoader
    BATCH_SIZE = 64
    NUM_WORKERS = 1
    print("-"*n)
    print(f"Creating DataLoader (batch_size: {BATCH_SIZE}, workers: {NUM_WORKERS})")

    loader = create_dataloader(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    print("Successful creation")

    print(loader)
    # Batching
    # Only first 3 iterations
    
    print("-"*n)
    print(f"Test iteration (only 3 first iterations): ")
    for i, batch in enumerate(loader):
        if i >= 3: break

        # Each batch is a tensor groupping 64 samples
        print(f" ** Batch {i}:")
        print(f"Batch shape: {batch.shape}")
        print(f"Device: {batch.device}")
        print(f"Mean of values: {batch.mean():.4f}")

    print("-"*n)
    print("100 batches timing")
    start = time.time()

    for i, batch in enumerate(loader):
        if i >= 100:
            break
        print(f" ** Batch {i}")

    end = time.time()

    print(f"Time for 100 batches: {end-start:.3f}s")
                


