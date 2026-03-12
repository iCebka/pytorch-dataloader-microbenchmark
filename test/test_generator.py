import os
import shutil
import numpy as np
import math
import time

"""
Generates a dataset in pure binary format (.bin) files.
Args:
    output_dir: Output Directory
    total_size_mb: Desired size for the dataset in mb
    min_size_mb: Minimum size per file in mb
    max_size_mb: Maximum size per file in mb
    seed: Seed for random generation
    chunk_size_mb: Chunk size for sequential generation to avoid RAM overflow while generating files
"""
def generate_dataset(
    output_dir: str,
    total_size_mb: float,
    min_size_mb: float,
    max_size_mb: float,
    seed: int = 42,
    chunk_size_mb: float = 64.0
):
    # Set up
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    rng = np.random.default_rng(seed)

    bytes_per_mb = 1024 * 1024
    total_size_bytes = int(total_size_mb * bytes_per_mb)
    accumulated_bytes = 0
    file_index = 0

    print(f"Beginning generation: Tarjet {total_size_mb} MB in '{output_dir}'")
    start_time = time.time()
    
    # Main Loop
    while accumulated_bytes < total_size_bytes:

        # Choose size randomly with a uniform distribution in [a,b]
        target_file_mb = rng.uniform(min_size_mb, max_size_mb)
        target_file_bytes = int(target_file_mb * bytes_per_mb)

        # Check if there is still space
        if accumulated_bytes + target_file_bytes > total_size_bytes:
            target_file_bytes = total_size_bytes - accumulated_bytes
        
        file_name = f"data_shard_{file_index:04d}.bin"
        file_path = os.path.join(output_dir, file_name)

        # Writing in chunks to avoid problems with RAM
        chunk_size_bytes = int(chunk_size_mb * bytes_per_mb)
        bytes_written_for_file = 0

        with open(file_path, 'wb') as f:
            while bytes_written_for_file < target_file_bytes:
                # Calculate current chunk size
                current_chunk_bytes = min(chunk_size_bytes, target_file_bytes - bytes_written_for_file)

                # Will change in future
                # Here we are assuming float32 (4 bytes per number), calculate how many floats are we generating
                num_floats = current_chunk_bytes // 4

                # Generate random data and write directly in binary
                chunk_data = rng.random(num_floats, dtype=np.float32)
                f.write(chunk_data.tobytes())

                bytes_written_for_file += current_chunk_bytes
                accumulated_bytes += current_chunk_bytes
            
            print(f"Generated {file_name} -> {bytes_written_for_file / bytes_per_mb:.2f} MB")
            file_index += 1

    end_time = time.time()
    print("-"*40)
    print(f"Successful generation in {end_time - start_time:.2f} seconds")
    print(f"Total written: {accumulated_bytes / bytes_per_mb:.2f} MB in {file_index} files.")

# Example
if __name__ == "__main__":
    generate_dataset(
        output_dir = "./synthetic_data",
        total_size_mb=5120,
        min_size_mb=100,
        max_size_mb=1024,
        seed=70525386
    )