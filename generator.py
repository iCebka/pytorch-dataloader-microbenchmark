import os
import shutil
import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

BYTES_PER_MB = 1024 * 1024
FLOAT32_BYTES = 4
n = 60 # Only for styling in terminal outputs

# Groups all dataset-generation parameters in a single configuration object.
# This keeps the API cleaner and easier to extend.
@dataclass
class GeneratorConfig:
    output_dir: str
    total_size_mb: float
    min_size_mb: float
    max_size_mb: float
    seed: int = 70525386
    chunk_size_mb: float = 64.0
    dtype: str = "float32"
    reset_output_dir: bool=True

# Generate a synthetic on-disk dataset composed of binary float32 shards
def generate_dataset(config: GeneratorConfig) -> Dict[str, Any]:
    if config.dtype != "float32":
        raise ValueError("This release only supports float32")
    if config.total_size_mb <= 0:
        raise ValueError("total_size_mb must be positive")
    if config.min_size_mb <= 0 or config.max_size_mb <= 0:
        raise ValueError("min_size_mb and max_size_mb must be positive")
    if config.min_size_mb > config.max_size_mb:
        raise ValueError("min_size_mb cannot be greater than max_size_mb")
    if config.chunk_size_mb <= 0:
        raise ValueError("chunk_size_mb must be positive")
    
    # clean directory
    if config.reset_output_dir and os.path.exists(config.output_dir):
        shutil.rmtree(config.output_dir)
    os.makedirs(config.output_dir, exist_ok=True)

    rng = np.random.default_rng(config.seed)

    # Restrict sizes for float32
    total_size_bytes = int(config.total_size_mb * BYTES_PER_MB)
    total_size_bytes = (total_size_bytes // FLOAT32_BYTES) * FLOAT32_BYTES

    accumulated_bytes = 0
    file_index = 0
    generated_files = []

    # Chunked writing avoids allocating huge arrays in memory that could provoke RAM overflows
    chunk_size_bytes = int(config.chunk_size_mb * BYTES_PER_MB)
    chunk_size_bytes = max(FLOAT32_BYTES, (chunk_size_bytes // FLOAT32_BYTES) * FLOAT32_BYTES)

    print(f"Generating dataset in '{config.output_dir}'")
    print(f"Target size: {config.total_size_mb:.2f} MB")
    print(f"Per-file range: [{config.min_size_mb:.2f}, {config.max_size_mb:.2f}] MB")
    print(f"Seed: {config.seed}")
    print("-" * n)

    start_time = time.perf_counter() # time.time()
    
    while accumulated_bytes < total_size_bytes:
        target_file_mb = rng.uniform(config.min_size_mb, config.max_size_mb)
        target_file_bytes = int(target_file_mb * BYTES_PER_MB)

        remaining = total_size_bytes - accumulated_bytes
        target_file_bytes = min(target_file_bytes, remaining)
        target_file_bytes = (target_file_bytes // FLOAT32_BYTES) * FLOAT32_BYTES

        if target_file_bytes == 0:
            break

        file_name = f"data_shard_{file_index:04d}.bin"
        file_path = os.path.join(config.output_dir, file_name)

        bytes_written_for_file = 0

        with open(file_path, "wb") as f:
            while bytes_written_for_file < target_file_bytes:
                current_chunk_bytes = min(
                    chunk_size_bytes,
                    target_file_bytes - bytes_written_for_file,
                )
                current_chunk_bytes = (current_chunk_bytes // FLOAT32_BYTES) * FLOAT32_BYTES

                if current_chunk_bytes == 0:
                    break

                num_floats = current_chunk_bytes // FLOAT32_BYTES
                chunk_data = rng.random(num_floats, dtype=np.float32)
                f.write(chunk_data.tobytes())

                bytes_written_for_file += current_chunk_bytes
                accumulated_bytes += current_chunk_bytes
        
        generated_files.append(
            {
                "file_name": file_name,
                "file_path": file_path,
                "size_bytes": bytes_written_for_file,
                "size_mb": bytes_written_for_file / BYTES_PER_MB
            }
        )

        print(f"Generated {file_name} -> {bytes_written_for_file / BYTES_PER_MB:.2f} MB")
        file_index += 1
    
    end_time = time.perf_counter()
    total_time = end_time - start_time

    summary = {
        "output_dir": config.output_dir,
        "seed": config.seed,
        "dtype": config.dtype,
        "total_written_bytes": accumulated_bytes,
        "total_written_mb": accumulated_bytes / BYTES_PER_MB,
        "num_files": len(generated_files),
        "generation_time_sec": total_time,
        "files": generated_files,
    }

    print("-"*n)
    print(f"Generation completed in {total_time:.2f} s")
    print(f"Total written: {summary['total_written_mb']:.2f} MB across {summary['num_files']} files")

    return summary

if __name__ == "__main__":

    print("=" * n)
    print("Synthetic Dataset Generator Test")
    print("=" * n)

    config = GeneratorConfig(
        output_dir="./synthetic_data",
        total_size_mb=5120, # 5GB
        min_size_mb=100,
        max_size_mb=1024, # 1 GB
        seed=70525386,
        chunk_size_mb=64.0,
        dtype="float32",
        reset_output_dir=True
    )

    # Our execution becomes cleaner when we use configuration as an object instead of a function
    summary = generate_dataset(config)

    print("-" * n)
    print("Generation summary")
    print("-" * n)

    print(f"Output directory : {summary['output_dir']}")
    print(f"Files generated  : {summary['num_files']}")
    print(f"Total size (MB)  : {summary['total_written_mb']:.2f}")
    print(f"Generation time  : {summary['generation_time_sec']:.2f} s")

    print("=" * n)