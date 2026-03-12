# PyTorch DataLoader I/O Microbenchmark

A reproducible microbenchmark for exploring I/O performance of PyTorch DataLoader pipelines using synthetic binary datasets.

This small project was developed as part of an exploration on ML data loading performance at scale, focusing on the interaction between:

- dataset layout
- shard size
- worker paralelism
- file I/O patterns

The repository provides a simple CLI equiped with tools to:
- generate synthetic datasets composed of binary shards
- inspect and validate dataset structure
- run small pipeline tests
- benchmark PyTorch DataLoader performance across different worker configurations

## Project structure

```
pytorch-dataloader-microbenchmark
|__ README.md
|
|__ cli.py
|__ generator.py
|__ dataset.py
|__ benchmark.py
|
|__ test/
|    |__ README.md
|    |__ test_benchmark.py
|    |__ test_generator.py
|    |__ test_loader.py
|    |__ test_read.py
|
|_requirements.txt
```

## Installation

### Requirements
- Python 3.9+
- PyTorch
- Numpy
- Pandas

Install dependencies:

`
pip install torch numpy pandas
`

#### Optional: Environment Setup
Create and activate a virtual environment:
```
python -m venv .venv
```
Activate the environment
Linux/macOS:
```
source .venv/bin/activate
```
Windows:
```
.venv\Scripts\activate
```
Then install the dependencies:
```
pip install -r requirements.txt
```

## Workflow

The tool follows a simple pipeline:

`
generate dataset > inspect dataset > smoke test > benchmark 
`

### 1. Generate Synthetic Dataset

Creates a dataset composed of binary shards containing `float32` values.

#### Example:
```
python cli.py generate \
--output-dir ./synthetic_data \
--total-size-mb 1024 \
--min-size-mb 50 \
--max-size-mb 100
```

#### Example output:
```
Generating dataset in './synthetic_data' 
Target size: 1024.00 MB 
Per-file range: [50.00, 100.00] MB 
Generated data_shard_0000.bin -> 97.49 MB 
Generated data_shard_0001.bin -> 88.76 MB 
...
Generation completed in 5.80 s
```

### 2. Inspect Dataset

Print dataset metadata and verify shards.

#### Example:
```
python cli.py inspect --data-dir ./synthetic_data
```
#### Example output:
```
Dataset inspection 
------------------------------------------------------------ 
Directory: ./synthetic_data 
Sample size (floats): 1024 
Sample size (bytes): 4096 
Mapped files: 14 
Total samples: 262135
```
#### Show shard files:
```
python cli.py inspect --data-dir ./synthetic_data --show-files
```
#### Output:
```
Dataset inspection
------------------------------------------------------------
Directory: .\synthetic_data\
Sample size (floats): 1024
Sample size (bytes): 4096
Mapped files: 14
Total samples: 262135
------------------------------------------------------------
   0: .\synthetic_data\data_shard_0000.bin
   1: .\synthetic_data\data_shard_0001.bin
   2: .\synthetic_data\data_shard_0002.bin
   3: .\synthetic_data\data_shard_0003.bin
   4: .\synthetic_data\data_shard_0004.bin
   5: .\synthetic_data\data_shard_0005.bin
   6: .\synthetic_data\data_shard_0006.bin
   7: .\synthetic_data\data_shard_0007.bin
   8: .\synthetic_data\data_shard_0008.bin
   9: .\synthetic_data\data_shard_0009.bin
  10: .\synthetic_data\data_shard_0010.bin
  11: .\synthetic_data\data_shard_0011.bin
  12: .\synthetic_data\data_shard_0012.bin
  13: .\synthetic_data\data_shard_0013.bin
```

#### Preview a sample:
```
python cli.py inspect --data-dir ./synthetic_data --preview-index 20
```
### Output:
```
Dataset inspection
------------------------------------------------------------
Directory: .\synthetic_data\
Sample size (floats): 1024
Sample size (bytes): 4096
Mapped files: 14
Total samples: 262135
------------------------------------------------------------
------------------------------------------------------------
Preview sample index: 20
Tensor shape: (1024,)
Tensor dtype: torch.float32
First 8 values:
tensor([0.4265, 0.2321, 0.7954, 0.7478, 0.2403, 0.6423, 0.0771, 0.6382])
```

### 3. Smoke Test
Runs a small DataLoader test to verify if the pipeline works properly.

```
python cli.py smoke \
--data-dir ./synthetic_data \
--num-workers 1 \
--num-batches 3
```

#### Example Output:
```
Batch 0
  shape: (64, 1024)
  dtype: torch.float32
  mean: 0.498257
```

### 4. Run Benchmark

Runs the full DataLoader benchmark across multiple worker configurations.

```
python cli.py benchmark \
--data-dir ./synthetic_data \
--workers 1,2,4,8
```

#### Example results:
```
Total samples: 262135
Total batches: 4096
Beginning benchmark over 262135 samples

Worker 1: batch 0/4096
Worker 1: batch 1000/4096
Worker 1: batch 2000/4096
Worker 1: batch 3000/4096
Worker 1: batch 4000/4096
Finished: 1 workers | Time: 43.29s | Throughput: 6055.64 samples/s | Efficiency: 100.00%
Worker 2: batch 0/4096
Worker 2: batch 1000/4096
Worker 2: batch 2000/4096
Worker 2: batch 3000/4096
Worker 2: batch 4000/4096
Finished: 2 workers | Time: 30.89s | Throughput: 8486.68 samples/s | Efficiency: 70.07%
Worker 4: batch 0/4096
Worker 4: batch 1000/4096
Worker 4: batch 2000/4096
Worker 4: batch 3000/4096
Worker 4: batch 4000/4096
Finished: 4 workers | Time: 25.19s | Throughput: 10404.28 samples/s | Efficiency: 42.95%
Worker 8: batch 0/4096
Worker 8: batch 1000/4096
Worker 8: batch 2000/4096
Worker 8: batch 3000/4096
Worker 8: batch 4000/4096
Finished: 8 workers | Time: 29.23s | Throughput: 8966.66 samples/s | Efficiency: 18.51%

Final results
------------------------------------------------------------
 workers  time_sec  throughput_samples_per_sec  efficiency_percent  loaded_samples
       1     43.29                     6055.64              100.00          262135
       2     30.89                     8486.68               70.07          262135
       4     25.19                    10404.28               42.95          262135
       8     29.23                     8966.66               18.51          262135
------------------------------------------------------------
```

#### Export Benchmark results
Results can be exported to CSV:
```
python cli.py benchmark \
--data-dir ./synthetic_data \
--workers 1,2,4,8 \
--output-csv results.csv
```

## Benchmark metrics

The benchmark reports:

Efficiency is computed as:
```
efficiency = T1 / (N * TN)
```

Where:
- T1 = runtime with one worker
- TN = runtime with N workers

This benchmark focuses exclusively on the data loading pipeline. It does not include:
- model computation
- GPU transfers
- preprocessing pipelines
- dataset decompression

## Research Context

This tool is intended as a baseline experimental framework for studying:
 - ML dataset storage layouts
 - DataLoader worker scaling
 - I/O bottlenecks in Python-based pipelines
 - synthetic dataset benchmarking

The current baseline implementation uses naive file I/O:
```
open -> seek -> read -> close
```
for each sample.

Future experiments may explore:
- memory-mapped datasets
- prefetching strategies
- batched reads
- async I/O
- shard caching

## Design Assumptions and Dataset Layout
This benchmark intentionally uses a simple and controlled dataset design in order to isolate the I/O behavior of the PyTorch DataLoader.

The goal is not to emulate a specific ML dataset, but rather to provide a synthetic and reproducible environment for studying data loading performance.

The main design choices are described below.

### Binary Shard Format
The dataset is composed of pure binary shard files (.bin).

Each file contains a continuous sequence of float32 values, written without any additional metadata or container format.

This choice was made to: 
- avoid overhead from parsing structured formats
- isolate raw disk I/O performance
- simplify offset-based random access
- minimize CPU preprocessing costs

Each sample is interpreted as a fixed-length vector of float32 values.

Current implementation:
```
sample_size_floats = 1024
sample_size_bytes = 4096
```

Future extensions may support float16, float64, mixed feature layouts or structured binary formats. Ideally the type of the values represented should be included as an arguent to the CLI.

### Variable Shard Sizes
Shard files are not fixed size.

Instead, the generator randomly samples file sizes from a user-defined interval:
```
[min_size_mb, max_size_mb]
```
For example:
```
min_size_mb = 50
max_size_mb = 100
```

Results in shard sizes such as:
```
data_shard_0000.bin -> 97 MB
data_shard_0001.bin -> 88 MB
...
```

Using variable shard sizes allows the benchmark to better approximate real-world datasets where files rarely have identical sizes.

### Chunk-Based Dataset Generation
Large binary shards are generated using chunked writes instead of allocating full arrays in memory.

Data generation proceeds as:
```
generate random chunk -> write to disk -> repeat
```

This design avoids:
- excessive RAM usage
- large temporary allocations
- memory fragmentation

Chunked generation was implemented to make it possible to create large datasets even on machines with limited memory.

### Sample Access Model
Each sample is retrieved using direct file offset access.

The benchmark therefore measures the cost of:
- frequent file open/close operations
- random disk seeks
- small read operations
- multiprocessing overhead in DataLoader workers

This also suggest the potential bottleneck of this approach.

### Reproducibility
Dataset generation is deterministic given a fixed seed:
```
seed = 70525386
```
This ensures that shard layout and data distribution remain identical across runs, allowing fair comparison between benchmark configurations.
