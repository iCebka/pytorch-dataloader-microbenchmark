# First results

## Experimental Setup
Dataset configuration

|Parameter | Value |
|-|-|
|Dataset size | 5, 6, 7, 8, 9, 10 GB |
|Shard size range | 100 - 1024 MB |
| Sample size | 1024 float32 values |
| Sample size (bytes) | 4096 B|
| Batch size | 64 |

Dataset generation command:
```
python cli.py generate \
--output-dir ./synthetic_data \
--total-size-mb X \
--mnin-size-mb 100 
--max-size-mb 1024
```
Benchmark command:
```
python cli.py benchmark \
--data-dir ./synthetic_data \
--workers 1,2,4,8 \
--output-csv results_Xgb.csv
```
X = 5, 6, 7, 8, 9, 10 GB.

## Hardware Environment

| Component | Description |
| - | - |
|CPU | AMD Ryzen 7 3700U with Radeon Vega Mobile Gfx |
|RAM | 16 GB |
|OS | Windows 11 |
|Python | 3.14.3 |
|PyTorch| 2.10.0 |

The benchmark was executed on a local workstation using a standard filesystem. Results are stored in the .csv files of this folder.

## Remarks
The results show that increasing the number of DataLoader workers initially improves performance, but the benefits quickly diminish.

Moving from one to four workers significantly reduces runtime and increases throughput, indicating that part of the data loading workload benefits from parallel execution.

However, scaling beyond four workers does not produce further improvements. In this experiment, using eight workers actually increases runtime compared to four workers.

This behavior suggests that the data loading pipeline benefits from moderate parallelism but does not scale linearly with the number of workers. The optimal configuration for this experiment appears to be around four workers.

Possible explanations include:

filesystem contention caused by concurrent small reads

overhead associated with worker process coordination

frequent file open/close operations in the current dataset access model

These results provide a baseline for evaluating alternative loading strategies in future experiments.