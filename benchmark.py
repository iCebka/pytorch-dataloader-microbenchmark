import time
import pandas as pd
from torch.utils.data import DataLoader
from dataclasses import dataclass
from math import ceil
from typing import List, Dict, Any, Optional
from dataset import BinaryShardedDataset, DatasetConfig

# Holds all configuration parameters for the benchmark experiment
@dataclass
class BenchmarkConfig:
    data_dir: str
    workers_list: List[int]
    batch_size: int=64
    sample_size_floats: int=1024
    warmup_batches: int=5 # To avoid noise for initializing workers
    progress_every: int = 1000
    #pin_memory: bool = False

# Executes the DataLoader benchmark across different worker configurations
class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig):
        if not config.workers_list:
            raise ValueError("workers_list must not be empty")
        if any(w <= 0 for w in config.workers_list):
            raise ValueError("All worker counts must be positive")
        if config.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.config = config

        # Instantiate dataset once so initialization time is not included in the measurements
        self.dataset = BinaryShardedDataset(
            DatasetConfig(
                data_dir=config.data_dir,
                sample_size_floats=config.sample_size_floats,
            )
        )
    
    def run(self):
        results: List[Dict[str, Any]] = []
        t1_time: Optional[float] = None

        total_samples = len(self.dataset)
        total_batches = ceil(total_samples / self.config.batch_size)

        print(f"Total samples: {total_samples}")
        print(f"Total batches: {total_batches}")
        print(f"Beginning benchmark over {total_samples} samples \n")

        # Iterate over different worker configurations
        for n in self.config.workers_list:

            loader = DataLoader(
                self.dataset,
                batch_size=self.config.batch_size,
                num_workers=n,
                shuffle=False,
                #pin_memory=self.config.pin_memory
            )

            # Warm up phase to initialize worker processes and stabilize runtine behavior
            warmup_steps = min(self.config.warmup_batches, len(loader))
            if warmup_steps > 0:
                it = iter(loader)
                for _ in range(warmup_steps):
                    next(it)
            
            # Now iterate over the entire dataset
            start_time = time.perf_counter() # time.time()
            loaded_samples = 0

            for i, batch in enumerate(loader):
                loaded_samples += batch.size(0)

                if self.config.progress_every > 0 and i % self.config.progress_every == 0:
                    print(f"Worker {n}: batch {i}/{len(loader)}")
            
            end_time = time.perf_counter()
            duration = end_time - start_time

            # Throughput in samples per second
            throughput = loaded_samples / duration

            # Efficiency relative to t1
            if n == 1:
                t1_time = duration
                efficiency = 100.0
            else:
                if t1_time is None:
                    raise RuntimeError(
                        "Benchmark must include num_workers=1 to compute efficiency"
                    )
                efficiency = (t1_time / (n * duration)) * 100.0
            
            results.append(
                {
                    "workers": n,
                    "time_sec": round(duration, 2),
                    "throughput_samples_per_sec": round(throughput, 2),
                    "efficiency_percent": round(efficiency, 2),
                    "loaded_samples": loaded_samples,
                }
            )

            print(
                f"Finished: {n} workers | "
                f"Time: {duration:.2f}s | "
                f"Throughput: {throughput:.2f} samples/s | "
                f"Efficiency: {efficiency:.2f}%"
            )
        return pd.DataFrame(results)
    

if __name__ == "__main__":

    n = 72

    print("=" * n)
    print("BenchmarkRunner Test")
    print("=" * n)

    config = BenchmarkConfig(
        data_dir="./synthetic_data",
        workers_list=[1, 2, 4, 8],
        batch_size=64,
        sample_size_floats=1024,
        warmup_batches=5,
        progress_every=1000,
        #pin_memory=False,
    )

    runner = BenchmarkRunner(config)
    df = runner.run()

    print("\n" + "=" * n)
    print("Final benchmark results")
    print("=" * n)
    print(df.to_string(index=False))
    print("=" * n)