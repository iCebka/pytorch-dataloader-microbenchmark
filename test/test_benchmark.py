import time
import pandas as pd
from math import ceil
from test_loader import *

def run_benchmark(data_dir, workers_list=[1, 2, 4, 8], batch_size=64):
    results = []
    t1_time = None # Time with one worker

    # Dataset is instantiated once
    dataset = BinaryShardedDataset(data_dir=data_dir)
    total_samples = len(dataset)

    print("Total samples:", total_samples)
    print("Total batches:", ceil(total_samples / batch_size) )

    print(f"Beginning Benchmark over {total_samples}\n")

    for n in workers_list:
        # Set up the loader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=n,
            shuffle=False,
            #pin_memory=True
        )

        # Warm up for workers with 5 batches 
        it = iter(loader)
        for _ in range(min(5, len(loader))):
            next(it)
        
        start_time = time.perf_counter() # time.time()
        loaded_samples = 0

        for i, batch in enumerate(loader):
            loaded_samples += batch.size(0)
            if i % 1000 == 0:
                print(f"Worker {n}: batch {i}/{len(loader)}")

        end_time = time.perf_counter()
        duration = end_time - start_time

        # Metrics calculate
        if n == 1:
            t1_time = duration
            efficiency = 100
        throughput = loaded_samples / duration # Samples per second
        if n != 1:
            efficiency = (t1_time / (n * duration)) * 100

        results.append({
            "workers": n,
            "tiempo (s)": round(duration, 2),
            "throughput (S/s)": round(throughput, 2),
            "efficiency (%)": round(efficiency, 2)
        })

        print(
            f"Finished: {n} workers | "
            f"Time: {duration:.2f}s | "
            f"Throughput: {throughput:.2f} samples/s | "
            f"Efficiency: {efficiency:.2f}%"
        )

    return results

if __name__ == "__main__":

    n = 50

    DATA_PATH = "./synthetic_data"

    benchmark_data = run_benchmark(DATA_PATH)

    df = pd.DataFrame(benchmark_data)
    print("\n" + "="*n)
    print("Final results")
    print("="*n)
    print(df.to_string(index=False))
    print("="*n)