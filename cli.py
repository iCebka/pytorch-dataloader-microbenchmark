import argparse
from generator import GeneratorConfig, generate_dataset
from dataset import BinaryShardedDataset, DatasetConfig, create_dataloader
from benchmark import BenchmarkConfig, BenchmarkRunner

# Helpers

# Convert comma-separated worker list into integers: 1,2,3,4 -> [1,2,3,4]
def parse_workers(raw: str):
    return [int(x.strip()) for x in raw.split(",") if x.strip()]
#
# CLI commands
#

# Command: generate
# Creates a synthethic binary dataset using the dataset generator
def cmd_generate(args):
    config = GeneratorConfig(
        output_dir=args.output_dir,
        total_size_mb=args.total_size_mb,
        min_size_mb=args.min_size_mb,
        max_size_mb=args.max_size_mb,
        seed=args.seed,
        chunk_size_mb=args.chunk_size_mb,
        dtype=args.dtype,
        reset_output_dir=not args.no_reset
    )
    generate_dataset(config)

# Command: inspect
# Loads the dataset and prints a summary
# Optionally prints file list and previews
def cmd_inspect(args):
    n = 60
    dataset=BinaryShardedDataset(
        DatasetConfig(
            data_dir=args.data_dir,
            sample_size_floats=args.sample_size_floats,
        )
    )

    info = dataset.describe()

    print("Dataset inspection")
    print("-"*n)
    print(f"Directory: {info['data_dir']}")
    print(f"Sample size (floats): {info['sample_size_floats']}")
    print(f"Sample size (bytes): {info['sample_size_bytes']}")
    print(f"Mapped files: {info['num_files']}")
    print(f"Total samples: {info['total_samples']}")
    print("-"*n)

    # optionally show all mapped shard files
    if args.show_files: 
        for i, path in enumerate(info["files"]):
            print(f"{i:4d}: {path}")
    
    # optional preview of a specific sample
    if args.preview_index is not None:
        sample = dataset[args.preview_index]
        preview_count = min(args.preview_count, sample.shape[0])

        print("-"*n)
        print(f"Preview sample index: {args.preview_index}")
        print(f"Tensor shape: {tuple(sample.shape)}")
        print(f"Tensor dtype: {sample.dtype}")
        print(f"First {preview_count} values:")
        print(sample[:preview_count])

# Command: smoke
# Performs a minimal DataLoader test to verify the pipeline works
# Useful before running large benchmarks
def cmd_smoke(args):
    n=60
    loader = create_dataloader(
        data_dir = args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_size_floats=args.sample_size_floats,
        shuffle=False,
        #pin_memory=args.pin_memory,
    )

    print("Running smoke test")
    print("-"*n)
    print(loader)

    # Iterate over a few batches only
    for i, batch in enumerate(loader):
        if i >= args.num_batches:
            break

        print(f"Batch {i}")
        print(f"  shape: {tuple(batch.shape)}")
        print(f"  dtype: {batch.dtype}")
        print(f"  mean: {batch.mean().item():.6f}")

# Command: benchmark
# Runs the full DataLoader performance using BenchmarkRunner
def cmd_benchmark(args):
    n=60
    config = BenchmarkConfig(
        data_dir=args.data_dir,
        workers_list=parse_workers(args.workers),
        batch_size=args.batch_size,
        sample_size_floats=args.sample_size_floats,
        warmup_batches=args.warmup_batches,
        progress_every=args.progress_every,
        #pin_memory=args.pin_memory,
    )

    runner = BenchmarkRunner(config)
    df = runner.run()

    print("\nFinal results")
    print("-"*n)
    print(df.to_string(index=False))
    print("-"*n)

    # Optionally export results to CSV
    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        print(f"Saved CSV results to: {args.output_csv}")

#
# CLI parser
#
# Build the top-level CLI parser 
def build_parser():

    parser = argparse.ArgumentParser(
        description="PyTorch DataLoader microbenchmark CLI"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # 
    # generate command
    # 
    
    p_gen = subparsers.add_parser(
        "generate",
        help="Generate synthetic binary dataset"
    )

    p_gen.add_argument("--output-dir", required=True)
    p_gen.add_argument("--total-size-mb", type=float, required=True)
    p_gen.add_argument("--min-size-mb", type=float, required=True)
    p_gen.add_argument("--max-size-mb", type=float, required=True)

    p_gen.add_argument("--seed", type=int, default=70525386)
    p_gen.add_argument("--chunk-size-mb", type=float, default=64.0)
    p_gen.add_argument("--dtype", default="float32")

    # Prevent deleting existing dataset directory if needed
    p_gen.add_argument("--no-reset", action="store_true")

    p_gen.set_defaults(func=cmd_generate)

    # 
    # inspect command
    #

    p_inspect = subparsers.add_parser(
        "inspect",
        help="Inspect dataset and preview a sample"
    )

    p_inspect.add_argument("--data-dir", required=True)
    p_inspect.add_argument("--sample-size-floats", type=int, default=1024)

    p_inspect.add_argument("--show-files", action="store_true")

    p_inspect.add_argument("--preview-index", type=int)
    p_inspect.add_argument("--preview-count", type=int, default=8)

    p_inspect.set_defaults(func=cmd_inspect)

    # 
    # smoke command
    # 

    p_smoke = subparsers.add_parser(
        "smoke",
        help="Small DataLoader smoke test"
    )

    p_smoke.add_argument("--data-dir", required=True)
    p_smoke.add_argument("--sample-size-floats", type=int, default=1024)

    p_smoke.add_argument("--batch-size", type=int, default=64)
    p_smoke.add_argument("--num-workers", type=int, default=1)

    p_smoke.add_argument("--num-batches", type=int, default=3)

    #p_smoke.add_argument("--pin-memory", action="store_true")

    p_smoke.set_defaults(func=cmd_smoke)

    # 
    # benchmark command
    # 

    p_bench = subparsers.add_parser(
        "benchmark",
        help="Run full benchmark"
    )

    p_bench.add_argument("--data-dir", required=True)

    p_bench.add_argument("--sample-size-floats", type=int, default=1024)
    p_bench.add_argument("--batch-size", type=int, default=64)

    # Worker configurations (comma separated)
    p_bench.add_argument("--workers", default="1,2,4,8")

    p_bench.add_argument("--warmup-batches", type=int, default=5)
    p_bench.add_argument("--progress-every", type=int, default=1000)

    #p_bench.add_argument("--pin-memory", action="store_true")

    # Optional output file
    p_bench.add_argument("--output-csv")

    p_bench.set_defaults(func=cmd_benchmark)

    return parser

#
# CLI main
#
def main():
    parser = build_parser()
    args = parser.parse_args()

    # Execute command
    args.func(args)

if __name__ == "__main__":
    main()