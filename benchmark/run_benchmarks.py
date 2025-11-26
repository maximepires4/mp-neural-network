import argparse
import pstats
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from pstats import SortKey


def run_benchmark(script_path: Path, output_dir: Path):
    script_name = script_path.stem

    mem_output = output_dir / f"{script_name}.bin"
    prof_output = output_dir / f"{script_name}.prof"

    print(f"\n=== Running Benchmark: {script_name} ===")
    print(f"Script: {script_path}")

    print("--> Running cProfile...")
    start_time = time.time()
    cmd_prof = [sys.executable, "-m", "cProfile", "-o", str(prof_output), str(script_path)]

    env = {"PYTHONPATH": "."}

    subprocess.run(cmd_prof, env=env, check=True)
    duration = time.time() - start_time
    print(f"    Done in {duration:.2f}s. Output: {prof_output}")

    print("--> Running Memray...")
    cmd_mem = [sys.executable, "-m", "memray", "run", "-f", "-o", str(mem_output), str(script_path)]

    subprocess.run(cmd_mem, env=env, check=True)
    print(f"    Done. Output: {mem_output}")

    print("--> Generating Memray Stats...")
    stats_output = output_dir / f"{script_name}_mem_stats.txt"
    with open(stats_output, "w") as f:
        cmd_stats = [sys.executable, "-m", "memray", "stats", str(mem_output)]
        subprocess.run(cmd_stats, env=env, check=True, stdout=f)
    print(f"    Stats saved to: {stats_output}")

    html_output = output_dir / f"{script_name}_mem.html"
    cmd_flame = [sys.executable, "-m", "memray", "flamegraph", "-f", "-o", str(html_output), str(mem_output)]
    subprocess.run(cmd_flame, env=env, check=True)
    print(f"    Flamegraph generated: {html_output}")


def compare_benchmarks(before_dir: Path, after_dir: Path):
    print(f"\n=== Comparing Benchmarks: {before_dir.name} vs {after_dir.name} ===")

    before_files = {f.stem: f for f in before_dir.glob("*.prof")}
    after_files = {f.stem: f for f in after_dir.glob("*.prof")}

    common_scripts = set(before_files.keys()) & set(after_files.keys())

    for script_name in common_scripts:
        print(f"\n--> Comparison for: {script_name}")
        before_prof = before_files[script_name]
        after_prof = after_files[script_name]

        stats_before = pstats.Stats(str(before_prof))
        stats_after = pstats.Stats(str(after_prof))

        time_before = stats_before.total_tt  # type: ignore[attr-defined]
        time_after = stats_after.total_tt  # type: ignore[attr-defined]
        diff = time_after - time_before
        percent = (diff / time_before) * 100 if time_before > 0 else 0.0

        print(f"    Total Time: {time_before:.4f}s -> {time_after:.4f}s")
        print(f"    Change: {diff:+.4f}s ({percent:+.2f}%)")

        if diff < 0:
            print("    ✅ IMPROVEMENT")
        else:
            print("    ❌ REGRESSION (or noise)")

        print("    Top 5 Functions (Before):")
        stats_before.sort_stats(SortKey.CUMULATIVE).print_stats(10)
        print("    Top 5 Functions (After):")
        stats_after.sort_stats(SortKey.CUMULATIVE).print_stats(10)


def main():
    parser = argparse.ArgumentParser(description="Run benchmarks or compare two benchmark runs.")
    parser.add_argument("--before", type=Path, help="Path to the 'before' benchmark output directory")
    parser.add_argument("--after", type=Path, help="Path to the 'after' benchmark output directory")
    args = parser.parse_args()

    if args.before and args.after:
        compare_benchmarks(args.before, args.after)
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"output/benchmark_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_dir = Path("benchmark")
    scripts = sorted(list(benchmark_dir.glob("*.py")))

    bench_scripts = [s for s in scripts if s.name != "__init__.py" and "analyze" not in s.name and "run_benchmarks" not in s.name]

    print(f"Found {len(bench_scripts)} benchmarks to run.")

    for script in bench_scripts:
        try:
            run_benchmark(script, output_dir)
        except subprocess.CalledProcessError as e:
            print(f"!!! Error running {script}: {e}")
        except Exception as e:
            print(f"!!! Unexpected error running {script}: {e}")

    print("\n=== All Benchmarks Completed ===")
    print(f"Results available in: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
