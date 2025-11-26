import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


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

    html_output = output_dir / f"{script_name}_mem.html"
    cmd_flame = [sys.executable, "-m", "memray", "flamegraph", "-f", "-o", str(html_output), str(mem_output)]
    subprocess.run(cmd_flame, env=env, check=True)
    print(f"    Flamegraph generated: {html_output}")


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = Path(f"output/benchmark_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_dir = Path("benchmark")

    scripts = sorted(list(benchmark_dir.glob("*.py")))

    print(f"Found {len(scripts)} benchmarks to run.")

    for script in scripts:
        if script.name == "__init__.py":
            continue

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
