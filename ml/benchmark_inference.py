"""Benchmark TorchScript inference for one user and a 30-user burst."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

try:
    from ml.train_tcn_corn import make_temporal_inputs
except ModuleNotFoundError:
    from train_tcn_corn import make_temporal_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("ml/artifacts/data"))
    parser.add_argument("--model-dir", type=Path, default=Path("ml/artifacts/model"))
    parser.add_argument("--model-file", default="model_tcn_corn.ts")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=50)
    return parser.parse_args()


def measure(model: torch.jit.ScriptModule, inputs: torch.Tensor, repetitions: int) -> np.ndarray:
    durations = []
    with torch.inference_mode():
        for _ in range(10):
            model(inputs)
        for _ in range(repetitions):
            started = time.perf_counter()
            model(inputs)
            durations.append((time.perf_counter() - started) * 1_000)
    return np.asarray(durations)


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    model = torch.jit.load(str(args.model_dir / args.model_file), map_location="cpu").eval()
    with np.load(args.data_dir / "test.npz", allow_pickle=False) as data:
        raw = data["X"][:30]
    with np.load(args.model_dir / "normalization.npz") as normalization:
        inputs = (make_temporal_inputs(raw) - normalization["mean"]) / normalization["std"]
    tensor = torch.from_numpy(inputs)
    single = measure(model, tensor[:1], args.repetitions)
    burst = measure(model, tensor, args.repetitions)
    print(f"single median={np.median(single):.3f} ms p95={np.percentile(single, 95):.3f} ms")
    print(f"batch30 median={np.median(burst):.3f} ms p95={np.percentile(burst, 95):.3f} ms")
    print(f"batch30 throughput={30_000 / np.median(burst):.1f} sequences/s")


if __name__ == "__main__":
    main()
