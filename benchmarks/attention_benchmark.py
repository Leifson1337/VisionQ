from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visionq.attention import (  # noqa: E402
    ChunkedStreamingAttention,
    FlashAttention,
    SparseAttention,
    SpatialNeighborhoodAttention,
    SpatioTemporalHybridAttention,
)
from visionq.core import AttentionContext  # noqa: E402
from visionq.kernels.triton import TritonAttentionKernel, triton_available  # noqa: E402


def environment_metadata(device: torch.device) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "requested_device": device.type,
        "triton_available": triton_available(),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        free, total = torch.cuda.mem_get_info(idx)
        metadata.update(
            {
                "cuda_device_name": props.name,
                "cuda_capability": f"{props.major}.{props.minor}",
                "cuda_total_gb": total / (1024**3),
                "cuda_free_gb": free / (1024**3),
            }
        )
    return metadata


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(
    fn: Callable[[], torch.Tensor], repeats: int, warmup: int, device: torch.device
) -> tuple[dict[str, float], int | None]:
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        samples: list[float] = []
        for _ in range(repeats):
            start = time.perf_counter()
            fn()
            synchronize(device)
            samples.append((time.perf_counter() - start) * 1000.0)
    samples_sorted = sorted(samples)
    p95_idx = min(len(samples_sorted) - 1, int(round(0.95 * (len(samples_sorted) - 1))))
    p99_idx = min(len(samples_sorted) - 1, int(round(0.99 * (len(samples_sorted) - 1))))
    stats = {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "min_ms": min(samples),
        "p95_ms": samples_sorted[p95_idx],
        "p99_ms": samples_sorted[p99_idx],
    }
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return stats, peak_memory


def sequence_case(seq_len: int, heads: int, head_dim: int, device: torch.device) -> dict[str, Any]:
    q = torch.randn(1, heads, seq_len, head_dim, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    ctx = AttentionContext(modality="sequence", sequence_length=seq_len, device=device)
    model_dim = heads * head_dim
    backends: dict[str, Callable[[], torch.Tensor]] = {
        "dense_sdpa": lambda q=q, k=k, v=v: F.scaled_dot_product_attention(q, k, v),
        "flash": lambda q=q, k=k, v=v, ctx=ctx: FlashAttention(model_dim, heads)(q, k, v, ctx),
        "sparse": lambda q=q, k=k, v=v, ctx=ctx: SparseAttention(model_dim, heads)(q, k, v, ctx),
        "chunked_streaming": lambda q=q, k=k, v=v, ctx=ctx: ChunkedStreamingAttention(
            model_dim, heads
        )(q, k, v, ctx, block_size=64),
    }
    if device.type == "cuda" and triton_available():
        backends["triton_dense"] = lambda q=q, k=k, v=v, ctx=ctx: TritonAttentionKernel().forward(
            q, k, v, ctx
        )
    return {"shape": f"seq={seq_len}", "tokens": seq_len, "backends": backends}


def image_case(
    height: int, width: int, heads: int, head_dim: int, device: torch.device
) -> dict[str, Any]:
    tokens = height * width
    q = torch.randn(1, heads, tokens, head_dim, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    ctx = AttentionContext(
        modality="image",
        sequence_length=tokens,
        spatial_shape=(height, width),
        spatial_window=(3, 3),
        device=device,
    )
    model_dim = heads * head_dim
    return {
        "shape": f"image={height}x{width}",
        "tokens": tokens,
        "backends": {
            "dense_sdpa": lambda q=q, k=k, v=v: F.scaled_dot_product_attention(q, k, v),
            "spatial_neighborhood": lambda q=q, k=k, v=v, ctx=ctx: SpatialNeighborhoodAttention(
                model_dim, heads
            )(q, k, v, ctx),
        },
    }


def video_case(
    frames: int, height: int, width: int, heads: int, head_dim: int, device: torch.device
) -> dict[str, Any]:
    tokens = frames * height * width
    q = torch.randn(1, heads, tokens, head_dim, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    ctx = AttentionContext(
        modality="video",
        sequence_length=tokens,
        spatial_shape=(height, width),
        temporal_dim=frames,
        spatial_window=(3, 3),
        temporal_window=3,
        device=device,
    )
    model_dim = heads * head_dim
    return {
        "shape": f"video={frames}x{height}x{width}",
        "tokens": tokens,
        "backends": {
            "dense_sdpa": lambda q=q, k=k, v=v: F.scaled_dot_product_attention(q, k, v),
            "spatiotemporal_hybrid": lambda q=q, k=k, v=v, ctx=ctx: SpatioTemporalHybridAttention(
                model_dim, heads
            )(q, k, v, ctx),
        },
    }


def write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, Any]], metadata: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# VisionQ Benchmark Report",
        "",
        "## Environment",
        "",
        "```json",
        json.dumps(metadata, indent=2, sort_keys=True),
        "```",
        "",
        "## Results",
        "",
        "| Backend | Device | Shape | Tokens | Heads | Head Dim | Median ms | Mean ms | p95 ms | p99 ms | Peak Memory Bytes |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {backend} | {device} | {shape} | {tokens} | {heads} | {head_dim} | "
            "{median_ms} | {mean_ms} | {p95_ms} | {p99_ms} | {peak_memory_bytes} |".format(
                **row
            )
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[128, 512, 1024])
    parser.add_argument("--image-shapes", nargs="+", default=["16x16"])
    parser.add_argument("--video-shapes", nargs="+", default=["2x8x8"])
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")
    device = torch.device(args.device)
    metadata = environment_metadata(device)
    print(json.dumps({"environment": metadata}, sort_keys=True))
    if args.metadata_output:
        args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_output.write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )

    cases: list[dict[str, Any]] = [
        sequence_case(seq_len, args.heads, args.head_dim, device) for seq_len in args.seq_lens
    ]
    for shape in args.image_shapes:
        height, width = (int(part) for part in shape.lower().split("x"))
        cases.append(image_case(height, width, args.heads, args.head_dim, device))
    for shape in args.video_shapes:
        frames, height, width = (int(part) for part in shape.lower().split("x"))
        cases.append(video_case(frames, height, width, args.heads, args.head_dim, device))

    rows: list[dict[str, Any]] = []
    for case in cases:
        for backend, fn in case["backends"].items():
            stats, peak_memory = measure(fn, args.repeats, args.warmup, device)
            row = {
                "backend": backend,
                "device": device.type,
                "shape": case["shape"],
                "tokens": case["tokens"],
                "heads": args.heads,
                "head_dim": args.head_dim,
                "median_ms": f"{stats['median_ms']:.4f}",
                "mean_ms": f"{stats['mean_ms']:.4f}",
                "min_ms": f"{stats['min_ms']:.4f}",
                "p95_ms": f"{stats['p95_ms']:.4f}",
                "p99_ms": f"{stats['p99_ms']:.4f}",
                "peak_memory_bytes": peak_memory if peak_memory is not None else "",
            }
            rows.append(row)
            print(",".join(str(row[key]) for key in row))
    if args.output and rows:
        write_csv(rows, args.output)
    if args.markdown_output and rows:
        write_markdown(rows, metadata, args.markdown_output)


if __name__ == "__main__":
    main()
