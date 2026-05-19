from pathlib import Path

from benchmarks.attention_benchmark import write_markdown


def test_markdown_benchmark_report(tmp_path: Path):
    output = tmp_path / "report.md"
    rows = [
        {
            "backend": "flash",
            "device": "cpu",
            "shape": "seq=8",
            "tokens": 8,
            "heads": 2,
            "head_dim": 4,
            "median_ms": "0.1",
            "mean_ms": "0.1",
            "p95_ms": "0.1",
            "p99_ms": "0.1",
            "peak_memory_bytes": "",
        }
    ]
    write_markdown(rows, {"torch": "test"}, output)
    text = output.read_text(encoding="utf-8")
    assert "VisionQ Benchmark Report" in text
    assert "flash" in text
    assert "p95 ms" in text
