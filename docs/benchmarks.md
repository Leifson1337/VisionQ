# Benchmarks

Run CPU benchmarks with:

```bash
python benchmarks/attention_benchmark.py --device cpu --seq-lens 128 512 1024
python benchmarks/attention_benchmark.py --device cpu --output results/cpu.csv --metadata-output results/cpu.json
```

GPU runs can be requested with `--device cuda` when PyTorch detects CUDA. Results
depend on hardware, PyTorch version, dtype and system load, so the project does
not ship fixed performance claims.

The benchmark prints environment metadata as JSON before the CSV-like measurement
rows. It covers sequence, image and video shapes. When CUDA and Triton are both
available, the optional `triton_dense` measurement is included automatically.
