# Contributing

Use a local editable install:

```bash
pip install -e ".[dev]"
```

Before opening a change, run:

```bash
python -m compileall visionq tests
python -m pytest
ruff check .
ruff format --check .
mypy visionq
```

You can also run the same groups through Nox:

```bash
nox
```

Install local hooks with:

```bash
pre-commit install
```

Backends must be registered with a key from `AttentionBackendName`, validate
input shapes, preserve dtype/device, include CPU tests, and document unsupported
masking or hardware requirements.
