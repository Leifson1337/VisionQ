# Release

Releases are tag-driven.

1. Ensure the working tree is clean.
2. Run `python -m compileall visionq tests`.
3. Run `python -m pytest`.
4. Run `ruff check .` and `ruff format --check .`.
5. Run `mypy visionq`.
6. Run `python -m build` and `twine check dist/*`.
7. Optionally run the TestPyPI workflow from GitHub Actions.
8. Create and push an annotated tag such as `v0.1.0`.

The GitHub Actions release workflow builds the package and publishes to PyPI on
`v*` tags. Publishing requires PyPI trusted publishing or the repository's PyPI
credentials to be configured outside the codebase.

Release artifacts should include the built wheel/sdist, the SBOM artifact from
the SBOM workflow, and benchmark artifacts when a release candidate was measured
on project-supported hardware.
