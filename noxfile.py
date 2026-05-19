from __future__ import annotations

import nox

nox.options.sessions = ["lint", "tests", "typecheck", "docs"]


@nox.session(python=["3.10", "3.11", "3.12"])
def tests(session: nox.Session) -> None:
    session.install("-e", ".[test]")
    session.run("python", "-m", "compileall", "visionq", "tests")
    session.run("python", "-m", "pytest")


@nox.session
def lint(session: nox.Session) -> None:
    session.install("ruff")
    session.run("ruff", "check", ".")
    session.run("ruff", "format", "--check", ".")


@nox.session
def typecheck(session: nox.Session) -> None:
    session.install("-e", ".[dev]")
    session.run("mypy", "visionq")


@nox.session
def docs(session: nox.Session) -> None:
    session.install("-e", ".[docs]")
    session.run("mkdocs", "build", "--strict")


@nox.session
def build(session: nox.Session) -> None:
    session.install("build", "twine")
    session.run("python", "-m", "build")
    session.run("twine", "check", "dist/*")
