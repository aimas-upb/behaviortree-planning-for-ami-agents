"""Shared helpers for scripts that need repository-relative paths."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = PROJECT_ROOT / "docker" / "Dockerfile"


def repo_path(*parts: str) -> Path:
    """Build a path relative to the repository root."""
    return PROJECT_ROOT.joinpath(*parts)


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve a path relative to the repository root when needed."""
    resolved = Path(path)
    return resolved if resolved.is_absolute() else PROJECT_ROOT / resolved
