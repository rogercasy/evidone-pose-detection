import ast
from pathlib import Path

import pytest

import evidone_pose_detection

PACKAGE = Path(evidone_pose_detection.__file__).parent
WRAPPER = "pose.py"
ISOLATED_ROOTS = {"torch", "ultralytics", "torchvision"}
OTHER_MODULES = sorted(path for path in PACKAGE.glob("*.py") if path.name != WRAPPER)


@pytest.mark.parametrize("module", OTHER_MODULES, ids=lambda path: path.name)
def test_only_the_pose_wrapper_reaches_the_ml_stack(module: Path) -> None:
    assert _imported_roots(module) & ISOLATED_ROOTS == set()


def test_the_pose_wrapper_owns_the_ultralytics_dependency() -> None:
    assert "ultralytics" in _imported_roots(PACKAGE / WRAPPER)


def _imported_roots(module: Path) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(module.read_text())):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            roots.add(node.module.split(".")[0])
    return roots
