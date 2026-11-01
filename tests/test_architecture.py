import ast
from pathlib import Path

import pytest

import evidone_pose_detection

PACKAGE = Path(evidone_pose_detection.__file__).parent
WRAPPER = "pose.py"
ISOLATED_ROOTS = {"torch", "ultralytics", "torchvision"}
OTHER_MODULES = sorted(path for path in PACKAGE.glob("*.py") if path.name != WRAPPER)
ALL_MODULES = sorted(PACKAGE.glob("*.py"))
SHARED_CONSTANT = "VISIBILITY_THRESHOLD"


@pytest.mark.parametrize("module", OTHER_MODULES, ids=lambda path: path.name)
def test_only_the_pose_wrapper_reaches_the_ml_stack(module: Path) -> None:
    assert _imported_roots(module) & ISOLATED_ROOTS == set()


def test_the_pose_wrapper_owns_the_ultralytics_dependency() -> None:
    assert "ultralytics" in _imported_roots(PACKAGE / WRAPPER)


def test_the_visibility_threshold_is_defined_exactly_once() -> None:
    definers = [module.name for module in ALL_MODULES if _assigns(module, SHARED_CONSTANT)]

    assert definers == ["models.py"]


def _assigns(module: Path, name: str) -> bool:
    for node in ast.walk(ast.parse(module.read_text())):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return True
    return False


def _imported_roots(module: Path) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(module.read_text())):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            roots.add(node.module.split(".")[0])
    return roots
