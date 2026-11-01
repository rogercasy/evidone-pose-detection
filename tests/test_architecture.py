import ast
import subprocess
import sys
from pathlib import Path

import pytest

import evidone_pose_detection

PACKAGE = Path(evidone_pose_detection.__file__).parent
WRAPPER = PACKAGE / "vision" / "pose.py"
ISOLATED_ROOTS = {"torch", "torchvision", "ultralytics"}
# Values that moved into config.toml; no module may declare them again.
CONFIGURED_NAMES = frozenset(
    {
        "VISIBILITY_THRESHOLD",
        "LANDMARK_COVERAGE_MARGIN",
        "FACE_SPANS",
        "SHOULDER_SPAN",
        "MATCH_DISTANCE_RATIO",
        "SMOOTHING",
        "DEFAULT_HOLD_FRAMES",
        "HOLD_GROWTH_PER_FRAME",
        "BLUR_SAMPLES",
        "PIXEL_BLOCKS",
        "FILL_COLOUR",
        "REGION_COLOURS",
        "JOINT_RADIUS",
        "BONE_THICKNESS",
        "PANEL_ORIGIN",
        "PANEL_PADDING",
        "PANEL_OPACITY",
        "PANEL_COLOUR",
        "TEXT_COLOUR",
        "FONT_SCALE",
        "FONT_THICKNESS",
        "LINE_HEIGHT",
        "WINDOW_TITLE",
        "VIEW_BACKGROUND",
        "MINIMUM_YIELD_MS",
        "FRAME_RATE_WINDOW",
        "FALLBACK_FPS",
        "FOURCC",
        "OUTPUT_STEM",
        "DEFAULT_CAMERA_INDEX",
        "DEFAULT_CONFIDENCE",
        "DEFAULT_WEIGHTS",
        "DEFAULT_HIDE_STYLE",
        "DEFAULT_DRAW_SKELETON",
        "STYLE_LABELS",
    }
)

ALL_MODULES = sorted(PACKAGE.rglob("*.py"))
OTHER_MODULES = [module for module in ALL_MODULES if module != WRAPPER]


def module_id(module: Path) -> str:
    return str(module.relative_to(PACKAGE))


def test_the_scan_actually_reaches_the_sub_packages() -> None:
    scanned = {module_id(module) for module in ALL_MODULES}

    assert WRAPPER in ALL_MODULES
    assert "vision/rendering.py" in scanned
    assert "ui/window.py" in scanned


@pytest.mark.parametrize("module", OTHER_MODULES, ids=module_id)
def test_only_the_pose_wrapper_reaches_the_ml_stack(module: Path) -> None:
    assert _imported_roots(module) & ISOLATED_ROOTS == set()


def test_the_pose_wrapper_owns_the_ultralytics_dependency() -> None:
    assert "ultralytics" in _imported_roots(WRAPPER)


def test_importing_the_ui_does_not_load_the_ml_stack() -> None:
    assert _ml_loaded_after("import evidone_pose_detection.ui") == "False"


def test_importing_the_vision_package_does_not_load_the_ml_stack() -> None:
    assert _ml_loaded_after("import evidone_pose_detection.vision") == "False"


def test_the_pose_module_is_still_the_way_in_to_the_ml_stack() -> None:
    assert _ml_loaded_after("import evidone_pose_detection.vision.pose") == "True"


@pytest.mark.parametrize("module", ALL_MODULES, ids=module_id)
def test_no_module_declares_a_configured_value(module: Path) -> None:
    declared = {name for name in CONFIGURED_NAMES if _assigns(module, name)}

    assert declared == set(), f"{module_id(module)} should read these from CONFIG"


def _ml_loaded_after(statement: str) -> str:
    probe = (
        f"{statement}; import sys; "
        f"print(any(name.split('.')[0] in {ISOLATED_ROOTS!r} for name in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def _assigns(module: Path, name: str) -> bool:
    return any(
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
        for node in ast.walk(ast.parse(module.read_text()))
    )


def _imported_roots(module: Path) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(module.read_text())):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            roots.add(node.module.split(".")[0])
    return roots
