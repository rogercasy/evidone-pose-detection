<div align="center">

<img src="assets/logo.svg" alt="" width="96">

# Evidone Pose Detection

**Real-Time Pose Detection with Face Anonymization**

[![QA](https://github.com/rogercasy/evidone-pose-detection/actions/workflows/qa.yml/badge.svg)](https://github.com/rogercasy/evidone-pose-detection/actions/workflows/qa.yml)
[![Release](https://img.shields.io/github/v/release/rogercasy/evidone-pose-detection?label=release)](https://github.com/rogercasy/evidone-pose-detection/releases/latest)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
<br>
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

<!-- Badges to move into the rows above once they apply:
[![PyPI](https://img.shields.io/pypi/v/evidone_pose_detection)](https://pypi.org/project/evidone_pose_detection/)
[![Docker image](https://img.shields.io/badge/docker-ghcr.io-2496ED?logo=docker&logoColor=white)](https://github.com/rogercasy/evidone-pose-detection/pkgs/container/evidone-pose-detection)
[![Docs](https://readthedocs.org/projects/evidone-pose-detection/badge/?version=latest)](https://evidone-pose-detection.readthedocs.io)
-->

[Quick start](#quick-start) · [Usage](#usage) · [Development](#development)

</div>

---

_What it does, for whom, and why it's different, in two or three sentences._

## Features

- _**Feature.** What it gives the user._

## Quick start

```bash
uv sync
```

```bash
cp .env.example .env  # Configure as needed
uv run evidone
```

## Usage

_The smallest example that shows the main thing it does._

## Module layout

```text
src/evidone_pose_detection/
├── main.py          # entry point: CLI -> Qt app -> camera lifetime
├── cli.py           # argument parsing into AppConfig
├── camera.py        # cv2.VideoCapture wrapper; releases the device on the way out
├── models.py        # shared dataclasses, COCO keypoints, skeleton and face landmark data
├── errors.py        # error types
├── vision/          # everything that reads or paints a frame
│   ├── pose.py          # the only module that imports torch / ultralytics
│   ├── faces.py         # face regions worked out from pose keypoints alone
│   ├── tracking.py      # matches regions across frames, smooths and holds them
│   ├── anonymization.py # where and how much to conceal: clipping, masking, dispatch
│   ├── styles.py        # what concealment looks like: blur, pixelate, solid fill
│   └── rendering.py     # draws the COCO-17 skeleton
└── ui/
    └── window.py    # Qt window; drives the per-frame loop
```

Each frame runs `estimate → blur → draw`, so the skeleton stays sharp on top of a hidden face.

`vision/__init__.py` re-exports `blur_faces`, `draw_poses` and `face_region` — but deliberately
not `PoseEstimator`. That class is the single door to torch/ultralytics (~0.9 s to import), so it
is reached at `evidone_pose_detection.vision.pose` and importing `vision` or `ui` loads no ML
stack at all.

`tests/` mirrors this tree (`vision/faces.py` → `tests/vision/test_faces.py`), and
`tests/test_architecture.py` enforces both rules: that only `vision/pose.py` names the ML
packages, and that importing `ui` or `vision` really does not pull them in.

## Privacy trade-off

A face region is kept and kept blurred for a few frames after the model stops reporting it
(`DEFAULT_HOLD_FRAMES` in `vision/tracking.py`; `0` turns holding off).

**What it buys.** The model drops a face now and then — a turned head, motion blur, an awkward
frame. Without holding, every drop is one clean, unblurred frame of that person's face.

**What it costs.**

- A held region sits where the face was *last seen*. We do not know where the face actually
  went, so if it kept moving it can travel out from under the cover. The region widens by a
  fixed share of its radius per held frame to absorb some of that, but widening is a guess, not
  a measurement, and a fast-moving face can still be exposed.
- That widening blurs more of the surrounding picture for every frame a face is held.
- A false detection is held on exactly the same terms, so a wrong blur lingers just as long.
- When the budget runs out the region is dropped whether or not the face is still there.
  Holding delays exposure; it does not remove it.

Raising the budget covers longer dropouts and enlarges every one of those costs in step.

## Development

Setup, QA, docs and releases: [DEVELOPMENT.md](DEVELOPMENT.md).

## Contributing

Issues and pull requests are welcome; `make qa` must pass.

---

<sub>🚀 Project starter provided by [Cookie Pyrate v1.1.0](https://github.com/gvieralopez/cookie-pyrate/tree/v1.1.0)</sub>
