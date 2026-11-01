import cv2
import numpy as np

from evidone_pose_detection.config import CONFIG
from evidone_pose_detection.models import Frame, FrameStats

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_stats(frame: Frame, stats: FrameStats) -> Frame:
    lines = _lines(stats)
    left, top = CONFIG.panel.origin
    height, width = frame.shape[:2]
    right = min(width, left + _panel_width(lines))
    bottom = min(height, top + _panel_height(lines))
    if right <= left or bottom <= top:
        return frame

    _shade(frame[top:bottom, left:right])
    for index, line in enumerate(lines, start=1):
        baseline = (
            top
            + CONFIG.panel.padding
            + CONFIG.panel.line_height * index
            - CONFIG.panel.font_thickness * 5
        )
        cv2.putText(
            frame,
            line,
            (left + CONFIG.panel.padding, baseline),
            FONT,
            CONFIG.panel.font_scale,
            CONFIG.panel.text_colour,
            CONFIG.panel.font_thickness,
            cv2.LINE_AA,
        )
    return frame


def _lines(stats: FrameStats) -> list[str]:
    return [
        f"inference {stats.inference_ms:6.1f} ms",
        f"display   {stats.frames_per_second:6.1f} fps",
        f"people    {stats.people:6d}",
    ]


def _panel_width(lines: list[str]) -> int:
    widest = max(
        cv2.getTextSize(line, FONT, CONFIG.panel.font_scale, CONFIG.panel.font_thickness)[0][0]
        for line in lines
    )
    return widest + CONFIG.panel.padding * 2


def _panel_height(lines: list[str]) -> int:
    return CONFIG.panel.line_height * len(lines) + CONFIG.panel.padding * 2


def _shade(panel: Frame) -> None:
    backdrop = np.full(panel.shape, CONFIG.panel.colour, dtype=np.uint8)
    cv2.addWeighted(backdrop, CONFIG.panel.alpha, panel, 1.0 - CONFIG.panel.alpha, 0.0, dst=panel)
