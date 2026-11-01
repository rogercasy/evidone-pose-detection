import cv2
import numpy as np

from evidone_pose_detection.models import Frame, FrameStats

PANEL_ORIGIN = (12, 12)
PANEL_PADDING = 10
PANEL_OPACITY = 0.55
PANEL_COLOUR = (24, 24, 24)
TEXT_COLOUR = (240, 240, 240)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 1
LINE_HEIGHT = 20


def draw_stats(frame: Frame, stats: FrameStats) -> Frame:
    lines = _lines(stats)
    left, top = PANEL_ORIGIN
    height, width = frame.shape[:2]
    right = min(width, left + _panel_width(lines))
    bottom = min(height, top + _panel_height(lines))
    if right <= left or bottom <= top:
        return frame

    _shade(frame[top:bottom, left:right])
    for index, line in enumerate(lines, start=1):
        baseline = top + PANEL_PADDING + LINE_HEIGHT * index - FONT_THICKNESS * 5
        cv2.putText(
            frame,
            line,
            (left + PANEL_PADDING, baseline),
            FONT,
            FONT_SCALE,
            TEXT_COLOUR,
            FONT_THICKNESS,
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
    widest = max(cv2.getTextSize(line, FONT, FONT_SCALE, FONT_THICKNESS)[0][0] for line in lines)
    return widest + PANEL_PADDING * 2


def _panel_height(lines: list[str]) -> int:
    return LINE_HEIGHT * len(lines) + PANEL_PADDING * 2


def _shade(panel: Frame) -> None:
    backdrop = np.full(panel.shape, PANEL_COLOUR, dtype=np.uint8)
    cv2.addWeighted(backdrop, PANEL_OPACITY, panel, 1.0 - PANEL_OPACITY, 0.0, dst=panel)
