import json
from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest

from evidone_pose_detection.detection_log import DetectionLog
from evidone_pose_detection.errors import DetectionLogError
from evidone_pose_detection.models import (
    KEYPOINT_COUNT,
    BoundingBox,
    COCOKeypoint,
    Keypoint,
    PersonPose,
)

LOW = 0.01


@pytest.fixture
def folder() -> Iterator[Path]:
    with TemporaryDirectory() as directory:
        yield Path(directory)


def person(score: float, keypoint_confidence: float) -> PersonPose:
    return PersonPose(
        box=BoundingBox(left=10.0, top=20.0, right=110.0, bottom=220.0),
        confidence=score,
        keypoints=tuple(
            Keypoint(x=float(index), y=float(index * 2), confidence=keypoint_confidence)
            for index in range(KEYPOINT_COUNT)
        ),
    )


def lines_of(destination: Path) -> list[dict[str, Any]]:
    text = destination.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line]


def test_a_log_starts_switched_off() -> None:
    log = DetectionLog()

    assert log.is_logging is False
    assert log.destination is None


def test_writing_while_switched_off_does_nothing(folder: Path) -> None:
    DetectionLog().write([person(0.9, 0.9)])

    assert list(folder.iterdir()) == []


def test_starting_opens_the_file_straight_away(folder: Path) -> None:
    destination = folder / "run.jsonl"
    log = DetectionLog()

    log.start(destination)

    assert log.is_logging is True
    assert log.destination == destination
    assert destination.exists(), "the file was not opened until a frame arrived"


def test_one_line_per_frame_numbered_from_zero(folder: Path) -> None:
    destination = folder / "run.jsonl"
    log = DetectionLog()
    log.start(destination)

    for _ in range(4):
        log.write([person(0.9, 0.9)])
    log.close()

    records = lines_of(destination)
    assert len(records) == 4
    assert [record["frame"] for record in records] == [0, 1, 2, 3]


def test_a_frame_with_nobody_in_it_still_gets_a_line(folder: Path) -> None:
    destination = folder / "run.jsonl"
    log = DetectionLog()
    log.start(destination)

    log.write([person(0.9, 0.9)])
    log.write([])
    log.write([person(0.9, 0.9)])
    log.close()

    records = lines_of(destination)
    assert [record["frame"] for record in records] == [0, 1, 2]
    assert records[1]["people"] == [], "an empty frame broke the numbering"


def test_a_person_carries_a_box_a_score_and_every_keypoint(folder: Path) -> None:
    destination = folder / "run.jsonl"
    log = DetectionLog()
    log.start(destination)

    log.write([person(0.77, 0.5)])
    log.close()

    (record,) = lines_of(destination)
    (logged,) = record["people"]
    assert logged["score"] == pytest.approx(0.77)
    assert logged["box"] == {"left": 10.0, "top": 20.0, "right": 110.0, "bottom": 220.0}
    assert len(logged["keypoints"]) == KEYPOINT_COUNT
    assert [point["name"] for point in logged["keypoints"]] == [
        name.name.lower() for name in COCOKeypoint
    ]
    assert all("confidence" in point for point in logged["keypoints"])


def test_low_confidence_keypoints_are_kept_not_dropped(folder: Path) -> None:
    destination = folder / "run.jsonl"
    log = DetectionLog()
    log.start(destination)

    log.write([person(0.9, LOW)])
    log.close()

    (record,) = lines_of(destination)
    (logged,) = record["people"]
    assert len(logged["keypoints"]) == KEYPOINT_COUNT, "the log filtered the model's output"
    assert all(point["confidence"] == pytest.approx(LOW) for point in logged["keypoints"])


def test_every_person_in_the_frame_is_logged(folder: Path) -> None:
    destination = folder / "run.jsonl"
    log = DetectionLog()
    log.start(destination)

    log.write([person(0.9, 0.8), person(0.4, 0.2), person(0.6, 0.6)])
    log.close()

    (record,) = lines_of(destination)
    assert [entry["score"] for entry in record["people"]] == pytest.approx([0.9, 0.4, 0.6])


def test_each_line_is_a_complete_object_on_its_own(folder: Path) -> None:
    destination = folder / "run.jsonl"
    log = DetectionLog()
    log.start(destination)

    for _ in range(3):
        log.write([person(0.9, 0.9)])
    log.close()

    for line in destination.read_text(encoding="utf-8").splitlines():
        assert json.loads(line)["frame"] >= 0, "a line was not parseable on its own"


def test_the_tail_is_readable_before_the_log_is_closed(folder: Path) -> None:
    destination = folder / "run.jsonl"
    log = DetectionLog()
    log.start(destination)

    log.write([person(0.9, 0.9)])
    log.write([person(0.8, 0.9)])
    streamed = lines_of(destination)

    log.close()
    assert [record["frame"] for record in streamed] == [0, 1], "lines were stuck in a buffer"


def test_a_run_cut_short_still_leaves_usable_lines(folder: Path) -> None:
    destination = folder / "run.jsonl"
    log = DetectionLog()
    log.start(destination)
    for _ in range(5):
        log.write([person(0.9, 0.9)])

    # No close(): the object is simply abandoned, as a hard kill would.
    assert len(lines_of(destination)) == 5


def test_closing_leaves_the_file_complete(folder: Path) -> None:
    destination = folder / "run.jsonl"
    log = DetectionLog()
    log.start(destination)
    log.write([person(0.9, 0.9)])

    log.close()

    assert log.is_logging is False
    assert log.destination is None
    assert len(lines_of(destination)) == 1


def test_closing_twice_is_harmless(folder: Path) -> None:
    log = DetectionLog()
    log.start(folder / "run.jsonl")

    log.close()
    log.close()

    assert log.is_logging is False


def test_closing_without_ever_starting_is_harmless() -> None:
    DetectionLog().close()


def test_writing_after_closing_does_nothing(folder: Path) -> None:
    destination = folder / "run.jsonl"
    log = DetectionLog()
    log.start(destination)
    log.write([person(0.9, 0.9)])
    log.close()

    log.write([person(0.9, 0.9)])

    assert len(lines_of(destination)) == 1


def test_logging_again_starts_a_fresh_file_and_numbering(folder: Path) -> None:
    first, second = folder / "one.jsonl", folder / "two.jsonl"
    log = DetectionLog()
    log.start(first)
    log.write([person(0.9, 0.9)])
    log.write([person(0.9, 0.9)])
    log.close()

    log.start(second)
    log.write([person(0.9, 0.9)])
    log.close()

    assert [record["frame"] for record in lines_of(first)] == [0, 1]
    assert [record["frame"] for record in lines_of(second)] == [0]


def test_a_destination_that_cannot_be_opened_is_reported(folder: Path) -> None:
    destination = folder / "no-such-folder" / "run.jsonl"
    log = DetectionLog()

    with pytest.raises(DetectionLogError) as error:
        log.start(destination)

    assert error.value.destination == destination
    assert str(destination) in str(error.value)
    assert log.is_logging is False
