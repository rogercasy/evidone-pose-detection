import numpy as np
import pytest

from evidone_pose_detection.models import (
    FACE_LANDMARKS,
    VISIBILITY_THRESHOLD,
    BoundingBox,
    COCOKeypoint,
    Frame,
    HideStyle,
    Keypoint,
    PersonPose,
)
from evidone_pose_detection.vision.anonymization import STYLES, hide_faces
from evidone_pose_detection.vision.faces import face_regions
from evidone_pose_detection.vision.styles import FILL_COLOUR

WIDTH = 400
HEIGHT = 400

# The suite below pins behaviour shared by every style; STYLE is the one it runs with.
STYLE = HideStyle.BLUR
SEEN = 0.9
UNSEEN = VISIBILITY_THRESHOLD - 0.01


def checkerboard(cell: int) -> Frame:
    rows = (np.arange(HEIGHT) // cell)[:, None]
    columns = (np.arange(WIDTH) // cell)[None, :]
    pattern = np.where((rows + columns) % 2 == 0, 0, 255).astype(np.uint8)
    return np.repeat(pattern[:, :, None], 3, axis=2)


def face_at(centre_x: float, centre_y: float, half_width: float) -> PersonPose:
    points = {
        COCOKeypoint.NOSE: (centre_x, centre_y),
        COCOKeypoint.LEFT_EYE: (centre_x - half_width * 0.3, centre_y - half_width * 0.2),
        COCOKeypoint.RIGHT_EYE: (centre_x + half_width * 0.3, centre_y - half_width * 0.2),
        COCOKeypoint.LEFT_EAR: (centre_x - half_width, centre_y),
        COCOKeypoint.RIGHT_EAR: (centre_x + half_width, centre_y),
    }
    keypoints = tuple(
        Keypoint(x=points[name][0], y=points[name][1], confidence=SEEN)
        if name in points
        else Keypoint(0.0, 0.0, 0.0)
        for name in COCOKeypoint
    )
    return PersonPose(
        box=BoundingBox(left=0.0, top=0.0, right=float(WIDTH), bottom=float(HEIGHT)),
        confidence=0.9,
        keypoints=keypoints,
    )


def faceless_person() -> PersonPose:
    keypoints = tuple(
        Keypoint(100.0, 100.0, UNSEEN) if name in FACE_LANDMARKS else Keypoint(0.0, 0.0, 0.0)
        for name in COCOKeypoint
    )
    return PersonPose(
        box=BoundingBox(left=0.0, top=0.0, right=1.0, bottom=1.0),
        confidence=0.9,
        keypoints=keypoints,
    )


def detail_at(frame: Frame, centre_x: int, centre_y: int, reach: int) -> float:
    patch = frame[
        max(0, centre_y - reach) : centre_y + reach, max(0, centre_x - reach) : centre_x + reach
    ]
    return float(patch.std())


@pytest.mark.parametrize("half_width", [12.0, 40.0, 120.0])
def test_the_face_is_unrecognisable_whatever_size_it_is(half_width: float) -> None:
    # Detail scales with the face, so a fixed-strength blur would fail the large cases.
    frame = checkerboard(cell=max(1, round(half_width / 4)))
    person = face_at(200.0, 200.0, half_width)
    reach = round(half_width / 2)
    before = detail_at(frame, 200, 200, reach)

    hide_faces(frame, face_regions([person]), STYLE)

    assert detail_at(frame, 200, 200, reach) < before * 0.1


def test_a_large_face_loses_as_much_detail_as_a_small_one() -> None:
    residues = []
    for half_width in (12.0, 120.0):
        frame = checkerboard(cell=max(1, round(half_width / 4)))
        reach = round(half_width / 2)
        before = detail_at(frame, 200, 200, reach)
        hide_faces(frame, face_regions([face_at(200.0, 200.0, half_width)]), STYLE)
        residues.append(detail_at(frame, 200, 200, reach) / before)

    small, large = residues
    assert large == pytest.approx(small, abs=0.1)


def test_pixels_well_outside_the_region_are_untouched() -> None:
    frame = checkerboard(cell=8)
    original = frame.copy()

    hide_faces(frame, face_regions([face_at(120.0, 120.0, 30.0)]), STYLE)

    assert (frame[350:, 350:] == original[350:, 350:]).all()


@pytest.mark.parametrize(
    ("centre_x", "centre_y"),
    [
        (0.0, 200.0),
        (float(WIDTH), 200.0),
        (200.0, 0.0),
        (200.0, float(HEIGHT)),
        (0.0, 0.0),
        (float(WIDTH), float(HEIGHT)),
        (-20.0, 200.0),
        (float(WIDTH) + 20.0, float(HEIGHT) + 20.0),
    ],
)
def test_a_region_crossing_the_frame_edge_is_clipped_not_fatal(
    centre_x: float, centre_y: float
) -> None:
    frame = checkerboard(cell=6)
    original = frame.copy()

    hide_faces(frame, face_regions([face_at(centre_x, centre_y, 60.0)]), STYLE)

    assert frame.shape == original.shape
    assert (frame != original).any()


def test_a_region_entirely_off_frame_changes_nothing() -> None:
    frame = checkerboard(cell=6)
    original = frame.copy()

    hide_faces(frame, face_regions([face_at(-500.0, -500.0, 40.0)]), STYLE)

    assert (frame == original).all()


def test_a_person_without_a_visible_face_is_left_alone() -> None:
    frame = checkerboard(cell=6)
    original = frame.copy()

    hide_faces(frame, face_regions([faceless_person()]), STYLE)

    assert (frame == original).all()


def test_a_frame_without_people_is_left_alone() -> None:
    frame = checkerboard(cell=6)
    original = frame.copy()

    hide_faces(frame, face_regions([]), STYLE)

    assert (frame == original).all()


def test_every_face_in_the_frame_is_hidden() -> None:
    frame = checkerboard(cell=8)
    people = [face_at(100.0, 100.0, 40.0), face_at(300.0, 300.0, 40.0)]
    before = [detail_at(frame, 100, 100, 20), detail_at(frame, 300, 300, 20)]

    hide_faces(frame, face_regions(people), STYLE)

    assert detail_at(frame, 100, 100, 20) < before[0] * 0.1
    assert detail_at(frame, 300, 300, 20) < before[1] * 0.1


def test_blurring_mutates_the_frame_it_was_given_and_hands_it_back() -> None:
    frame = checkerboard(cell=8)

    hidden = hide_faces(frame, face_regions([face_at(200.0, 200.0, 40.0)]), STYLE)

    assert hidden is frame


# --- every style, on its own terms ------------------------------------------------------

ALL_STYLES = list(HideStyle)
FACE_CENTRE = (200, 200)
FACE_HALF_WIDTH = 60.0
OUTSIDE = (slice(0, 40), slice(0, 40))


def hidden_frame(style: HideStyle, cell: int) -> tuple[Frame, Frame]:
    frame = checkerboard(cell=cell)
    original = frame.copy()
    hide_faces(frame, face_regions([face_at(200.0, 200.0, FACE_HALF_WIDTH)]), style)
    return frame, original


def test_every_style_has_an_entry_and_nothing_is_missing() -> None:
    assert set(STYLES) == set(HideStyle)
    assert len(STYLES) == len(ALL_STYLES)


@pytest.mark.parametrize("style", ALL_STYLES, ids=str)
def test_every_style_changes_the_region(style: HideStyle) -> None:
    frame, original = hidden_frame(style, cell=8)

    centre = frame[FACE_CENTRE[1], FACE_CENTRE[0]]
    assert (frame != original).any(), "the style left the frame untouched"
    assert (centre != original[FACE_CENTRE[1], FACE_CENTRE[0]]).any()


@pytest.mark.parametrize("style", ALL_STYLES, ids=str)
def test_every_style_leaves_the_rest_of_the_frame_alone(style: HideStyle) -> None:
    frame, original = hidden_frame(style, cell=8)

    assert (frame[OUTSIDE] == original[OUTSIDE]).all(), "a far corner was touched"
    assert (frame[360:, 360:] == original[360:, 360:]).all()
    changed_rows = np.any(frame != original, axis=(1, 2)).nonzero()[0]
    assert changed_rows.min() >= 200 - FACE_HALF_WIDTH * 3
    assert changed_rows.max() <= 200 + FACE_HALF_WIDTH * 3


@pytest.mark.parametrize("style", ALL_STYLES, ids=str)
def test_every_style_makes_the_face_unrecognisable(style: HideStyle) -> None:
    frame = checkerboard(cell=8)
    before = detail_at(frame, 200, 200, 30)

    hide_faces(frame, face_regions([face_at(200.0, 200.0, FACE_HALF_WIDTH)]), style)

    assert detail_at(frame, 200, 200, 30) < before * 0.3


@pytest.mark.parametrize("style", ALL_STYLES, ids=str)
@pytest.mark.parametrize("half_width", [12.0, 40.0, 120.0])
def test_every_style_holds_up_at_any_face_size(style: HideStyle, half_width: float) -> None:
    frame = checkerboard(cell=max(1, round(half_width / 4)))
    reach = round(half_width / 2)
    before = detail_at(frame, 200, 200, reach)

    hide_faces(frame, face_regions([face_at(200.0, 200.0, half_width)]), style)

    assert detail_at(frame, 200, 200, reach) < before * 0.25


@pytest.mark.parametrize("style", ALL_STYLES, ids=str)
@pytest.mark.parametrize(
    ("centre_x", "centre_y"), [(0.0, 200.0), (float(WIDTH), 200.0), (200.0, 0.0), (-20.0, -20.0)]
)
def test_every_style_clips_at_the_frame_edge(
    style: HideStyle, centre_x: float, centre_y: float
) -> None:
    frame = checkerboard(cell=6)
    original = frame.copy()

    hide_faces(frame, face_regions([face_at(centre_x, centre_y, 60.0)]), style)

    assert frame.shape == original.shape
    assert (frame != original).any()


@pytest.mark.parametrize("style", ALL_STYLES, ids=str)
def test_every_style_ignores_a_region_entirely_off_frame(style: HideStyle) -> None:
    frame = checkerboard(cell=6)
    original = frame.copy()

    hide_faces(frame, face_regions([face_at(-900.0, -900.0, 40.0)]), style)

    assert (frame == original).all()


@pytest.mark.parametrize("style", ALL_STYLES, ids=str)
def test_every_style_is_given_nothing_and_does_nothing(style: HideStyle) -> None:
    frame = checkerboard(cell=6)
    original = frame.copy()

    hide_faces(frame, [], style)

    assert (frame == original).all()


def test_a_solid_fill_leaves_one_colour_behind() -> None:
    frame = checkerboard(cell=8)

    hide_faces(frame, face_regions([face_at(200.0, 200.0, FACE_HALF_WIDTH)]), HideStyle.FILL)

    assert tuple(int(channel) for channel in frame[200, 200]) == FILL_COLOUR
    assert detail_at(frame, 200, 200, 20) == pytest.approx(0.0)
