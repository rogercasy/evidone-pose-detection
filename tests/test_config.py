import tomllib
from importlib.resources import files
from pathlib import Path

import pytest

from evidone_pose_detection.config import CONFIG, CONFIG_FILE, SECTIONS, load_config
from evidone_pose_detection.errors import ConfigError
from evidone_pose_detection.models import BodyRegion, Config, HideStyle

PACKAGED = files("evidone_pose_detection").joinpath(CONFIG_FILE).read_text(encoding="utf-8")


def test_the_configuration_ships_inside_the_package() -> None:
    assert PACKAGED.strip() != ""
    assert (Path(__file__).parent.parent / "src" / "evidone_pose_detection" / CONFIG_FILE).exists()


def test_it_is_loaded_once_as_typed_objects() -> None:
    assert isinstance(CONFIG, Config)
    assert isinstance(CONFIG.panel.alpha, float)
    assert isinstance(CONFIG.window.minimum_yield_ms, int)
    assert isinstance(CONFIG.skeleton.draw_by_default, bool)
    assert CONFIG is load_config.__globals__["CONFIG"]


def test_every_section_has_a_table_of_its_own() -> None:
    data = tomllib.loads(PACKAGED)

    assert set(data) == set(SECTIONS)


def test_enum_keyed_tables_are_resolved_to_members() -> None:
    assert set(CONFIG.skeleton.colours) == set(BodyRegion)
    assert set(CONFIG.labels.styles) == set(HideStyle)
    assert CONFIG.hiding.default_style is HideStyle.BLUR


def test_arrays_become_tuples_not_lists() -> None:
    assert isinstance(CONFIG.panel.origin, tuple)
    assert isinstance(CONFIG.hiding.fill_colour, tuple)
    assert isinstance(CONFIG.faces.spans, tuple)


def test_the_face_spans_are_resolved_to_keypoints() -> None:
    span = CONFIG.faces.spans[0]

    assert span.start.name.lower() in {"left_eye", "right_eye", "nose", "left_ear", "right_ear"}
    assert isinstance(span.radius_scale, float)


@pytest.mark.parametrize("section", list(SECTIONS))
def test_a_missing_section_is_refused(section: str) -> None:
    data = tomllib.loads(PACKAGED)
    del data[section]

    with pytest.raises(ConfigError, match=section):
        load_config(_as_toml(data))


@pytest.mark.parametrize(
    ("section", "key"),
    [("panel", "alpha"), ("tracking", "smoothing"), ("window", "title"), ("pose", "weights")],
)
def test_a_missing_key_is_refused(section: str, key: str) -> None:
    data = tomllib.loads(PACKAGED)
    del data[section][key]

    with pytest.raises(ConfigError, match=key):
        load_config(_as_toml(data))


def test_an_unknown_key_in_a_section_is_refused() -> None:
    data = tomllib.loads(PACKAGED)
    data["panel"]["brightness"] = 1

    with pytest.raises(ConfigError, match="brightness"):
        load_config(_as_toml(data))


def test_an_unknown_section_is_refused() -> None:
    data = tomllib.loads(PACKAGED)
    data["telemetry"] = {"enabled": True}

    with pytest.raises(ConfigError, match="telemetry"):
        load_config(_as_toml(data))


def test_an_unknown_enum_key_is_refused() -> None:
    data = tomllib.loads(PACKAGED)
    data["skeleton"]["colours"]["tail"] = [1, 2, 3]

    with pytest.raises(ConfigError, match="tail"):
        load_config(_as_toml(data))


def test_an_unknown_style_name_is_refused() -> None:
    data = tomllib.loads(PACKAGED)
    data["hiding"]["default_style"] = "melt"

    with pytest.raises(ConfigError, match="melt"):
        load_config(_as_toml(data))


def test_an_unknown_keypoint_in_a_span_is_refused() -> None:
    data = tomllib.loads(PACKAGED)
    data["faces"]["spans"][0]["start"] = "antenna"

    with pytest.raises(ConfigError, match="antenna"):
        load_config(_as_toml(data))


def test_a_span_missing_a_field_is_refused() -> None:
    data = tomllib.loads(PACKAGED)
    del data["faces"]["spans"][0]["scale"]

    with pytest.raises(ConfigError, match="spans"):
        load_config(_as_toml(data))


def test_text_that_is_not_toml_is_refused() -> None:
    with pytest.raises(ConfigError, match="TOML"):
        load_config("this is not = = toml")


def _as_toml(data: dict) -> str:
    lines: list[str] = []
    for section, table in data.items():
        lines.append(f"[{section}]")
        for key, value in table.items():
            if isinstance(value, dict):
                continue
            if isinstance(value, list) and value and isinstance(value[0], dict):
                continue
            lines.append(f"{key} = {_literal(value)}")
        for key, value in table.items():
            if isinstance(value, dict):
                lines.append(f"[{section}.{key}]")
                lines.extend(f"{inner} = {_literal(item)}" for inner, item in value.items())
            if isinstance(value, list) and value and isinstance(value[0], dict):
                for entry in value:
                    lines.append(f"[[{section}.{key}]]")
                    lines.extend(f"{inner} = {_literal(item)}" for inner, item in entry.items())
    return "\n".join(lines)


def _literal(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, list):
        return "[" + ", ".join(_literal(item) for item in value) + "]"
    return str(value)
