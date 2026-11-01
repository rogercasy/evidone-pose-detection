import tomllib
from collections.abc import Mapping
from dataclasses import fields
from enum import Enum
from importlib.resources import files
from types import MappingProxyType
from typing import Any

from evidone_pose_detection.errors import ConfigError
from evidone_pose_detection.models import (
    BodyRegion,
    CameraSettings,
    COCOKeypoint,
    Config,
    FaceSettings,
    FaceSpan,
    HideStyle,
    HidingSettings,
    LabelSettings,
    OutputSettings,
    PanelSettings,
    PoseSettings,
    SkeletonSettings,
    TrackingSettings,
    WindowSettings,
)

CONFIG_FILE = "config.toml"

SECTIONS = MappingProxyType(
    {
        "camera": CameraSettings,
        "pose": PoseSettings,
        "faces": FaceSettings,
        "tracking": TrackingSettings,
        "hiding": HidingSettings,
        "skeleton": SkeletonSettings,
        "panel": PanelSettings,
        "window": WindowSettings,
        "output": OutputSettings,
        "labels": LabelSettings,
    }
)


def load_config(text: str) -> Config:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"the configuration is not valid TOML: {error}") from error

    _reject_unknown("", set(data) - set(SECTIONS))
    built = {name: _section(data, name) for name in SECTIONS}
    return Config(**built)


def read_packaged_config() -> Config:
    source = files("evidone_pose_detection").joinpath(CONFIG_FILE)
    try:
        text = source.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError) as error:
        raise ConfigError(f"the packaged {CONFIG_FILE} is missing") from error
    return load_config(text)


def _section(data: Mapping[str, Any], name: str) -> Any:
    table = data.get(name)
    if not isinstance(table, dict):
        raise ConfigError(f"the configuration has no [{name}] section")

    values = dict(table)
    expected = {field.name for field in fields(SECTIONS[name])}
    _reject_unknown(name, set(values) - expected)
    _require(name, expected - set(values))
    return SECTIONS[name](**_converted(name, values))


def _converted(name: str, values: dict[str, Any]) -> dict[str, Any]:
    if name == "faces":
        values["spans"] = tuple(_span(span) for span in values["spans"])
    if name == "hiding":
        values["default_style"] = _member(
            HideStyle, values["default_style"], "hiding.default_style"
        )
    if name == "skeleton":
        values["colours"] = _keyed(BodyRegion, values["colours"], "skeleton.colours")
    if name == "labels":
        values["styles"] = _keyed(HideStyle, values["styles"], "labels.styles")
    return {key: _as_tuple(value) for key, value in values.items()}


def _span(span: Any) -> FaceSpan:
    if not isinstance(span, dict) or set(span) != {"start", "end", "scale"}:
        raise ConfigError("every [[faces.spans]] needs exactly start, end and scale")
    return FaceSpan(
        start=_member(COCOKeypoint, span["start"], "faces.spans.start"),
        end=_member(COCOKeypoint, span["end"], "faces.spans.end"),
        radius_scale=span["scale"],
    )


def _keyed[T: Enum](kind: type[T], table: Any, where: str) -> Mapping[T, Any]:
    if not isinstance(table, dict):
        raise ConfigError(f"[{where}] must be a table")
    expected = {member.name.lower() for member in kind}
    _reject_unknown(where, set(table) - expected)
    _require(where, expected - set(table))
    return MappingProxyType(
        {_member(kind, key, where): _as_tuple(value) for key, value in table.items()}
    )


def _member[T: Enum](kind: type[T], name: Any, where: str) -> T:
    for member in kind:
        if member.name.lower() == name:
            return member
    known = ", ".join(sorted(member.name.lower() for member in kind))
    raise ConfigError(f"{where}: {name!r} is not one of {known}")


def _as_tuple(value: Any) -> Any:
    return tuple(value) if isinstance(value, list) else value


def _reject_unknown(where: str, unknown: set[str]) -> None:
    if unknown:
        place = f"[{where}]" if where else "the configuration"
        raise ConfigError(f"{place} has unknown keys: {', '.join(sorted(unknown))}")


def _require(where: str, missing: set[str]) -> None:
    if missing:
        raise ConfigError(f"[{where}] is missing keys: {', '.join(sorted(missing))}")


CONFIG = read_packaged_config()
