#!/usr/bin/env python3
"""Dynamic GenICam feature discovery and editing for HIKROBOT cameras.

The MVS SDK deliberately exposes typed get/set calls rather than a modern
"enumerate every feature" call.  This module therefore uses the device's
GenICam XML for the ordered category tree and descriptive metadata, then asks
the SDK for every node's *current* interface type and access mode.  The latter
is important: availability and writability change while grabbing and when
selector/automatic-control nodes change.

No MVS binding is imported at module import time.  ``CameraFeatureAccessor``
loads the binding module belonging to the bound camera object only when the
first SDK-backed operation is performed.  The XML parser can consequently be
used and tested on machines without MVS or PyQt installed.

``CameraFeaturePanel.after_change`` is a cleanup callback as well as a success
notification: once ``before_change`` has approved an operation it is called in
a ``finally`` block, including when the SDK write fails.  This lets a host that
temporarily stopped acquisition reliably restart it.  Failures are also
reported through ``change_failed``.
"""

from __future__ import annotations

import importlib
import io
import re
import zipfile
from contextlib import contextmanager
from ctypes import c_bool, c_int, c_ubyte, c_uint
from dataclasses import dataclass, replace
from typing import Callable, Iterator
from xml.etree import ElementTree


# Values are fixed by CameraParams.h / GenApi and intentionally duplicated
# here so that parsing and policy helpers do not import the vendor binding.
AM_NI = 0
AM_NA = 1
AM_WO = 2
AM_RO = 3
AM_RW = 4
AM_UNDEFINED = 5
AM_CYCLE_DETECT = 6

IFT_IVALUE = 0
IFT_IBASE = 1
IFT_IINTEGER = 2
IFT_IBOOLEAN = 3
IFT_ICOMMAND = 4
IFT_IFLOAT = 5
IFT_ISTRING = 6
IFT_IREGISTER = 7
IFT_ICATEGORY = 8
IFT_IENUMERATION = 9
IFT_IENUMENTRY = 10
IFT_IPORT = 11

ACCESS_MODE_NAMES = {
    AM_NI: "NI",
    AM_NA: "NA",
    AM_WO: "WO",
    AM_RO: "RO",
    AM_RW: "RW",
    AM_UNDEFINED: "Undefined",
    AM_CYCLE_DETECT: "CycleDetect",
}

INTERFACE_ID_NAMES = {
    IFT_IINTEGER: "Integer",
    IFT_IBOOLEAN: "Boolean",
    IFT_ICOMMAND: "Command",
    IFT_IFLOAT: "Float",
    IFT_ISTRING: "String",
    IFT_IREGISTER: "Register",
    IFT_ICATEGORY: "Category",
    IFT_IENUMERATION: "Enumeration",
    IFT_IENUMENTRY: "EnumEntry",
    IFT_IPORT: "Port",
}

_DECLARED_INTERFACE_TYPES = {
    "Integer": "Integer",
    "IntReg": "Integer",
    "MaskedIntReg": "Integer",
    "IntConverter": "Integer",
    "IntSwissKnife": "Integer",
    "Boolean": "Boolean",
    "Command": "Command",
    "Float": "Float",
    "Converter": "Float",
    "SwissKnife": "Float",
    "String": "String",
    "StringReg": "String",
    "Enumeration": "Enumeration",
    "Category": "Category",
    "Register": "Register",
    "StructReg": "Register",
    "Port": "Port",
}

_EDITABLE_INTERFACE_TYPES = {
    "Integer",
    "Boolean",
    "Command",
    "Float",
    "String",
    "Enumeration",
}


class CameraFeatureError(RuntimeError):
    """Raised when feature discovery, reading, or writing fails."""


@dataclass(frozen=True)
class EnumEntrySpec:
    """Static metadata for one enumeration entry from GenICam XML."""

    name: str
    value: int | None = None
    display_name: str = ""
    description: str = ""


@dataclass(frozen=True)
class EnumChoice:
    """One enumeration value currently supported by the device."""

    value: int
    symbolic: str
    display_name: str = ""


@dataclass(frozen=True)
class FeatureSpec:
    """Static XML metadata plus optional runtime GenApi information."""

    name: str
    category: str = ""
    category_path: tuple[str, ...] = ()
    interface_type: str = ""
    display_name: str = ""
    description: str = ""
    tooltip: str = ""
    visibility: str = "Beginner"
    unit: str = ""
    representation: str = ""
    streamable: bool | None = None
    declared_access_mode: str | None = None
    enum_entries: tuple[EnumEntrySpec, ...] = ()
    selected_features: tuple[str, ...] = ()
    category_display_path: tuple[str, ...] = ()
    xml_tag: str = ""
    access_mode: int | None = None
    runtime_interface_type: int | None = None

    @property
    def readable(self) -> bool:
        return self.access_mode in (AM_RO, AM_RW)

    @property
    def writable(self) -> bool:
        return self.access_mode in (AM_WO, AM_RW)

    @property
    def access_name(self) -> str:
        if self.access_mode is None:
            return "?"
        return ACCESS_MODE_NAMES.get(self.access_mode, str(self.access_mode))

    @property
    def effective_interface_type(self) -> str:
        return INTERFACE_ID_NAMES.get(
            self.runtime_interface_type, self.interface_type
        )


@dataclass(frozen=True)
class FeatureValue:
    """A feature value and the constraints returned by the MVS SDK."""

    value: object | None = None
    minimum: int | float | None = None
    maximum: int | float | None = None
    increment: int | None = None
    choices: tuple[EnumChoice, ...] = ()
    max_length: int | None = None


_STRUCTURAL_FEATURES = {
    "ADCBitDepth",
    "BinningHorizontal",
    "BinningMode",
    "BinningSelector",
    "BinningVertical",
    "DecimationHorizontal",
    "DecimationVertical",
    "Height",
    "ImageCompressionMode",
    "LinePitch",
    "OffsetX",
    "OffsetY",
    "PixelFormat",
    "RegionMode",
    "RegionSelector",
    "ReverseX",
    "ReverseY",
    "Rotation",
    "SensorHeight",
    "SensorShutterMode",
    "SensorWidth",
    "SuperBayerEnable",
    "SuperBinningEnable",
    "Width",
}

_DANGEROUS_FEATURES = {
    "AcquisitionStart",
    "AcquisitionStop",
    "ActivateShading",
    "AdjustFocalLength",
    "CounterReset",
    "DeviceReset",
    "DPCSave",
    "FileOperationExecute",
    "FocalLengthInitialize",
    "FPNCSave",
    "LineTriggerSoftware",
    "LUTSave",
    "PRNUCSave",
    "ReceiveQueueClear",
    "SequencerRestart",
    "SequencerSetLoad",
    "SequencerSetSave",
    "SoftwareSignalTrigger",
    "TimerReset",
    "TransferAbort",
    "TransferPause",
    "TransferResume",
    "TransferStart",
    "TransferStop",
    "TriggerSoftware",
    "UserDataSave",
    "UserSetDefault",
    "UserSetLoad",
    "UserSetSave",
}


def is_structural_feature(name: str) -> bool:
    """Return whether changing *name* changes image geometry/interpretation."""
    if name in _STRUCTURAL_FEATURES:
        return True
    return bool(
        re.match(
            r"^(?:SubRoi_(?:Width|Height|OffsetX|OffsetY)|"
            r"Region\d*(?:Width|Height|OffsetX|OffsetY))$",
            name,
        )
    )


def is_dangerous_feature(name: str) -> bool:
    """Return whether a feature can reset, persist, move, or disrupt a device."""
    return name in _DANGEROUS_FEATURES or name in {
        "HardwareTriggerActivation",
        "HardwareTriggerSource",
        "LineInverter",
        "LineMode",
        "LineSource",
        "LineTriggerArmDelay",
        "UserOutputValue",
    } or name.startswith("Strobe")


def feature_change_requires_refresh(spec: FeatureSpec) -> bool:
    """Return whether a successful write can change other nodes or their access."""
    if spec.effective_interface_type in {"Boolean", "Command", "Enumeration"}:
        return True
    if spec.selected_features:
        return True
    if is_structural_feature(spec.name) or is_dangerous_feature(spec.name):
        return True
    return spec.name.endswith(("Auto", "Enable", "Index", "Mode", "Selector", "Source"))


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _direct_children(
    element: ElementTree.Element, tag: str
) -> Iterator[ElementTree.Element]:
    for child in element:
        if _local_name(child.tag) == tag:
            yield child


def _child_text(
    element: ElementTree.Element, tag: str, default: str = ""
) -> str:
    child = next(_direct_children(element, tag), None)
    if child is None or child.text is None:
        return default
    return child.text.strip()


def _humanize(name: str) -> str:
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    text = text.replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()


def _normalise_xml_bytes(xml_data: bytes | bytearray | memoryview | str) -> bytes:
    if isinstance(xml_data, str):
        raw = xml_data.encode("utf-8")
    else:
        raw = bytes(xml_data)
    raw = raw.rstrip(b"\x00")
    if raw.startswith(b"PK\x03\x04"):
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            members = [
                name
                for name in archive.namelist()
                if name.lower().endswith(".xml")
            ]
            if not members:
                raise CameraFeatureError("GenICam ZIP contains no XML document")
            raw = archive.read(members[0])
    return raw


def _element_score(element: ElementTree.Element) -> int:
    """Prefer public interfaces over identically named internal XML entries."""
    tag = _local_name(element.tag)
    if tag in {
        "Category",
        "Integer",
        "Boolean",
        "Command",
        "Float",
        "String",
        "Enumeration",
    }:
        return 30
    if tag in {"StringReg", "IntConverter", "Converter"}:
        return 20
    if tag in _DECLARED_INTERFACE_TYPES:
        return 10
    return 0


def _parse_int(text: str) -> int | None:
    if not text:
        return None
    try:
        return int(text, 0)
    except ValueError:
        return None


def parse_genicam_xml(
    xml_data: bytes | bytearray | memoryview | str,
) -> list[FeatureSpec]:
    """Parse the ordered, ``Root``-reachable public GenICam feature tree.

    Static ``ImposedAccessMode`` values are retained as descriptive metadata,
    but ``FeatureSpec.access_mode`` remains ``None``.  Only the connected
    device can evaluate pIsImplemented/pIsAvailable/pIsLocked correctly.
    """
    raw = _normalise_xml_bytes(xml_data)
    try:
        root = ElementTree.fromstring(raw)
    except ElementTree.ParseError as exc:
        raise CameraFeatureError(f"Invalid GenICam XML: {exc}") from exc

    named: dict[str, ElementTree.Element] = {}
    scores: dict[str, int] = {}
    for element in root.iter():
        name = (element.get("Name") or "").strip()
        if not name:
            continue
        score = _element_score(element)
        if score > scores.get(name, -1):
            named[name] = element
            scores[name] = score

    root_category = named.get("Root")
    if root_category is None or _local_name(root_category.tag) != "Category":
        raise CameraFeatureError("GenICam XML has no Root category")

    features: list[FeatureSpec] = []
    seen_features: set[str] = set()
    active_categories: set[str] = set()

    def walk_category(
        category_element: ElementTree.Element,
        category_path: tuple[str, ...],
        display_path: tuple[str, ...],
    ) -> None:
        category_name = (category_element.get("Name") or "").strip()
        if category_name in active_categories:
            return
        active_categories.add(category_name)
        try:
            for reference in _direct_children(category_element, "pFeature"):
                feature_name = (reference.text or "").strip()
                if not feature_name:
                    continue
                element = named.get(feature_name)
                if element is None:
                    continue
                xml_tag = _local_name(element.tag)
                interface_type = _DECLARED_INTERFACE_TYPES.get(xml_tag, "")
                if interface_type == "Category":
                    category_label = _child_text(
                        element, "DisplayName", _humanize(feature_name)
                    )
                    walk_category(
                        element,
                        category_path + (feature_name,),
                        display_path + (category_label,),
                    )
                    continue
                if interface_type not in _EDITABLE_INTERFACE_TYPES:
                    continue
                if feature_name in seen_features:
                    continue
                seen_features.add(feature_name)

                entries: list[EnumEntrySpec] = []
                if interface_type == "Enumeration":
                    for entry in _direct_children(element, "EnumEntry"):
                        entry_name = (entry.get("Name") or "").strip()
                        if not entry_name:
                            continue
                        entries.append(
                            EnumEntrySpec(
                                name=entry_name,
                                value=_parse_int(_child_text(entry, "Value")),
                                display_name=_child_text(
                                    entry, "DisplayName", _humanize(entry_name)
                                ),
                                description=_child_text(entry, "Description"),
                            )
                        )

                streamable_text = _child_text(element, "Streamable").lower()
                streamable: bool | None
                if streamable_text in {"yes", "true", "1"}:
                    streamable = True
                elif streamable_text in {"no", "false", "0"}:
                    streamable = False
                else:
                    streamable = None

                declared_access = _child_text(element, "ImposedAccessMode")
                if not declared_access:
                    declared_access = _child_text(element, "AccessMode")
                features.append(
                    FeatureSpec(
                        name=feature_name,
                        category=category_path[-1] if category_path else "Root",
                        category_path=category_path,
                        interface_type=interface_type,
                        display_name=_child_text(
                            element, "DisplayName", _humanize(feature_name)
                        ),
                        description=_child_text(element, "Description"),
                        tooltip=_child_text(element, "ToolTip"),
                        visibility=_child_text(
                            element, "Visibility", "Beginner"
                        ),
                        unit=_child_text(element, "Unit"),
                        representation=_child_text(element, "Representation"),
                        streamable=streamable,
                        declared_access_mode=declared_access or None,
                        enum_entries=tuple(entries),
                        selected_features=tuple(
                            (selected.text or "").strip()
                            for selected in _direct_children(element, "pSelected")
                            if (selected.text or "").strip()
                        ),
                        category_display_path=display_path,
                        xml_tag=xml_tag,
                    )
                )
        finally:
            active_categories.remove(category_name)

    walk_category(root_category, (), ())
    return features


def _decode_c_string(value: object) -> str:
    if isinstance(value, bytes):
        raw = value
    else:
        try:
            raw = memoryview(value).tobytes()
        except TypeError:
            raw = bytes(value)  # type: ignore[arg-type]
    raw = raw.split(b"\0", 1)[0]
    for encoding in ("utf-8", "gbk", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


class CameraFeatureAccessor:
    """Runtime feature access for a controller or raw ``MvCamera`` instance."""

    def __init__(self, controller: object, mvs_module: object | None = None) -> None:
        self.controller = controller
        self._mvs_module = mvs_module
        self._catalog: list[FeatureSpec] | None = None

    @property
    def cam(self) -> object:
        return getattr(self.controller, "cam", self.controller)

    @property
    def connected(self) -> bool:
        if hasattr(self.controller, "device_open"):
            return bool(getattr(self.controller, "device_open"))
        handle = getattr(self.cam, "handle", None)
        return handle is not None and bool(handle)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        lock = getattr(self.controller, "lock", None)
        if lock is None:
            yield
        else:
            with lock:
                yield

    def _sdk(self) -> object:
        if self._mvs_module is not None:
            return self._mvs_module
        module_name = self.cam.__class__.__module__
        candidates = [module_name, "MvCameraControl_class"]
        last_error: Exception | None = None
        for candidate in candidates:
            try:
                module = importlib.import_module(candidate)
            except (ImportError, OSError) as exc:
                last_error = exc
                continue
            if hasattr(module, "MVCC_INTVALUE_EX"):
                self._mvs_module = module
                return module
        raise CameraFeatureError(
            "Unable to load the MVS Python binding for feature access"
        ) from last_error

    @staticmethod
    def _check(operation: str, result: object) -> None:
        try:
            code = int(result)
        except (TypeError, ValueError):
            code = int(getattr(result, "value"))
        if code != 0:
            raise CameraFeatureError(
                f"{operation} failed: 0x{code & 0xFFFFFFFF:08x}"
            )

    def _require_connected(self) -> None:
        if not self.connected:
            raise CameraFeatureError("Camera is not connected")

    def get_genicam_xml(self) -> bytes:
        """Download GenICam XML using the documented two-pass SDK API."""
        self._require_connected()
        sdk = self._sdk()
        data_length = c_uint(0)
        with self._locked():
            first = self.cam.MV_XML_GetGenICamXML(  # type: ignore[attr-defined]
                None, 0, data_length
            )
        insufficient = int(getattr(sdk, "MV_E_NOENOUGH_BUF", 0x8000000A))
        if int(first) not in (0, insufficient):
            self._check("MV_XML_GetGenICamXML(size)", first)
        if not 0 < data_length.value <= 128 * 1024 * 1024:
            raise CameraFeatureError(
                f"Invalid GenICam XML size returned by SDK: {data_length.value}"
            )

        buffer = (c_ubyte * data_length.value)()
        with self._locked():
            result = self.cam.MV_XML_GetGenICamXML(  # type: ignore[attr-defined]
                buffer, len(buffer), data_length
            )
        self._check("MV_XML_GetGenICamXML(data)", result)
        return bytes(buffer[: data_length.value]).rstrip(b"\0")

    def catalog(self, *, refresh_xml: bool = False) -> list[FeatureSpec]:
        if self._catalog is None or refresh_xml:
            self._catalog = parse_genicam_xml(self.get_genicam_xml())
        return list(self._catalog)

    def get_access_mode(self, name: str) -> int:
        self._require_connected()
        sdk = self._sdk()
        value_type = getattr(sdk, "MV_XML_AccessMode", c_int)
        value = value_type()
        with self._locked():
            result = self.cam.MV_XML_GetNodeAccessMode(  # type: ignore[attr-defined]
                name, value
            )
        self._check(f"Get access mode {name}", result)
        return int(value.value)

    def get_interface_type(self, name: str) -> int:
        self._require_connected()
        sdk = self._sdk()
        value_type = getattr(sdk, "MV_XML_InterfaceType", c_int)
        value = value_type()
        with self._locked():
            result = self.cam.MV_XML_GetNodeInterfaceType(  # type: ignore[attr-defined]
                name, value
            )
        self._check(f"Get interface type {name}", result)
        return int(value.value)

    def discover_features(self, *, refresh_xml: bool = False) -> list[FeatureSpec]:
        """Return XML features decorated with their current runtime state."""
        discovered: list[FeatureSpec] = []
        for spec in self.catalog(refresh_xml=refresh_xml):
            try:
                access_mode = self.get_access_mode(spec.name)
            except CameraFeatureError:
                access_mode = AM_NI
            runtime_type: int | None = None
            if access_mode not in (AM_NI, AM_NA, AM_UNDEFINED):
                try:
                    runtime_type = self.get_interface_type(spec.name)
                except CameraFeatureError:
                    runtime_type = None
            discovered.append(
                replace(
                    spec,
                    access_mode=access_mode,
                    runtime_interface_type=runtime_type,
                )
            )
        return discovered

    def read(self, spec: FeatureSpec) -> FeatureValue:
        if not spec.readable:
            return FeatureValue()
        sdk = self._sdk()
        interface_type = spec.effective_interface_type

        if interface_type == "Integer":
            value = sdk.MVCC_INTVALUE_EX()
            with self._locked():
                result = self.cam.MV_CC_GetIntValueEx(  # type: ignore[attr-defined]
                    spec.name, value
                )
            self._check(f"Read {spec.name}", result)
            return FeatureValue(
                value=int(value.nCurValue),
                minimum=int(value.nMin),
                maximum=int(value.nMax),
                increment=max(1, int(value.nInc)),
            )

        if interface_type == "Float":
            value = sdk.MVCC_FLOATVALUE()
            with self._locked():
                result = self.cam.MV_CC_GetFloatValue(  # type: ignore[attr-defined]
                    spec.name, value
                )
            self._check(f"Read {spec.name}", result)
            return FeatureValue(
                value=float(value.fCurValue),
                minimum=float(value.fMin),
                maximum=float(value.fMax),
            )

        if interface_type == "Enumeration":
            enum_type = getattr(sdk, "MVCC_ENUMVALUE_EX", sdk.MVCC_ENUMVALUE)
            value = enum_type()
            getter = getattr(
                self.cam,
                "MV_CC_GetEnumValueEx",
                getattr(self.cam, "MV_CC_GetEnumValue"),
            )
            with self._locked():
                result = getter(spec.name, value)
            self._check(f"Read {spec.name}", result)
            supported_count = min(
                int(value.nSupportedNum), len(value.nSupportValue)
            )
            xml_by_value = {
                entry.value: entry
                for entry in spec.enum_entries
                if entry.value is not None
            }
            choices: list[EnumChoice] = []
            entry_getter = getattr(self.cam, "MV_CC_GetEnumEntrySymbolic")
            for index in range(supported_count):
                numeric = int(value.nSupportValue[index])
                symbolic = str(numeric)
                entry_value = sdk.MVCC_ENUMENTRY()
                entry_value.nValue = numeric
                with self._locked():
                    entry_result = entry_getter(spec.name, entry_value)
                if int(entry_result) == 0:
                    symbolic = _decode_c_string(entry_value.chSymbolic)
                elif numeric in xml_by_value:
                    symbolic = xml_by_value[numeric].name
                xml_entry = xml_by_value.get(numeric)
                choices.append(
                    EnumChoice(
                        value=numeric,
                        symbolic=symbolic,
                        display_name=(
                            xml_entry.display_name
                            if xml_entry is not None
                            else symbolic
                        ),
                    )
                )
            return FeatureValue(value=int(value.nCurValue), choices=tuple(choices))

        if interface_type == "Boolean":
            value = c_bool()
            with self._locked():
                result = self.cam.MV_CC_GetBoolValue(  # type: ignore[attr-defined]
                    spec.name, value
                )
            self._check(f"Read {spec.name}", result)
            return FeatureValue(value=bool(value.value))

        if interface_type == "String":
            value = sdk.MVCC_STRINGVALUE()
            with self._locked():
                result = self.cam.MV_CC_GetStringValue(  # type: ignore[attr-defined]
                    spec.name, value
                )
            self._check(f"Read {spec.name}", result)
            return FeatureValue(
                value=_decode_c_string(value.chCurValue),
                max_length=int(value.nMaxLength),
            )

        if interface_type == "Command":
            return FeatureValue()
        raise CameraFeatureError(
            f"Unsupported interface type for {spec.name}: {interface_type}"
        )

    def write(self, spec: FeatureSpec, value: object | None = None) -> None:
        """Write one feature, rechecking its dynamic access mode first."""
        access_mode = self.get_access_mode(spec.name)
        if access_mode not in (AM_WO, AM_RW):
            raise CameraFeatureError(
                f"{spec.name} is not writable (access "
                f"{ACCESS_MODE_NAMES.get(access_mode, access_mode)})"
            )
        interface_type = spec.effective_interface_type
        with self._locked():
            if interface_type == "Integer":
                result = self.cam.MV_CC_SetIntValueEx(spec.name, int(value))
            elif interface_type == "Float":
                result = self.cam.MV_CC_SetFloatValue(spec.name, float(value))
            elif interface_type == "Enumeration":
                if isinstance(value, EnumChoice):
                    value = value.value
                if isinstance(value, str):
                    result = self.cam.MV_CC_SetEnumValueByString(spec.name, value)
                else:
                    result = self.cam.MV_CC_SetEnumValue(spec.name, int(value))
            elif interface_type == "Boolean":
                result = self.cam.MV_CC_SetBoolValue(  # type: ignore[attr-defined]
                    spec.name, bool(value)
                )
            elif interface_type == "String":
                try:
                    str(value).encode("ascii")
                except UnicodeEncodeError as exc:
                    raise CameraFeatureError(
                        "The supplied MVS Python binding accepts ASCII strings only"
                    ) from exc
                result = self.cam.MV_CC_SetStringValue(  # type: ignore[attr-defined]
                    spec.name, str(value)
                )
            elif interface_type == "Command":
                result = self.cam.MV_CC_SetCommandValue(  # type: ignore[attr-defined]
                    spec.name
                )
            else:
                raise CameraFeatureError(
                    f"Unsupported interface type for {spec.name}: {interface_type}"
                )
        self._check(f"Write {spec.name}", result)


# PyQt is optional for parser/accessor users.  In the configured revo3_ros
# environment this import succeeds and the full panel class below is exposed.
try:
    from PyQt5.QtCore import QSignalBlocker, Qt, QTimer, pyqtSignal
    from PyQt5.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )

    _QT_IMPORT_ERROR: Exception | None = None
except (ImportError, OSError) as exc:  # pragma: no cover - depends on host
    _QT_IMPORT_ERROR = exc


_VISIBILITY_RANK = {
    "Beginner": 0,
    "Expert": 1,
    "Guru": 2,
    "Invisible": 3,
}

_CATEGORY_LABELS_ZH = {
    "DeviceControl": "设备控制",
    "ImageFormatControl": "图像格式",
    "AcquisitionControl": "采集与曝光",
    "SoftwareSignalControl": "软件信号",
    "AnalogControl": "模拟与图像质量",
    "ColorTransformationControl": "颜色变换",
    "SuperPaletteControl": "调色板",
    "LUTControl": "LUT 控制",
    "ShadingCorrection": "阴影校正",
    "DigitalIOControl": "数字 IO",
    "CounterAndTimerControl": "计数器与定时器",
    "SerialPortControl": "串口控制",
    "FileAccessControl": "文件访问",
    "SequencerControl": "序列器",
    "EventControl": "事件控制",
    "ChunkDataControl": "Chunk 数据",
    "TransportLayerControl": "传输层",
    "StreamControl": "数据流",
    "TransferControl": "传输控制",
    "OpticControl": "光学控制",
    "UserSetControl": "用户参数组",
}

_FEATURE_LABELS_ZH = {
    "ExposureAuto": "自动曝光",
    "ExposureTime": "曝光时间",
    "AutoExposureTimeLowerLimit": "自动曝光下限",
    "AutoExposureTimeUpperLimit": "自动曝光上限",
    "GainAuto": "自动增益",
    "Gain": "增益",
    "AutoGainLowerLimit": "自动增益下限",
    "AutoGainUpperLimit": "自动增益上限",
    "AcquisitionFrameRateEnable": "帧率控制使能",
    "AcquisitionFrameRate": "采集帧率",
    "GammaEnable": "Gamma 使能",
    "Gamma": "Gamma",
    "SharpnessEnable": "锐化使能",
    "Sharpness": "锐度",
    "BlackLevelEnable": "黑电平使能",
    "BlackLevel": "黑电平",
    "Width": "图像宽度",
    "Height": "图像高度",
    "OffsetX": "水平偏移",
    "OffsetY": "垂直偏移",
    "PixelFormat": "像素格式",
    "ReverseX": "水平翻转",
    "ReverseY": "垂直翻转",
    "TriggerMode": "触发模式",
    "TriggerSource": "触发源",
    "TriggerDelay": "触发延迟",
    "DeviceTemperature": "设备温度",
}


if _QT_IMPORT_ERROR is None:

    class _FeatureRow(QWidget):
        def __init__(
            self,
            spec: FeatureSpec,
            editor: QWidget,
            search_text: str,
            parent: QWidget | None = None,
        ) -> None:
            super().__init__(parent)
            self.spec = spec
            self.search_text = search_text.casefold()
            layout = QHBoxLayout(self)
            layout.setContentsMargins(4, 2, 4, 2)
            label_text = _FEATURE_LABELS_ZH.get(spec.name, spec.display_name)
            if label_text != spec.display_name and spec.display_name:
                label_text += f" / {spec.display_name}"
            if is_structural_feature(spec.name):
                label_text = "◆ " + label_text
            if (
                spec.effective_interface_type == "Command"
                or is_dangerous_feature(spec.name)
            ):
                label_text = "⚠ " + label_text
            label = QLabel(label_text)
            label.setMinimumWidth(230)
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            details = spec.tooltip or spec.description
            label.setToolTip(
                f"{spec.name}\n{details}" if details else spec.name
            )
            layout.addWidget(label)
            layout.addWidget(editor, 1)
            access = QLabel(spec.access_name)
            access.setMinimumWidth(28)
            access.setToolTip("当前 GenICam 节点访问权限")
            layout.addWidget(access)

    class CameraFeaturePanel(QWidget):
        """Searchable, category-based editor for all runtime-supported nodes."""

        status_changed = pyqtSignal(str)
        change_failed = pyqtSignal(object, str)
        feature_changed = pyqtSignal(object)

        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self.controller: object | None = None
            self.accessor: CameraFeatureAccessor | None = None
            self.before_change: Callable[[FeatureSpec], bool] | None = None
            self.after_change: Callable[[FeatureSpec], None] | None = None
            self._specs: list[FeatureSpec] = []
            self._rows: list[_FeatureRow] = []
            self._tab_rows: list[list[_FeatureRow]] = []
            self._refresh_pending = False

            self.search = QLineEdit()
            self.search.setPlaceholderText("搜索参数名称、节点名或分类…")
            self.visibility = QComboBox()
            self.visibility.addItem("基础", 0)
            self.visibility.addItem("专家", 1)
            self.visibility.addItem("高级", 2)
            self.visibility.addItem("全部（含隐藏）", 3)
            self.visibility.setCurrentIndex(1)
            self.show_read_only = QCheckBox("显示只读")
            self.refresh_button = QPushButton("刷新参数")
            self.help_label = QLabel(
                "按相机当前 GenICam 权限显示参数；若 ROI/像素格式等项目未出现，"
                "请先暂停取流。◆ 表示会改变成像几何，⚠ 表示执行前需要确认。"
            )
            self.help_label.setWordWrap(True)
            self.help_label.setStyleSheet(
                "padding: 6px; background: #eef5ff; border: 1px solid #b9d1ef;"
            )
            self.status_label = QLabel("相机未连接")
            self.status_label.setWordWrap(True)
            self.tabs = QTabWidget()
            self.tabs.setDocumentMode(True)

            toolbar = QHBoxLayout()
            toolbar.addWidget(self.search, 1)
            toolbar.addWidget(self.visibility)
            toolbar.addWidget(self.show_read_only)
            toolbar.addWidget(self.refresh_button)
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addLayout(toolbar)
            layout.addWidget(self.help_label)
            layout.addWidget(self.tabs, 1)
            layout.addWidget(self.status_label)

            self.search.textChanged.connect(self._apply_search_filter)
            self.visibility.currentIndexChanged.connect(self._rebuild_rows)
            self.show_read_only.toggled.connect(self._rebuild_rows)
            self.refresh_button.clicked.connect(self.refresh_features)
            self._set_bound_state(False)

        def bind_controller(self, controller: object | None) -> None:
            """Bind a controller exposing cam, lock, and device_open attributes."""
            self.controller = controller
            self.accessor = (
                CameraFeatureAccessor(controller) if controller is not None else None
            )
            self._specs = []
            self._clear_tabs()
            connected = bool(self.accessor and self.accessor.connected)
            self._set_bound_state(connected)
            if connected:
                self.refresh_features()
            else:
                self._set_status("相机未连接")

        def refresh_features(self) -> None:
            """Re-query dynamic access/type state and rebuild visible editors."""
            if self.accessor is None or not self.accessor.connected:
                self._specs = []
                self._clear_tabs()
                self._set_bound_state(False)
                self._set_status("相机未连接")
                return
            self._set_bound_state(True)
            self.refresh_button.setEnabled(False)
            self._set_status("正在读取相机参数…")
            try:
                self._specs = self.accessor.discover_features()
                self._rebuild_rows()
                writable = sum(spec.writable for spec in self._specs)
                readable = sum(spec.readable for spec in self._specs)
                self._set_status(
                    f"检测到 {writable} 个可调参数，{readable} 个可读参数"
                )
            except Exception as exc:
                self._specs = []
                self._clear_tabs()
                self._set_status(f"读取参数失败：{exc}")
            finally:
                self.refresh_button.setEnabled(True)

        def _set_bound_state(self, connected: bool) -> None:
            self.search.setEnabled(connected)
            self.visibility.setEnabled(connected)
            self.show_read_only.setEnabled(connected)
            self.refresh_button.setEnabled(connected)
            self.tabs.setEnabled(connected)

        def _set_status(self, message: str) -> None:
            self.status_label.setText(message)
            self.status_changed.emit(message)

        def _clear_tabs(self) -> None:
            self._rows = []
            self._tab_rows = []
            while self.tabs.count():
                page = self.tabs.widget(0)
                self.tabs.removeTab(0)
                page.deleteLater()

        def _visibility_limit(self) -> int:
            value = self.visibility.currentData()
            return int(value) if value is not None else 0

        def _eligible(self, spec: FeatureSpec) -> bool:
            if spec.access_mode in (AM_NI, AM_NA, AM_UNDEFINED, AM_CYCLE_DETECT):
                return False
            if not spec.writable and not (
                self.show_read_only.isChecked() and spec.readable
            ):
                return False
            return _VISIBILITY_RANK.get(spec.visibility, 0) <= self._visibility_limit()

        def _rebuild_rows(self, *_args: object) -> None:
            current_category: tuple[str, ...] | None = None
            scroll_positions: dict[tuple[str, ...], int] = {}
            for tab_index in range(self.tabs.count()):
                scroll = self.tabs.widget(tab_index)
                category_data = scroll.property("categoryPath")
                category_path = (
                    tuple(category_data) if category_data is not None else ()
                )
                if category_path:
                    scroll_positions[category_path] = (
                        scroll.verticalScrollBar().value()
                    )
                    if tab_index == self.tabs.currentIndex():
                        current_category = category_path
            self._clear_tabs()
            if self.accessor is None:
                return
            pages: dict[
                tuple[str, ...], tuple[QWidget, QVBoxLayout, list[_FeatureRow]]
            ] = {}
            scrolls: dict[tuple[str, ...], QScrollArea] = {}
            for spec in self._specs:
                if not self._eligible(spec):
                    continue
                try:
                    feature_value = self.accessor.read(spec)
                    editor = self._make_editor(spec, feature_value)
                except Exception as exc:
                    editor = QLabel(f"读取失败：{exc}")
                    editor.setStyleSheet("color: #b33;")

                category_path = spec.category_path or ("Root",)
                if category_path not in pages:
                    page = QWidget()
                    page_layout = QVBoxLayout(page)
                    page_layout.setContentsMargins(6, 6, 6, 6)
                    rows: list[_FeatureRow] = []
                    pages[category_path] = (page, page_layout, rows)
                    category_name = category_path[-1]
                    category_label = _CATEGORY_LABELS_ZH.get(
                        category_name,
                        (
                            spec.category_display_path[-1]
                            if spec.category_display_path
                            else _humanize(category_name)
                        ),
                    )
                    scroll = QScrollArea()
                    scroll.setWidgetResizable(True)
                    scroll.setWidget(page)
                    scroll.setProperty("categoryPath", category_path)
                    scrolls[category_path] = scroll
                    self.tabs.addTab(scroll, category_label)

                page, page_layout, rows = pages[category_path]
                del page
                text = " ".join(
                    (
                        spec.name,
                        spec.display_name,
                        spec.description,
                        spec.tooltip,
                        " ".join(spec.category_path),
                    )
                )
                row = _FeatureRow(spec, editor, text)
                page_layout.addWidget(row)
                rows.append(row)
                self._rows.append(row)

            self._tab_rows = []
            for page, page_layout, rows in pages.values():
                del page
                page_layout.addStretch(1)
                self._tab_rows.append(rows)
            self._apply_search_filter()
            for category_path, scroll in scrolls.items():
                scroll.verticalScrollBar().setValue(
                    scroll_positions.get(category_path, 0)
                )
                if category_path == current_category:
                    index = self.tabs.indexOf(scroll)
                    if index >= 0 and self.tabs.isTabEnabled(index):
                        self.tabs.setCurrentIndex(index)

        def _make_editor(
            self, spec: FeatureSpec, feature_value: FeatureValue
        ) -> QWidget:
            interface_type = spec.effective_interface_type
            if not spec.writable:
                value = QLabel(self._format_value(spec, feature_value))
                value.setTextInteractionFlags(Qt.TextSelectableByMouse)
                return value

            if interface_type == "Integer":
                container = QWidget()
                layout = QHBoxLayout(container)
                layout.setContentsMargins(0, 0, 0, 0)
                edit = QLineEdit(
                    "" if feature_value.value is None else str(feature_value.value)
                )
                bounds = self._bounds_text(feature_value)
                edit.setPlaceholderText(bounds)
                edit.setToolTip(bounds)
                button = QPushButton("应用")

                def apply_integer() -> None:
                    try:
                        value = int(edit.text().strip(), 0)
                    except ValueError:
                        self._set_status(f"{spec.name} 需要整数")
                        return
                    if not self._validate_numeric(spec, value, feature_value):
                        return
                    self._request_change(spec, value)

                button.clicked.connect(apply_integer)
                edit.returnPressed.connect(apply_integer)
                layout.addWidget(edit, 1)
                layout.addWidget(button)
                return container

            if interface_type == "Float":
                container = QWidget()
                layout = QHBoxLayout(container)
                layout.setContentsMargins(0, 0, 0, 0)
                spin = QDoubleSpinBox()
                spin.setDecimals(6)
                spin.setKeyboardTracking(False)
                minimum = float(feature_value.minimum or 0.0)
                maximum = float(feature_value.maximum or 0.0)
                if maximum < minimum or maximum == minimum:
                    minimum, maximum = -3.402823e38, 3.402823e38
                spin.setRange(minimum, maximum)
                if feature_value.value is not None:
                    spin.setValue(float(feature_value.value))
                span = maximum - minimum
                if 0 < span < 1e100:
                    spin.setSingleStep(max(span / 1000.0, 1e-6))
                if spec.unit:
                    spin.setSuffix(f" {spec.unit}")
                button = QPushButton("应用")
                button.clicked.connect(
                    lambda _checked=False: self._request_change(spec, spin.value())
                )
                layout.addWidget(spin, 1)
                layout.addWidget(button)
                return container

            if interface_type == "Enumeration":
                combo = QComboBox()
                current_index = -1
                for index, choice in enumerate(feature_value.choices):
                    label = choice.display_name or choice.symbolic
                    if label != choice.symbolic:
                        label += f" / {choice.symbolic}"
                    combo.addItem(label, choice.value)
                    if choice.value == feature_value.value:
                        current_index = index
                if current_index >= 0:
                    with QSignalBlocker(combo):
                        combo.setCurrentIndex(current_index)
                combo.activated.connect(
                    lambda _index: self._request_change(spec, combo.currentData())
                )
                return combo

            if interface_type == "Boolean":
                check = QCheckBox("启用")
                check.setChecked(bool(feature_value.value))
                check.clicked.connect(
                    lambda checked: self._request_change(spec, bool(checked))
                )
                return check

            if interface_type == "String":
                container = QWidget()
                layout = QHBoxLayout(container)
                layout.setContentsMargins(0, 0, 0, 0)
                edit = QLineEdit(str(feature_value.value or ""))
                if feature_value.max_length and feature_value.max_length > 0:
                    edit.setMaxLength(feature_value.max_length)
                button = QPushButton("应用")
                button.clicked.connect(
                    lambda _checked=False: self._request_change(spec, edit.text())
                )
                edit.returnPressed.connect(
                    lambda: self._request_change(spec, edit.text())
                )
                layout.addWidget(edit, 1)
                layout.addWidget(button)
                return container

            if interface_type == "Command":
                button = QPushButton("执行命令")
                if is_dangerous_feature(spec.name):
                    button.setStyleSheet("color: #a33; font-weight: bold;")
                button.clicked.connect(
                    lambda _checked=False: self._request_change(spec, None)
                )
                return button

            return QLabel(f"不支持的节点类型：{interface_type}")

        @staticmethod
        def _bounds_text(value: FeatureValue) -> str:
            parts: list[str] = []
            if value.minimum is not None and value.maximum is not None:
                parts.append(f"范围 {value.minimum}…{value.maximum}")
            if value.increment is not None:
                parts.append(f"步进 {value.increment}")
            return "，".join(parts)

        def _validate_numeric(
            self, spec: FeatureSpec, value: int | float, limits: FeatureValue
        ) -> bool:
            if limits.minimum is not None and value < limits.minimum:
                self._set_status(f"{spec.name} 小于最小值 {limits.minimum}")
                return False
            if limits.maximum is not None and value > limits.maximum:
                self._set_status(f"{spec.name} 大于最大值 {limits.maximum}")
                return False
            if (
                isinstance(value, int)
                and limits.increment
                and limits.minimum is not None
                and (value - int(limits.minimum)) % limits.increment
            ):
                self._set_status(
                    f"{spec.name} 必须满足最小值 {limits.minimum}、"
                    f"步进 {limits.increment}"
                )
                return False
            return True

        @staticmethod
        def _format_value(spec: FeatureSpec, value: FeatureValue) -> str:
            if spec.effective_interface_type == "Enumeration":
                for choice in value.choices:
                    if choice.value == value.value:
                        return choice.display_name or choice.symbolic
            if value.value is None:
                return "—"
            suffix = f" {spec.unit}" if spec.unit else ""
            return f"{value.value}{suffix}"

        def _request_change(self, spec: FeatureSpec, value: object | None) -> None:
            if self.accessor is None or not self.accessor.connected:
                self._set_status("相机未连接")
                return

            if self.before_change is None and (
                spec.effective_interface_type == "Command"
                or is_dangerous_feature(spec.name)
            ):
                answer = QMessageBox.warning(
                    self,
                    "确认相机命令",
                    f"{spec.name} 可能中断采集、移动机构、复位设备或写入持久存储。\n"
                    "确定执行吗？",
                    QMessageBox.Yes | QMessageBox.Cancel,
                    QMessageBox.Cancel,
                )
                if answer != QMessageBox.Yes:
                    QTimer.singleShot(0, self.refresh_features)
                    return

            approved = True
            if self.before_change is not None:
                try:
                    approved = bool(self.before_change(spec))
                except Exception as exc:
                    self.change_failed.emit(spec, str(exc))
                    self._set_status(f"准备修改 {spec.name} 失败：{exc}")
                    return
            if not approved:
                self._set_status(f"已取消修改 {spec.name}")
                QTimer.singleShot(0, self.refresh_features)
                return

            succeeded = False
            try:
                self.accessor.write(spec, value)
                succeeded = True
                self.feature_changed.emit(spec)
                self._set_status(f"已更新 {spec.name}")
            except Exception as exc:
                message = str(exc)
                self.change_failed.emit(spec, message)
                self._set_status(f"设置 {spec.name} 失败：{message}")
            finally:
                if self.after_change is not None:
                    try:
                        self.after_change(spec)
                    except Exception as exc:
                        self.change_failed.emit(spec, str(exc))
                        self._set_status(f"修改后恢复失败：{exc}")

            # Selectors and automatic modes can alter many other nodes.  A
            # debounced full runtime refresh is more reliable than guessing
            # every pSelected dependency, and also verifies successful writes.
            # Hosts own refresh ordering after successful changes.  A failed
            # write always refreshes so an optimistic checkbox/combo change is
            # rolled back to the value actually held by the camera.
            should_refresh = not succeeded or (
                self.after_change is None
                and feature_change_requires_refresh(spec)
            )
            if should_refresh and not self._refresh_pending:
                self._refresh_pending = True

                def refresh_once() -> None:
                    self._refresh_pending = False
                    self.refresh_features()

                QTimer.singleShot(0, refresh_once)

        def _apply_search_filter(self, *_args: object) -> None:
            query = self.search.text().strip().casefold()
            for tab_index, rows in enumerate(self._tab_rows):
                any_visible = False
                for row in rows:
                    visible = not query or query in row.search_text
                    row.setVisible(visible)
                    any_visible = any_visible or visible
                self.tabs.setTabEnabled(tab_index, any_visible)
            if self.tabs.count() and not self.tabs.isTabEnabled(
                self.tabs.currentIndex()
            ):
                for index in range(self.tabs.count()):
                    if self.tabs.isTabEnabled(index):
                        self.tabs.setCurrentIndex(index)
                        break


else:

    class CameraFeaturePanel:  # pragma: no cover - exercised without Qt only
        """Placeholder that explains how to make the optional Qt UI available."""

        def __init__(self, parent: object | None = None) -> None:
            del parent
            raise RuntimeError(
                "CameraFeaturePanel requires PyQt5; activate the revo3_ros "
                f"environment first ({_QT_IMPORT_ERROR})"
            )


__all__ = [
    "ACCESS_MODE_NAMES",
    "AM_NA",
    "AM_NI",
    "AM_RO",
    "AM_RW",
    "AM_WO",
    "CameraFeatureAccessor",
    "CameraFeatureError",
    "CameraFeaturePanel",
    "EnumChoice",
    "EnumEntrySpec",
    "FeatureSpec",
    "FeatureValue",
    "feature_change_requires_refresh",
    "is_dangerous_feature",
    "is_structural_feature",
    "parse_genicam_xml",
]
