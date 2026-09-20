from __future__ import annotations

import os
import threading
import unittest
from ctypes import Structure, c_char, c_double, c_int, c_uint, c_uint64
from unittest.mock import MagicMock

# QApplication must choose a headless platform before PyQt (which is imported
# optionally by camera_features) is loaded.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from camera_features import (
    AM_RO,
    AM_RW,
    CameraFeatureAccessor,
    CameraFeatureError,
    CameraFeaturePanel,
    EnumChoice,
    EnumEntrySpec,
    FeatureSpec,
    is_dangerous_feature,
    is_structural_feature,
    parse_genicam_xml,
)

try:
    from PyQt5.QtWidgets import QApplication
except (ImportError, OSError):  # pragma: no cover - depends on test host
    QApplication = None  # type: ignore[assignment,misc]


SYNTHETIC_GENICAM_XML = b"""\
<?xml version="1.0" encoding="UTF-8"?>
<RegisterDescription xmlns="http://www.genicam.org/GenApi/Version_1_1">
  <Category Name="Root" NameSpace="Standard">
    <pFeature>AnalogControl</pFeature>
    <pFeature>ImageFormatControl</pFeature>
  </Category>

  <Category Name="AnalogControl" NameSpace="Standard">
    <DisplayName>Analog Control</DisplayName>
    <pFeature>ExposureTime</pFeature>
    <pFeature>Gain</pFeature>
    <pFeature>AdvancedAnalogControl</pFeature>
    <pFeature>MissingFeature</pFeature>
  </Category>
  <Category Name="AdvancedAnalogControl" NameSpace="Custom">
    <DisplayName>Advanced Analog Control</DisplayName>
    <Visibility>Expert</Visibility>
    <pFeature>NoiseReduction</pFeature>
  </Category>
  <Category Name="ImageFormatControl" NameSpace="Standard">
    <DisplayName>Image Format Control</DisplayName>
    <pFeature>Width</pFeature>
    <pFeature>PixelFormat</pFeature>
    <pFeature>RawLutData</pFeature>
  </Category>

  <Group Comment="Public feature definitions">
    <Float Name="ExposureTime" NameSpace="Standard">
      <DisplayName>Exposure Time</DisplayName>
      <Description>Controls how long the sensor integrates light.</Description>
      <ToolTip>Exposure duration in microseconds.</ToolTip>
      <Visibility>Beginner</Visibility>
      <ImposedAccessMode>RW</ImposedAccessMode>
      <Streamable>Yes</Streamable>
      <Unit>us</Unit>
      <Representation>Logarithmic</Representation>
      <pValue>ExposureTimeRegister</pValue>
    </Float>
    <Float Name="Gain" NameSpace="Standard">
      <DisplayName>Sensor Gain</DisplayName>
      <Visibility>Beginner</Visibility>
      <Unit>dB</Unit>
      <pValue>GainRegister</pValue>
    </Float>
    <Integer Name="NoiseReduction" NameSpace="Custom">
      <DisplayName>Noise Reduction</DisplayName>
      <Visibility>Guru</Visibility>
      <AccessMode>RO</AccessMode>
      <pValue>NoiseReductionRegister</pValue>
    </Integer>
    <Integer Name="Width" NameSpace="Standard">
      <DisplayName>Width</DisplayName>
      <Visibility>Beginner</Visibility>
      <pIsLocked>TransportLayerParametersLocked</pIsLocked>
      <pValue>WidthRegister</pValue>
    </Integer>
    <Enumeration Name="PixelFormat" NameSpace="Standard">
      <DisplayName>Pixel Format</DisplayName>
      <Visibility>Expert</Visibility>
      <EnumEntry Name="Mono8" NameSpace="Standard">
        <DisplayName>Mono 8</DisplayName>
        <Value>0x01080001</Value>
      </EnumEntry>
      <EnumEntry Name="Mono12" NameSpace="Standard">
        <DisplayName>Mono 12</DisplayName>
        <Value>0x01100005</Value>
      </EnumEntry>
      <pValue>PixelFormatRegister</pValue>
    </Enumeration>
    <Register Name="RawLutData" NameSpace="Custom">
      <AccessMode>RW</AccessMode>
    </Register>
  </Group>

  <!-- These names deliberately collide with public features.  A parser must
       resolve Category.pFeature references to public interfaces, not to the
       later internal selector entries or expression variables. -->
  <Enumeration Name="SequencerFeatureSelector" NameSpace="Custom">
    <EnumEntry Name="Gain" NameSpace="Custom">
      <Value>3</Value>
    </EnumEntry>
    <EnumEntry Name="Width" NameSpace="Custom">
      <Value>4</Value>
    </EnumEntry>
  </Enumeration>
  <IntSwissKnife Name="ExposureLockExpression">
    <pVariable Name="ExposureTime">ExposureTimeRegister</pVariable>
    <Formula>ExposureTime</Formula>
  </IntSwissKnife>

  <!-- Existing but unreachable/internal nodes must not become GUI controls. -->
  <Category Name="OrphanCategory">
    <pFeature>OrphanValue</pFeature>
  </Category>
  <Float Name="OrphanValue"><Value>1.0</Value></Float>
  <Integer Name="UnreferencedRegister"><Value>7</Value></Integer>
</RegisterDescription>
"""


class _FakeIntValue(Structure):
    _fields_ = [
        ("nCurValue", c_uint64),
        ("nMax", c_uint64),
        ("nMin", c_uint64),
        ("nInc", c_uint64),
    ]


class _FakeFloatValue(Structure):
    _fields_ = [
        ("fCurValue", c_double),
        ("fMax", c_double),
        ("fMin", c_double),
    ]


class _FakeEnumValue(Structure):
    _fields_ = [
        ("nCurValue", c_uint64),
        ("nSupportedNum", c_uint),
        ("nSupportValue", c_uint64 * 16),
    ]


class _FakeEnumEntry(Structure):
    _fields_ = [("nValue", c_uint64), ("chSymbolic", c_char * 64)]


class _FakeStringValue(Structure):
    _fields_ = [
        ("chCurValue", c_char * 128),
        ("nMaxLength", c_int),
    ]


class _FakeMvsSdk:
    MV_XML_AccessMode = c_int
    MV_XML_InterfaceType = c_int
    MVCC_INTVALUE_EX = _FakeIntValue
    MVCC_FLOATVALUE = _FakeFloatValue
    MVCC_ENUMVALUE_EX = _FakeEnumValue
    MVCC_ENUMVALUE = _FakeEnumValue
    MVCC_ENUMENTRY = _FakeEnumEntry
    MVCC_STRINGVALUE = _FakeStringValue


class _FakeCamera:
    def __init__(self) -> None:
        self.handle = object()
        self.access_modes = {
            "IntegerNode": AM_RW,
            "FloatNode": AM_RW,
            "EnumNode": AM_RW,
            "BoolNode": AM_RW,
            "StringNode": AM_RW,
            "CommandNode": AM_RW,
        }
        self.interface_types: dict[str, int] = {}
        self.set_calls: list[tuple[str, str, object]] = []
        self.fail_writes: set[str] = set()

    def MV_XML_GetNodeAccessMode(self, name: str, output: c_int) -> int:
        output.value = self.access_modes.get(name, AM_RO)
        return 0

    def MV_XML_GetNodeInterfaceType(self, name: str, output: c_int) -> int:
        output.value = self.interface_types.get(name, 0)
        return 0

    def MV_CC_GetIntValueEx(self, name: str, output: _FakeIntValue) -> int:
        self.assert_node(name, "IntegerNode")
        output.nCurValue = 640
        output.nMin = 64
        output.nMax = 4096
        output.nInc = 8
        return 0

    def MV_CC_GetFloatValue(self, name: str, output: _FakeFloatValue) -> int:
        self.assert_node(name, "FloatNode")
        output.fCurValue = 1500.25
        output.fMin = 10.0
        output.fMax = 20000.0
        return 0

    def MV_CC_GetEnumValueEx(self, name: str, output: _FakeEnumValue) -> int:
        self.assert_node(name, "EnumNode")
        output.nCurValue = 1
        supported = (0, 1, 7)
        output.nSupportedNum = len(supported)
        for index, value in enumerate(supported):
            output.nSupportValue[index] = value
        return 0

    # CameraFeatureAccessor deliberately supports old and new SDK method
    # names.  The fallback attribute is evaluated eagerly by Python's getattr.
    MV_CC_GetEnumValue = MV_CC_GetEnumValueEx

    def MV_CC_GetEnumEntrySymbolic(
        self, name: str, output: _FakeEnumEntry
    ) -> int:
        self.assert_node(name, "EnumNode")
        symbolic = {0: b"Off", 1: b"Once", 7: b"Continuous"}
        output.chSymbolic = symbolic[int(output.nValue)]
        return 0

    def MV_CC_GetBoolValue(self, name: str, output: object) -> int:
        self.assert_node(name, "BoolNode")
        output.value = True
        return 0

    def MV_CC_GetStringValue(self, name: str, output: _FakeStringValue) -> int:
        self.assert_node(name, "StringNode")
        output.chCurValue = b"calibration-camera"
        output.nMaxLength = 127
        return 0

    def _set(self, kind: str, name: str, value: object) -> int:
        self.set_calls.append((kind, name, value))
        return 0x80000001 if name in self.fail_writes else 0

    def MV_CC_SetIntValueEx(self, name: str, value: int) -> int:
        return self._set("Integer", name, value)

    def MV_CC_SetFloatValue(self, name: str, value: float) -> int:
        return self._set("Float", name, value)

    def MV_CC_SetEnumValue(self, name: str, value: int) -> int:
        return self._set("Enumeration", name, value)

    def MV_CC_SetEnumValueByString(self, name: str, value: str) -> int:
        return self._set("EnumerationString", name, value)

    def MV_CC_SetBoolValue(self, name: str, value: bool) -> int:
        return self._set("Boolean", name, value)

    def MV_CC_SetStringValue(self, name: str, value: str) -> int:
        return self._set("String", name, value)

    def MV_CC_SetCommandValue(self, name: str) -> int:
        return self._set("Command", name, None)

    @staticmethod
    def assert_node(actual: str, expected: str) -> None:
        if actual != expected:
            raise AssertionError(f"expected node {expected}, got {actual}")


class _FakeController:
    def __init__(self) -> None:
        self.cam = _FakeCamera()
        self.lock = threading.RLock()
        self.device_open = True


def _feature(name: str, interface_type: str) -> FeatureSpec:
    return FeatureSpec(
        name=name,
        interface_type=interface_type,
        access_mode=AM_RW,
    )


class GenICamXmlParsingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.features = parse_genicam_xml(SYNTHETIC_GENICAM_XML)
        self.by_name = {feature.name: feature for feature in self.features}

    def test_walks_only_categories_reachable_from_root_in_feature_order(self) -> None:
        self.assertEqual(
            [feature.name for feature in self.features],
            ["ExposureTime", "Gain", "NoiseReduction", "Width", "PixelFormat"],
        )
        self.assertNotIn("OrphanValue", self.by_name)
        self.assertNotIn("UnreferencedRegister", self.by_name)
        self.assertNotIn("MissingFeature", self.by_name)
        self.assertNotIn("RawLutData", self.by_name)

    def test_preserves_nested_category_hierarchy(self) -> None:
        exposure = self.by_name["ExposureTime"]
        noise_reduction = self.by_name["NoiseReduction"]

        self.assertEqual(exposure.category, "AnalogControl")
        self.assertEqual(exposure.category_path, ("AnalogControl",))
        self.assertEqual(noise_reduction.category, "AdvancedAnalogControl")
        self.assertEqual(
            noise_reduction.category_path,
            ("AnalogControl", "AdvancedAnalogControl"),
        )

    def test_internal_duplicate_names_do_not_shadow_public_interfaces(self) -> None:
        names = [feature.name for feature in self.features]
        self.assertEqual(names.count("Gain"), 1)
        self.assertEqual(names.count("Width"), 1)
        self.assertEqual(names.count("ExposureTime"), 1)
        self.assertEqual(self.by_name["Gain"].interface_type, "Float")
        self.assertEqual(self.by_name["Width"].interface_type, "Integer")
        self.assertEqual(self.by_name["ExposureTime"].interface_type, "Float")

    def test_preserves_display_and_control_metadata_without_claiming_runtime_access(self) -> None:
        exposure = self.by_name["ExposureTime"]
        self.assertEqual(exposure.display_name, "Exposure Time")
        self.assertEqual(
            exposure.description,
            "Controls how long the sensor integrates light.",
        )
        self.assertEqual(exposure.tooltip, "Exposure duration in microseconds.")
        self.assertEqual(exposure.visibility, "Beginner")
        self.assertEqual(exposure.unit, "us")
        self.assertEqual(exposure.representation, "Logarithmic")
        self.assertTrue(exposure.streamable)
        self.assertEqual(exposure.declared_access_mode, "RW")
        self.assertIsNone(exposure.access_mode)

        noise_reduction = self.by_name["NoiseReduction"]
        self.assertEqual(noise_reduction.visibility, "Guru")
        self.assertEqual(noise_reduction.declared_access_mode, "RO")

    def test_accepts_text_as_well_as_bytes(self) -> None:
        decoded = SYNTHETIC_GENICAM_XML.decode("utf-8")
        from_text = parse_genicam_xml(decoded)
        self.assertEqual(
            [(feature.name, feature.category_path) for feature in from_text],
            [(feature.name, feature.category_path) for feature in self.features],
        )


class FeatureSafetyClassificationTest(unittest.TestCase):
    def test_structural_features_are_identified(self) -> None:
        for name in (
            "Width",
            "Height",
            "OffsetX",
            "OffsetY",
            "PixelFormat",
            "BinningHorizontal",
            "DecimationVertical",
        ):
            with self.subTest(name=name):
                self.assertTrue(is_structural_feature(name))

        self.assertFalse(is_structural_feature("ExposureTime"))
        self.assertFalse(is_structural_feature("Gain"))

    def test_dangerous_or_persistent_actions_are_identified(self) -> None:
        for name in (
            "DeviceReset",
            "UserSetLoad",
            "UserSetSave",
            "LUTSave",
            "ActivateShading",
            "LineMode",
            "LineSource",
            "StrobeEnable",
            "UserOutputValue",
        ):
            with self.subTest(name=name):
                self.assertTrue(is_dangerous_feature(name))

        self.assertFalse(is_dangerous_feature("ExposureTime"))
        self.assertFalse(is_dangerous_feature("Gain"))


class CameraFeatureAccessorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = _FakeController()
        self.accessor = CameraFeatureAccessor(
            self.controller, mvs_module=_FakeMvsSdk
        )
        self.integer = _feature("IntegerNode", "Integer")
        self.floating = _feature("FloatNode", "Float")
        self.enumeration = FeatureSpec(
            name="EnumNode",
            interface_type="Enumeration",
            access_mode=AM_RW,
            enum_entries=(
                EnumEntrySpec("Off", value=0, display_name="关闭"),
                EnumEntrySpec("Once", value=1, display_name="单次"),
                EnumEntrySpec("Continuous", value=7, display_name="连续"),
            ),
        )
        self.boolean = _feature("BoolNode", "Boolean")
        self.string = _feature("StringNode", "String")
        self.command = _feature("CommandNode", "Command")

    def test_reads_all_supported_value_interfaces(self) -> None:
        integer = self.accessor.read(self.integer)
        self.assertEqual(integer.value, 640)
        self.assertEqual(integer.minimum, 64)
        self.assertEqual(integer.maximum, 4096)
        self.assertEqual(integer.increment, 8)

        floating = self.accessor.read(self.floating)
        self.assertAlmostEqual(floating.value, 1500.25)
        self.assertEqual(floating.minimum, 10.0)
        self.assertEqual(floating.maximum, 20000.0)

        enumeration = self.accessor.read(self.enumeration)
        self.assertEqual(enumeration.value, 1)
        self.assertEqual(
            [
                (choice.value, choice.symbolic, choice.display_name)
                for choice in enumeration.choices
            ],
            [
                (0, "Off", "关闭"),
                (1, "Once", "单次"),
                (7, "Continuous", "连续"),
            ],
        )

        self.assertIs(self.accessor.read(self.boolean).value, True)
        string = self.accessor.read(self.string)
        self.assertEqual(string.value, "calibration-camera")
        self.assertEqual(string.max_length, 127)
        self.assertIsNone(self.accessor.read(self.command).value)

    def test_writes_all_supported_interfaces(self) -> None:
        self.accessor.write(self.integer, 800)
        self.accessor.write(self.floating, 2500.5)
        self.accessor.write(
            self.enumeration, EnumChoice(value=7, symbolic="Continuous")
        )
        self.accessor.write(self.enumeration, "Once")
        self.accessor.write(self.boolean, False)
        self.accessor.write(self.string, "camera-a")
        self.accessor.write(self.command)

        self.assertEqual(
            self.controller.cam.set_calls,
            [
                ("Integer", "IntegerNode", 800),
                ("Float", "FloatNode", 2500.5),
                ("Enumeration", "EnumNode", 7),
                ("EnumerationString", "EnumNode", "Once"),
                ("Boolean", "BoolNode", False),
                ("String", "StringNode", "camera-a"),
                ("Command", "CommandNode", None),
            ],
        )

    def test_rechecks_dynamic_access_and_refuses_stale_writable_node(self) -> None:
        # The catalog said RW, but acquisition/selector state changed it to RO
        # before the user clicked Apply.
        self.controller.cam.access_modes[self.integer.name] = AM_RO

        with self.assertRaisesRegex(
            CameraFeatureError, r"IntegerNode is not writable \(access RO\)"
        ):
            self.accessor.write(self.integer, 1024)

        self.assertEqual(self.controller.cam.set_calls, [])

    def test_sdk_write_error_contains_unsigned_hex_code(self) -> None:
        self.controller.cam.fail_writes.add(self.floating.name)
        with self.assertRaisesRegex(CameraFeatureError, "0x80000001"):
            self.accessor.write(self.floating, 999.0)


class _PanelAccessor:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.connected = True
        self.error = error
        self.write_calls: list[tuple[FeatureSpec, object | None]] = []

    def write(self, spec: FeatureSpec, value: object | None = None) -> None:
        self.write_calls.append((spec, value))
        if self.error is not None:
            raise self.error


@unittest.skipIf(QApplication is None, "PyQt5 is unavailable")
class CameraFeaturePanelCallbackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.panel = CameraFeaturePanel()
        self.spec = _feature("FloatNode", "Float")

    def tearDown(self) -> None:
        self.panel.close()
        self.panel.deleteLater()
        self._drain_events()

    @classmethod
    def _drain_events(cls) -> None:
        # Zero-delay singleShot callbacks run when control returns to Qt.
        cls.app.processEvents()
        cls.app.processEvents()

    def test_after_change_runs_even_when_sdk_write_fails(self) -> None:
        accessor = _PanelAccessor(error=CameraFeatureError("simulated SDK error"))
        self.panel.accessor = accessor  # type: ignore[assignment]
        self.panel.before_change = lambda _spec: True
        after_calls: list[FeatureSpec] = []
        failures: list[tuple[FeatureSpec, str]] = []
        self.panel.after_change = after_calls.append
        self.panel.change_failed.connect(
            lambda spec, message: failures.append((spec, message))
        )

        self.panel._request_change(self.spec, 42.0)

        self.assertEqual(accessor.write_calls, [(self.spec, 42.0)])
        self.assertEqual(after_calls, [self.spec])
        self.assertEqual(len(failures), 1)
        self.assertIn("simulated SDK error", failures[0][1])

    def test_cancelled_change_refreshes_editor_without_writing(self) -> None:
        accessor = _PanelAccessor()
        self.panel.accessor = accessor  # type: ignore[assignment]
        self.panel.before_change = lambda _spec: False
        refresh = MagicMock()
        self.panel.refresh_features = refresh

        self.panel._request_change(self.spec, 42.0)
        self._drain_events()

        self.assertEqual(accessor.write_calls, [])
        refresh.assert_called_once_with()
        self.assertIn("已取消修改", self.panel.status_label.text())

    def test_failed_change_without_host_callback_refreshes_editor(self) -> None:
        accessor = _PanelAccessor(error=CameraFeatureError("write rejected"))
        self.panel.accessor = accessor  # type: ignore[assignment]
        self.panel.before_change = lambda _spec: True
        self.panel.after_change = None
        refresh = MagicMock()
        self.panel.refresh_features = refresh

        self.panel._request_change(self.spec, 42.0)
        self._drain_events()

        self.assertEqual(accessor.write_calls, [(self.spec, 42.0)])
        refresh.assert_called_once_with()
        self.assertFalse(self.panel._refresh_pending)
        self.assertIn("设置 FloatNode 失败", self.panel.status_label.text())


if __name__ == "__main__":
    unittest.main()
