#!/usr/bin/env python3
"""Capture an image from a HIKROBOT camera with the MVS Python SDK."""

from __future__ import annotations

import argparse
import os
import platform
import sys
from ctypes import POINTER, byref, cast, memset, sizeof
from datetime import datetime
from pathlib import Path


class CameraError(RuntimeError):
    """Raised when an MVS SDK call fails."""


def sdk_error(operation: str, code: int) -> CameraError:
    unsigned_code = code & 0xFFFFFFFF
    hints = {
        0x80000203: (
            "access denied; disconnect this camera in the MVS client and check "
            "the USB udev permissions"
        ),
        0x80000214: (
            "UDP receive failed; put the camera and wired network adapter on "
            "the same IPv4 subnet"
        ),
    }
    suffix = f" ({hints[unsigned_code]})" if unsigned_code in hints else ""
    return CameraError(f"{operation} failed: 0x{unsigned_code:08x}{suffix}")


def require_ok(operation: str, code: int) -> None:
    if code != 0:
        raise sdk_error(operation, code)


def configure_sdk_import() -> Path:
    if platform.system() != "Linux":
        raise CameraError("This example currently supports Linux only")

    mvs_root = Path(os.environ.get("MVCAM_SDK_PATH", "/opt/MVS")).resolve()
    architecture = "64" if sys.maxsize > 2**32 else "32"
    binding_dir = mvs_root / "Samples" / architecture / "Python" / "MvImport"
    library_dir = mvs_root / "lib"

    if not binding_dir.is_dir():
        raise CameraError(
            f"MVS Python bindings not found at {binding_dir}. "
            "Install the HIKROBOT MVS SDK or set MVCAM_SDK_PATH."
        )

    os.environ.setdefault("MVCAM_COMMON_RUNENV", str(library_dir))
    sys.path.insert(0, str(binding_dir))
    return mvs_root


MVS_ROOT = configure_sdk_import()

try:
    # The bindings and constants are supplied by the installed HIKROBOT MVS SDK.
    from MvCameraControl_class import (  # type: ignore[import-not-found]  # noqa: E402,F403
        MV_ACCESS_Exclusive,
        MV_CC_DEVICE_INFO,
        MV_CC_DEVICE_INFO_LIST,
        MV_CC_IMAGE,
        MV_CC_SAVE_IMAGE_PARAM,
        MV_FRAME_OUT,
        MV_GIGE_DEVICE,
        MV_Image_Png,
        MV_TRIGGER_MODE_OFF,
        MV_USB_DEVICE,
        MvCamera,
    )
except (ImportError, OSError) as exc:
    raise CameraError(f"Unable to load the MVS SDK from {MVS_ROOT}: {exc}") from exc


SUPPORTED_TRANSPORTS = MV_GIGE_DEVICE | MV_USB_DEVICE


def decode_c_string(value: object) -> str:
    raw = memoryview(value).tobytes().split(b"\0", 1)[0]
    for encoding in ("utf-8", "gbk", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def device_identity(device_info: MV_CC_DEVICE_INFO) -> tuple[str, str, str]:
    if device_info.nTLayerType == MV_USB_DEVICE:
        info = device_info.SpecialInfo.stUsb3VInfo
        return (
            "USB3 Vision",
            decode_c_string(info.chModelName),
            decode_c_string(info.chSerialNumber),
        )
    if device_info.nTLayerType == MV_GIGE_DEVICE:
        info = device_info.SpecialInfo.stGigEInfo
        return (
            "GigE Vision",
            decode_c_string(info.chModelName),
            decode_c_string(info.chSerialNumber),
        )
    return (f"transport=0x{device_info.nTLayerType:x}", "unknown", "unknown")


def ipv4_from_int(value: int) -> str:
    """Format the big-endian IPv4 integers used by the MVS structures."""

    return ".".join(str((int(value) >> shift) & 0xFF) for shift in (24, 16, 8, 0))


def device_network_info(device_info: MV_CC_DEVICE_INFO) -> tuple[str, str] | None:
    """Return ``(camera_ip, host_adapter_ip)`` for a GigE camera."""

    if device_info.nTLayerType != MV_GIGE_DEVICE:
        return None
    info = device_info.SpecialInfo.stGigEInfo
    return ipv4_from_int(info.nCurrentIp), ipv4_from_int(info.nNetExport)


def enumerate_devices() -> tuple[MV_CC_DEVICE_INFO_LIST, list[MV_CC_DEVICE_INFO]]:
    device_list = MV_CC_DEVICE_INFO_LIST()
    require_ok(
        "MV_CC_EnumDevices",
        MvCamera.MV_CC_EnumDevices(SUPPORTED_TRANSPORTS, device_list),
    )

    devices = [
        cast(device_list.pDeviceInfo[index], POINTER(MV_CC_DEVICE_INFO)).contents
        for index in range(device_list.nDeviceNum)
    ]
    return device_list, devices


def print_devices(devices: list[MV_CC_DEVICE_INFO]) -> None:
    print(f"Found {len(devices)} camera(s):")
    for index, device in enumerate(devices):
        transport, model, serial = device_identity(device)
        network = device_network_info(device)
        network_text = (
            f"  camera_ip={network[0]}  host_ip={network[1]}" if network else ""
        )
        print(
            f"  [{index}] {model}  serial={serial}  transport={transport}"
            f"{network_text}"
        )


def select_device(
    devices: list[MV_CC_DEVICE_INFO], index: int, serial: str | None
) -> tuple[int, MV_CC_DEVICE_INFO]:
    if not devices:
        raise CameraError("No USB3 Vision or GigE Vision camera was found")

    if serial:
        for device_index, device in enumerate(devices):
            if device_identity(device)[2] == serial:
                return device_index, device
        raise CameraError(f"No camera with serial number {serial!r} was found")

    if index < 0 or index >= len(devices):
        raise CameraError(
            f"Device index {index} is out of range; found {len(devices)} camera(s)"
        )
    return index, devices[index]


def default_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("captures") / f"hikrobot_{timestamp}.png"


def validate_output_path(output: Path) -> Path:
    resolved = output.expanduser().resolve()
    if resolved.suffix.lower() != ".png":
        raise CameraError("The output filename must end in .png")
    try:
        str(resolved).encode("ascii")
    except UnicodeEncodeError as exc:
        raise CameraError(
            "The MVS Python wrapper requires an ASCII-only output path"
        ) from exc
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def save_frame_as_png(cam: MvCamera, frame: MV_FRAME_OUT, output: Path) -> None:
    frame_info = frame.stFrameInfo
    image = MV_CC_IMAGE()
    memset(byref(image), 0, sizeof(image))
    image.nWidth = frame_info.nExtendWidth or frame_info.nWidth
    image.nHeight = frame_info.nExtendHeight or frame_info.nHeight
    image.enPixelType = frame_info.enPixelType
    image.pImageBuf = frame.pBufAddr
    image.nImageBufSize = frame_info.nFrameLenEx or frame_info.nFrameLen
    image.nImageLen = frame_info.nFrameLenEx or frame_info.nFrameLen

    save_parameters = MV_CC_SAVE_IMAGE_PARAM()
    memset(byref(save_parameters), 0, sizeof(save_parameters))
    save_parameters.enImageType = MV_Image_Png
    save_parameters.iMethodValue = 1
    save_parameters.nQuality = 90
    save_parameters.nEndian = 0

    require_ok(
        "MV_CC_SaveImageToFileEx2",
        cam.MV_CC_SaveImageToFileEx2(image, save_parameters, str(output)),
    )


def capture(args: argparse.Namespace, devices: list[MV_CC_DEVICE_INFO]) -> Path:
    selected_index, device = select_device(devices, args.device_index, args.serial)
    transport, model, serial = device_identity(device)
    print(
        f"Opening [{selected_index}] {model} "
        f"(serial={serial}, transport={transport})"
    )

    output = validate_output_path(args.output)
    cam = MvCamera()
    handle_created = False
    device_open = False
    grabbing = False

    try:
        require_ok("MV_CC_CreateHandle", cam.MV_CC_CreateHandle(device))
        handle_created = True
        require_ok(
            "MV_CC_OpenDevice", cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        )
        device_open = True

        if device.nTLayerType == MV_GIGE_DEVICE:
            packet_size = int(cam.MV_CC_GetOptimalPacketSize())
            if packet_size > 0:
                packet_result = cam.MV_CC_SetIntValue(
                    "GevSCPSPacketSize", packet_size
                )
                if packet_result != 0:
                    print(
                        "Warning: setting the optimal GigE packet size failed: "
                        f"0x{packet_result & 0xFFFFFFFF:08x}",
                        file=sys.stderr,
                    )
            else:
                print(
                    "Warning: unable to determine the optimal GigE packet size: "
                    f"0x{packet_size & 0xFFFFFFFF:08x}",
                    file=sys.stderr,
                )

        require_ok(
            "Set TriggerMode=Off",
            cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF),
        )
        if args.exposure_us is not None:
            require_ok(
                "Set ExposureAuto=Off", cam.MV_CC_SetEnumValue("ExposureAuto", 0)
            )
            require_ok(
                "Set ExposureTime",
                cam.MV_CC_SetFloatValue("ExposureTime", args.exposure_us),
            )
        if args.gain is not None:
            require_ok("Set GainAuto=Off", cam.MV_CC_SetEnumValue("GainAuto", 0))
            require_ok("Set Gain", cam.MV_CC_SetFloatValue("Gain", args.gain))

        require_ok("MV_CC_StartGrabbing", cam.MV_CC_StartGrabbing())
        grabbing = True

        frame = MV_FRAME_OUT()
        memset(byref(frame), 0, sizeof(frame))
        require_ok(
            "MV_CC_GetImageBuffer",
            cam.MV_CC_GetImageBuffer(frame, args.timeout_ms),
        )
        if not frame.pBufAddr:
            raise CameraError("The SDK returned an empty image buffer")

        try:
            save_frame_as_png(cam, frame, output)
            info = frame.stFrameInfo
            print(
                f"Captured frame={info.nFrameNum} size={info.nWidth}x{info.nHeight} "
                f"pixel_type=0x{info.enPixelType:08x}"
            )
        finally:
            require_ok("MV_CC_FreeImageBuffer", cam.MV_CC_FreeImageBuffer(frame))

        return output
    finally:
        if grabbing:
            stop_result = cam.MV_CC_StopGrabbing()
            if stop_result != 0:
                print(
                    f"Warning: MV_CC_StopGrabbing failed: "
                    f"0x{stop_result & 0xFFFFFFFF:08x}",
                    file=sys.stderr,
                )
        if device_open:
            close_result = cam.MV_CC_CloseDevice()
            if close_result != 0:
                print(
                    f"Warning: MV_CC_CloseDevice failed: "
                    f"0x{close_result & 0xFFFFFFFF:08x}",
                    file=sys.stderr,
                )
        if handle_created:
            destroy_result = cam.MV_CC_DestroyHandle()
            if destroy_result != 0:
                print(
                    f"Warning: MV_CC_DestroyHandle failed: "
                    f"0x{destroy_result & 0xFFFFFFFF:08x}",
                    file=sys.stderr,
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List HIKROBOT cameras or capture one frame through the MVS SDK."
    )
    parser.add_argument(
        "--list", action="store_true", help="list cameras without opening one"
    )
    parser.add_argument(
        "--device-index", type=int, default=0, help="camera index to open (default: 0)"
    )
    parser.add_argument("--serial", help="select a camera by serial number")
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output_path(),
        help="PNG output path (default: captures/hikrobot_TIMESTAMP.png)",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=5000,
        help="frame wait timeout in milliseconds (default: 5000)",
    )
    parser.add_argument(
        "--exposure-us", type=float, help="optional manual exposure time in microseconds"
    )
    parser.add_argument("--gain", type=float, help="optional manual analog gain")
    args = parser.parse_args()
    if args.timeout_ms <= 0:
        parser.error("--timeout-ms must be greater than zero")
    if args.exposure_us is not None and args.exposure_us <= 0:
        parser.error("--exposure-us must be greater than zero")
    if args.gain is not None and args.gain < 0:
        parser.error("--gain must be zero or greater")
    return args


def main() -> int:
    args = parse_args()
    initialized = False
    try:
        require_ok("MV_CC_Initialize", MvCamera.MV_CC_Initialize())
        initialized = True
        version = MvCamera.MV_CC_GetSDKVersion()
        print(f"MVS SDK version: 0x{version:08x}")

        # Keep device_list alive while device structures are in use; its memory is SDK-owned.
        device_list, devices = enumerate_devices()
        _ = device_list
        print_devices(devices)
        if args.list:
            return 0

        output = capture(args, devices)
        print(f"Saved: {output}")
        return 0
    except CameraError as exc:
        sys.stdout.flush()
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if initialized:
            final_result = MvCamera.MV_CC_Finalize()
            if final_result != 0:
                print(
                    f"Warning: MV_CC_Finalize failed: "
                    f"0x{final_result & 0xFFFFFFFF:08x}",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    raise SystemExit(main())
