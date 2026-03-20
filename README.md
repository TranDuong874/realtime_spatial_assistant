# realtime_spatial_assistant

SLAM-only Python bindings for the vendored `thirdparty/ORB_SLAM3` tree.

The Python-facing package is `slam`, which exposes a small direct wrapper over
the ORB-SLAM3 `System` API for monocular, stereo, RGB-D, and inertial modes.

## Quick Start

Build the vendored native dependencies and the Python extension:

```bash
cd realtime_spatial_assistant
MAKE_JOBS=2 bash ./build_orbslam3_python.sh
```

Run the import smoke test:

```bash
cd realtime_spatial_assistant
python3 test_slam_import.py
```

## Package

```python
import slam

system = slam.System(
    "thirdparty/ORB_SLAM3/Vocabulary/ORBvoc.txt",
    "path/to/settings.yaml",
    slam.Sensor.IMU_STEREO,
    use_viewer=False,
)
```

Exports:

- `slam.System`
- `slam.Sensor`
- `slam.ImuMeasurement`

## Docs

- `docs/API.md`: Python API reference for `slam`
- `docs/BUILD.md`: build prerequisites, build flow, rebuild notes

## Notes

- The build defaults to `MAKE_JOBS=2` because ORB-SLAM3 is memory-heavy.
- The extension binary is generated at `slam/_orbslam3*.so`.
- The vendored ORB-SLAM3 native build products are treated as generated files and are ignored by `.gitignore`.
