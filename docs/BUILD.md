# Build Guide

## Prerequisites

The local build expects these system dependencies to already be installed:

- C++14 compiler
- CMake
- OpenCV 4
- Eigen3
- Pangolin
- epoxy
- pybind11 headers or Python `pybind11`
- Python 3.10+
- `pkg-config`

## Build Command

Use low parallelism. ORB-SLAM3 and its third-party dependencies are memory-heavy
to compile.

```bash
cd realtime_spatial_assistant
MAKE_JOBS=2 bash ./build_orbslam3_python.sh
```

What the build script does:

1. builds `thirdparty/ORB_SLAM3/Thirdparty/DBoW2`
2. builds `thirdparty/ORB_SLAM3/Thirdparty/g2o`
3. builds `thirdparty/ORB_SLAM3/Thirdparty/Sophus`
4. builds `thirdparty/ORB_SLAM3/lib/libORB_SLAM3.so`
5. builds `slam/_orbslam3*.so`

The build script looks for pybind11 headers in this order:

1. the active Python environment via `import pybind11`
2. `/usr/include/pybind11`

## Smoke Test

```bash
cd realtime_spatial_assistant
python3 test_slam_import.py
```

Expected output:

```text
slam import ok
```

## Clean Rebuild

If you need to force a clean native rebuild:

```bash
rm -rf thirdparty/ORB_SLAM3/build
rm -rf thirdparty/ORB_SLAM3/lib
rm -rf thirdparty/ORB_SLAM3/Thirdparty/DBoW2/build
rm -rf thirdparty/ORB_SLAM3/Thirdparty/DBoW2/lib
rm -rf thirdparty/ORB_SLAM3/Thirdparty/g2o/build
rm -rf thirdparty/ORB_SLAM3/Thirdparty/g2o/lib
rm -rf thirdparty/ORB_SLAM3/Thirdparty/Sophus/build
rm -f slam/_orbslam3*.so
MAKE_JOBS=2 bash ./build_orbslam3_python.sh
```

## Generated Files

These are treated as build outputs and are ignored by `.gitignore`:

- `slam/_orbslam3*.so`
- `thirdparty/ORB_SLAM3/build/`
- `thirdparty/ORB_SLAM3/lib/`
- `thirdparty/ORB_SLAM3/Thirdparty/DBoW2/build/`
- `thirdparty/ORB_SLAM3/Thirdparty/DBoW2/lib/`
- `thirdparty/ORB_SLAM3/Thirdparty/g2o/build/`
- `thirdparty/ORB_SLAM3/Thirdparty/g2o/lib/`
- `thirdparty/ORB_SLAM3/Thirdparty/Sophus/build/`
- `thirdparty/ORB_SLAM3/Vocabulary/ORBvoc.txt`

## Notes

- The vendored ORB-SLAM3 build uses the repo's existing CMake and build scripts.
- The extension links against the vendored `libORB_SLAM3.so`, `libDBoW2.so`,
  and `libg2o.so` using runtime library paths baked into the generated module.
- ORB-SLAM3 emits many upstream compiler warnings on modern toolchains; these
  do not prevent the Python module from building successfully.
