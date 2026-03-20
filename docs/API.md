# Python SLAM API

## Package

```python
import slam
```

Exports:

- `slam.System`
- `slam.Sensor`
- `slam.ImuMeasurement`

The package is a direct Python binding over the vendored ORB-SLAM3 `System`
class in `thirdparty/ORB_SLAM3`.

## Sensor Enum

Available sensor modes:

- `slam.Sensor.MONOCULAR`
- `slam.Sensor.STEREO`
- `slam.Sensor.RGBD`
- `slam.Sensor.IMU_MONOCULAR`
- `slam.Sensor.IMU_STEREO`
- `slam.Sensor.IMU_RGBD`

Pick the enum that matches both your input stream and the YAML settings file
you pass into ORB-SLAM3.

## System Construction

```python
system = slam.System(
    vocabulary_path,
    settings_path,
    slam.Sensor.IMU_STEREO,
    use_viewer=False,
    init_frame=0,
    sequence="",
)
```

Arguments:

- `vocabulary_path`: path to `ORBvoc.txt`
- `settings_path`: path to an ORB-SLAM3 camera or sensor YAML file
- `sensor`: one of the `slam.Sensor` enum values
- `use_viewer`: whether to enable the Pangolin viewer
- `init_frame`: optional initial frame id
- `sequence`: optional sequence label passed to ORB-SLAM3

Example:

```python
import slam

system = slam.System(
    "thirdparty/ORB_SLAM3/Vocabulary/ORBvoc.txt",
    "configs/TUM-VI.yaml",
    slam.Sensor.IMU_STEREO,
    use_viewer=False,
)
```

## IMU Samples

Use `slam.ImuMeasurement` when running one of the inertial sensor modes.

```python
sample = slam.ImuMeasurement(
    timestamp_s,
    (ax, ay, az),
    (gx, gy, gz),
)
```

Arguments:

- `timestamp_s`: sample timestamp in seconds
- `linear_acceleration_m_s2`: `(ax, ay, az)`
- `angular_velocity_rad_s`: `(gx, gy, gz)`

Notes:

- Acceleration comes first, angular velocity second.
- The binding sorts IMU samples by timestamp before passing them into ORB-SLAM3.
- For non-IMU sensor modes, pass an empty list.

## Input Requirements

Images accepted by the tracking calls:

- grayscale `uint8` arrays shaped `(H, W)`
- color `uint8` arrays shaped `(H, W, 3)`

Depth images accepted by `track_rgbd`:

- `float32` or `float64`
- shape `(H, W)`

If the shape or dtype does not match these requirements, the binding raises
`ValueError` from Python with the validation message from the C++ wrapper.

## Tracking Methods

### `track_monocular`

```python
result = system.track_monocular(
    image,
    timestamp_s,
    imu_measurements=[],
    filename="",
)
```

Use this for:

- `slam.Sensor.MONOCULAR`
- `slam.Sensor.IMU_MONOCULAR`

### `track_stereo`

```python
result = system.track_stereo(
    left_image,
    right_image,
    timestamp_s,
    imu_measurements=[],
    filename="",
)
```

Use this for:

- `slam.Sensor.STEREO`
- `slam.Sensor.IMU_STEREO`

### `track_rgbd`

```python
result = system.track_rgbd(
    image,
    depth_image,
    timestamp_s,
    imu_measurements=[],
    filename="",
)
```

Use this for:

- `slam.Sensor.RGBD`
- `slam.Sensor.IMU_RGBD`

The optional `filename` is forwarded to ORB-SLAM3 and can be useful for
dataset-style logging or debugging.

## Tracking Result

Each `track_*` call returns a dictionary with:

- `tracking_state`: integer ORB-SLAM3 tracking state
- `tracking_state_name`: human-readable state label
- `is_keyframe`: whether ORB-SLAM3 promoted the last frame to a keyframe
- `pose_valid`: `True` when the returned pose is finite
- `pose_matrix`: `4x4 float32` pose matrix, or `None`
- `translation_xyz`: `(x, y, z)` tuple, or `None`
- `quaternion_wxyz`: `(w, x, y, z)` tuple, or `None`

Possible `tracking_state_name` values:

- `SYSTEM_NOT_READY`
- `NO_IMAGES_YET`
- `NOT_INITIALIZED`
- `OK`
- `RECENTLY_LOST`
- `LOST`
- `OK_KLT`
- `UNKNOWN`

Notes:

- Invalid or non-finite poses are returned as `pose_valid=False` and the pose
  fields are `None`.
- The matrix and pose values are returned exactly as exposed by ORB-SLAM3.
  If your downstream code expects world-to-camera or camera-to-world
  transforms, verify the convention in your pipeline before logging or fusing.

## State and Map Access

Available methods:

- `system.get_tracking_state()`
- `system.get_tracking_state_name()`
- `system.get_current_map_points()`
- `system.get_tracked_keypoints()`
- `system.get_tracked_observations()`
- `system.get_image_scale()`
- `system.reset()`
- `system.reset_active_map()`
- `system.shutdown()`
- `system.is_shutdown()`

### `get_tracking_state`

Returns the raw integer tracking state from ORB-SLAM3.

### `get_tracking_state_name`

Returns the string form of the current tracking state.

### `get_current_map_points`

Returns an `N x 3` `float32` NumPy array of sparse world points from the
current map.

### `get_tracked_keypoints`

Returns an `N x 2` `float32` NumPy array of current-frame keypoints in pixel
coordinates.

### `get_tracked_observations`

Returns a dictionary:

- `keypoints_uv`: `N x 2 float32`
- `world_points_xyz`: `N x 3 float32`

Entries are filtered so they only include valid tracked map points with finite
world coordinates.

### `get_image_scale`

Returns the current image scale factor used internally by ORB-SLAM3.

### `reset`

Requests a full ORB-SLAM3 reset.

### `reset_active_map`

Resets only the active map inside ORB-SLAM3.

### `shutdown`

Shuts down the underlying ORB-SLAM3 system. After shutdown, additional method
calls raise an exception.

### `is_shutdown`

Returns `True` if `shutdown()` has already been called.

## Minimal Example

```python
import numpy as np
import slam

system = slam.System(
    "thirdparty/ORB_SLAM3/Vocabulary/ORBvoc.txt",
    "configs/camera.yaml",
    slam.Sensor.MONOCULAR,
    use_viewer=False,
)

image = np.zeros((480, 640), dtype=np.uint8)
result = system.track_monocular(image, timestamp_s=0.0)

print(result["tracking_state_name"])
print(result["pose_valid"])
system.shutdown()
```

## Error Behavior

The binding raises Python exceptions in these common cases:

- invalid image shape or dtype
- invalid depth shape or dtype
- calling methods after `shutdown()`
- passing a sensor mode that does not match the tracking method being used
  after ORB-SLAM3 rejects it internally
