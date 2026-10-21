# Included USB camera runtime

Copied from the verified 2026-10-20 test-01 archive's real_robot_collection `.ros_local_root/opt/ros/humble`, not downloaded from a different driver version. `lib/libusb_cam.so` SHA256:

`f9fd906092b666bf017898928c19d1f6e3240dad1a5ffbbe54138e63fdeb11ee`

The timestamp correction is required by the existing source-time pairing checks. The launcher's hash check is retained. Prefix includes usb_cam executable/library, camera_info_manager and camera_calibration_parsers libraries plus their ament metadata. CMake export files with development install paths are omitted; this is a runtime prefix, not a source-build SDK. Package metadata retains original licensing information.

Target: Ubuntu22.04 x86_64 / ROS2 Humble. Ubuntu/ROS shared libraries remain external prerequisites. `libusb_cam.so` directly needs libavcodec.so.58, libavutil.so.56, libswscale.so.5, libopencv_imgproc.so.4.5d, libopencv_core.so.4.5d and standard C/C++ runtime. ROS node additionally needs the matching Humble rclcpp, sensor_msgs, image_transport and related system libraries. On the customer machine, after sourcing the configured environment, inspect unresolved libraries with `ldd tools/customer/runtime/ros_humble_x86_64/lib/usb_cam/usb_cam_node_exe`; any `not found` must be resolved before camera tests. This command does not launch a camera.

The vendor native libraries and service files elsewhere in SONIC retain their own notices. This bundle does not claim to contain vendor source that was not provided.
