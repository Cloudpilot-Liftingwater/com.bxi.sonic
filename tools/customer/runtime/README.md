# Included camera runtime

Ubuntu22.04 x86_64 / ROS2 Humble. The runtime originally came from the test-01 archive. libusb_cam.so now includes the clock-refresh patch from tools/camera_clock; its SHA256 is 513f7143ccbb376d92350ef28ab084e340e3c2bb840cafdf7c6874283a2e9c76. Source, license, rebuild test and field evidence are bundled there.

The usb_cam executable, camera_info_manager and camera_calibration_parsers runtimes and ament metadata are retained. External dependencies include matching ROS2 Humble, libavcodec.so.58, libavutil.so.56, libswscale.so.5, libopencv_imgproc.so.4.5d, libopencv_core.so.4.5d and the C/C++ runtime. This is a runtime prefix, not a complete ROS SDK.

After sourcing the site environment, use ldd on lib/usb_cam/usb_cam_node_exe to check unresolved dependencies before camera startup. Changing timestamps does not remove the remaining approximately129ms frame age. This source release never restarts robot services.
