ros2 bag record --compression-mode message --compression-format zstd \
    /camera/accel/sample \
    /camera/color/camera_info \
    /camera/color/image_raw \
    /camera/gyro/sample \
    /camera/imu \
    /fmu/out/sensor_baro \
    /fmu/out/sensor_combined \
    /fmu/out/sensor_mag \
    /fmu/out/vehicle_attitude \
    /fmu/out/vehicle_global_position \
    /fmu/out/vehicle_local_position \
    /fmu/out/vehicle_odometry \
    /fmu/out/vehicle_gps_position \
    /fmu/out/vehicle_magnetometer \
    /tf_static \
    /teraranger_evo_40m \
    /fmu/out/timesync_status
