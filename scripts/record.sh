source /opt/ros/humble/setup.bash && source ~/thesis/ros_ws/install/setup.bash
MicroXRCEAgent serial --dev /dev/ttyUSB0 -b 921600 &

sleep 5
ros2 bag record --compression-mode message --compression-format zstd \
    /fmu/out/sensor_baro \
    /fmu/out/sensor_combined \
    /fmu/out/sensor_mag \
    /fmu/out/vehicle_attitude \
    /fmu/out/vehicle_global_position \
    /fmu/out/vehicle_local_position \
    /fmu/out/vehicle_odometry \
    /fmu/out/vehicle_gps_position \
    /fmu/out/vehicle_magnetometer \
    /fmu/out/timesync_status
