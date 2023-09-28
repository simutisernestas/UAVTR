gz plugin libs were copied from /home/ernie/thesis/PX4-Autopilot/ros_ws/build/mbzirc_ign to:
 - `cp libSurface.so /home/ernie/.gz/sim/plugins/`

image bridge:
 - `ros2 run ros_gz_image image_bridge /world/default/model/x500_0/link/realsense_d435/base_link/sensor/realsense_d435/image`

general bridge config:
 - https://github.com/gazebosim/ros_gz/tree/ros2/ros_gz_bridge
 - ros2 run ros_gz_bridge parameter_bridge --ros-args -p config_file:=topic_map.yml

world inspired by:
 - https://github.com/osrf/mbzirc/blob/main/mbzirc_ign/worlds/coast.sdf

think how to orginize the code

# MicroXRCEAgent udp4 -p 8888
# make px4_sitl gz_x500

read the compatability guide
https://gazebosim.org/docs/harmonic/ros_installation

should prob create an image for enviroment

