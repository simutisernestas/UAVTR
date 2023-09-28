# Install packages
sudo apt update
sudo apt-get install -y curl gnupg git wget lsb-release software-properties-common tmux unzip zip gcc g++ cmake python3-pip

sudo apt-get clean all

# Locale
locale  # check for UTF-8
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

locale  # verify settings

# Install ROS2 HUMBLE
sudo apt install software-properties-common -y
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt upgrade -y
sudo apt install -y ros-humble-desktop
sudo apt install -y python3-colcon-common-extensions
sudo apt install ros-dev-tools -y
# Source ROS in profile script
echo 'source /opt/ros/humble/setup.bash' | sudo tee --append ~/.profile
source /opt/ros/humble/setup.bash

# PX4
# cd
# mkdir ~/
# cd ~/github
# git clone https://github.com/PX4/PX4-Autopilot.git --recursive
# bash ./PX4-Autopilot/Tools/setup/ubuntu.sh
#cd PX4-Autopilot/
#make px4_sitl

# Micro XRCE-DDS Agent
cd
git clone https://github.com/eProsima/Micro-XRCE-DDS-Agent.git
cd Micro-XRCE-DDS-Agent
mkdir build
cd build
cmake ..
make
sudo make install
sudo ldconfig /usr/local/lib/

# Create ROS2 workspace
mkdir -p ~/ros2_ws/src
# PX4 workspace
cd ~/ros2_ws/src
git clone https://github.com/PX4/px4_msgs.git -b release/1.14
cd ..
colcon build
echo 'source ~/ros2_ws/install/local_setup.bash' | sudo tee --append ~/.profile

# Gazebo
# sudo wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
# echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
# sudo apt-get update
# sudo apt-get install gz-garden


sudo mkdir -p /etc/apt/keyrings
curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null
echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo `lsb_release -cs` main" | \
sudo tee /etc/apt/sources.list.d/librealsense.list
sudo apt-get update
sudo apt-get install -y librealsense2-dkms
sudo apt-get install -y librealsense2-utils
sudo apt install -y ros-humble-realsense2-*

# for teraranger
sudo apt install -y ros-humble-serial-driver

sudo usermod -a -G dialout $USER

pip install parameterized


