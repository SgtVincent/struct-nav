# habitat_ros

Integrate Habitat simulator with ROS

## Requirements

- ROS noetic
- CMake version at least 2.8.3
- Python version at least 3.6
- Habitat simulator
- Rtabmap-ros
- Python packages: numpy, gym, keyboard, cv_bridge

## Installation

1. Install ROS noetic on your system ([https://www.ros.org/install/](https://www.ros.org/install/)).

2. Install ros numpy: `sudo apt-get install ros-noetic-ros-numpy`

3. Clone this repo into your ROS workspace (e.g. `~/catkin_ws/src`), and run `catkin_make`.

4. Download and build [Habitat-sim](https://github.com/facebookresearch/habitat-sim) and [Habitat-lab](https://github.com/facebookresearch/habitat-lab).

5. Install required Python packages: `pip install -r requirements.txt`

6. In file `habitat-lab/habitat/tasks/nav/nav.py`, find method `step` in class `StopAction`, and change line `task.is_stop_called = True` to `task.is_stop_called=False` in code of this method.

7. Rebuild habitat-lab.

8. Install [Rtabmap-ros](https://github.com/introlab/rtabmap_ros)

<<<<<<< HEAD 9. Install [explore_lite](http://wiki.ros.org/explore_lite) with: `sudo apt install ros-${ROS_DISTRO}-explore-lite`

=======

> > > > > > > 70b01e994fc7e3318b56468537da0b5872b8076b

## Launch habitat node

Run agent in simulator via `roslaunch habitat_ros habitat_agent.launch`.

## Node parameters

- rgb_topic (default: None) - topic to publish RGB image from simulator
- depth_topic (default: None) - topic to publish depth image from simulator
- camera_info_topic (default: None) - topic to publish information about camera calibration (read from file)
- camera_calib (default: None) - path to camera calibration file

## Launch RTAB-MAP node

```bash

roslaunch rtabmap_ros rtabmap.launch \
    rtabmap_args:="--delete_db_on_start" \
    depth_topic:=/camera/depth/image \
    rgb_topic:=/camera/rgb/image \
    camera_info_topic:=/camera/rgb/camera_info \
    approx_sync:=false

```

Get cloud map with a higher resolution

```bash

roslaunch rtabmap_ros rtabmap.launch \
    rtabmap_args:="--delete_db_on_start --Grid/CellSize 0.01 --Crid/DepthDecimation 1" \
    depth_topic:=/camera/depth/image \
    rgb_topic:=/camera/rgb/image \
    camera_info_topic:=/camera/rgb/camera_info \
    approx_sync:=false

```

## Extra documentations

### TF tree in this module

[demo](./img/tf_tree.png)

### TODO LIST

- [ ] consider if we need to move agent & planning functions from habitat_ros to a new package
- [ ] load habitat configuration from yaml file instead of using hard-coded settings in [simulator.py](habitat/scripts/simulator.py)
