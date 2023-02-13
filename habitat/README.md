# habitat_ros

Integrate Habitat simulator with ROS

## Requirements

-   RoboStack
-   mini-conda, mamba
-   Python versions 3.8, 3.9
-   Habitat simulator
-   Rtabmap-ros
-   Python packages: numpy, gym, pytorch, etc.

## Installation (recommended)

Please refer to [Installation_py38](Installation_py38.md) or [Installtion_py39](Installation_py39.md) for detailed installation instructions. We recommend to first try install with [Installtion_py39](Installation_py39.md), which requires compiling
habitat simulator from source, but with latest rtabmap_ros(0.20.18). If you have any installation issues, please follow [Installation_py38](Installation_py38.md) instead, which uses older version of rtabmap_ros (0.20.14) but with fewer problems.

## Data preparation

### Prepare Gibson dataset

You need to process semantic scene graph annotations to load gibson dataset. Refer to https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#gibson-and-3dscenegraph-datasets for more details.

1. Download semantic scene graph annotations from Stanford [3DSceneGraph](https://github.com/StanfordVL/3DSceneGraph) repository. Unzip the tiny and medium split zip files.

2. Convert to habitat-compatible formats

```bash
# you need to clone habitat-sim repo to run following scripts
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
git checkout v0.2.1

# run conversion script
tools/gen_gibson_semantics.sh /path/to/3DSceneGraph_medium/automated_graph /path/to/habitat_data/scene_datasets/gibson  /path/to/habitat_data/scene_datasets/gibson

tools/gen_gibson_semantics.sh /path/to/3DSceneGraph_tiny/verified_graph /path/to/habitat_data/scene_datasets/gibson  /path/to/habitat_data/scene_datasets/gibson
```

## Run evaluation

We assume you follow our given installation instructions to install ROS and python dependencies over RoboStack environment. You need to activate RoboStack environment to run this repository.

```bash
# assume your environment name is ros_env and ros workspace is ~/catkin_ws
conda activate ros_env
cd ~/catkin_ws
source devel.sh
```

### Evaluate frontier+2D detection baseline

```bash
roslaunch habitat_ros eval_frontier_2d_detect.launch
```

### Evaluate scene graph navigation

```bash
roslaunch habitat_ros eval_sgnav.launch
```

## Extra documentations

### TF tree in this module

[demo](./img/tf_tree.png)

### TODO LIST

-   [x] Load habitat configuration from yaml file instead of using hard-coded settings in [simulator.py](habitat/scripts/simulator.py)
-   [ ] Implement internal odometry to avoid visual odometry lost
-   [ ] Implement collision map tracking with considering map growing (controlled by rtabmap) to avoid being stuck by navmesh deficiency or map prediction error.
-   [ ] consider if we need to move agent & planning functions from habitat_ros to a new package
