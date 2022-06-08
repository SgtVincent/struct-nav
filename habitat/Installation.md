## Installation with RoboStack

### Install RoboStack environments

You can follow the instructions below or refer to [Robostack Installation page](https://robostack.github.io/GettingStarted.html) for more details

1. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) or [miniforge](https://github.com/conda-forge/miniforge) from official site
2. Install mamba in your **base** environment

```bash
conda install mamba -n base -c conda-forge
```

3. Create environment for ROS + conda

```bash
# now create a new environment
mamba create -n ros_env python=3.8
conda activate ros_env

# this adds the conda-forge channel to the new created environment configuration
conda config --env --add channels conda-forge
# and the robostack channels
conda config --env --add channels robostack
conda config --env --add channels robostack-experimental

# install ros-noetic
mamba install ros-noetic-desktop

# optional development packages
mamba install libgcc libstdcxx-ng compilers cmake pkg-config make ninja colcon-common-extensions catkin_tools rosdep

# ros utilities packages
mamba install ros-noetic-tf ros-noetic-tf2 ros-noetic-tf2-geometry-msgs ros-noetic-ros-numpy

# install rtabmap packages
mamba install ros-noetic-rtabmap=0.20.14 ros-noetic-rtabmap-ros=0.20.14
# Note that the latest version 0.20.18 not compiled with python=3.8x in robostack
# while habitat not released with python=3.9
```

For more available ROS packages in Robostack channel, you can refer to [this page](https://robostack.github.io/noetic.html).

Make sure you **DO NOT** install any ros packages with apt commands in the system, since this will lead to environment conflicts.

4. Install Habitat dependencies

```bash
mamba install habitat-sim withbullet headless -c conda-forge -c aihabitat
git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e .
```

5. Install other dependencies

```bash
# for example, install pytorch 1.9.0 with cuda 11.3
# mamba install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

6. Install detectron2 

Please see https://detectron2.readthedocs.io/en/latest/tutorials/install.html for install instructions. If you follow the same package versions in steps above to install environment, you could also execute following commands to install detectron2.

```bash
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
```

7. Build ROS packages
   TODO: add this denpendency to rosdep
   PyKDL: Since PyKDL installed by ros-noetic-tf2-geometry-msgs is outdated and depends on old releases of PyQt5-sip, sip packages, you should rebuild this module from source code:

```bash
cd ~/catkin_ws/src/struct-nav
git clone https://github.com/orocos/orocos_kinematics_dynamics.git
cd orocos_kinematics_dynamics
git submodule update --init
cd ~/catkin_ws
catkin build python_orocos_kdl
source devel/setup.sh
```

## Prepare datasets
If you want to use ground truth semantic sensor in habitat, you should follow the following instructions, otherwise semantic sensor will fail only with `.glb` file.

### Gibson 
https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#gibson-and-3dscenegraph-datasets

## Trouble Shooting

### Conflicts of Qt and sip

There are some conflicts in dependencies of different packages. Fix conflicts by executing following commands in `ros_env` environment. If you meet unknown conflicts from Qt or sip, run following commands

```bash
pip uninstall PyQt5 PyQt5-sip
pip install PyQt5 PyQt5-sip
```
