## Installation with RoboStack

### Install miniconda and create environment 

You can follow the instructions below or refer to [Robostack Installation page](https://robostack.github.io/GettingStarted.html) for more details

1. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) or [miniforge](https://github.com/conda-forge/miniforge) from official site
2. Install mamba in your **base** environment. **DO NOT** install mamba in your robostack environment 
```bash
conda install mamba -n base -c conda-forge
```

3. Create envrionment to install robostack 
```bash
# now create a new environment
mamba create -n ros_env python=3.9
conda activate ros_env

# this adds the conda-forge channel to the new created environment configuration
conda config --env --add channels conda-forge
# and the robostack channels
conda config --env --add channels robostack
conda config --env --add channels robostack-experimental
```


### Install habitat-sim and habitat-lab
Note that after installation of robostack dependencies, system C/C++ dependencies will be overwritten by 
robostack, which leads to conflicts when building habitat-sim from source. (Possibly caused by that cmake/make system default paths 
are redirected to conda environment paths, and this leads to cuda/opengl link problem.)

```bash
git clone --branch v0.2.1 https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim 
pip install -r requirements.txt    
python setup.py install --with-cuda
cd ..
git clone --branch v0.2.1 https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e .
```

### Install ros packages under Robostack framework

NOTE: The reason to use python 3.9 is that, Robostack has moved whole project to 
python=3.9, there are a lot of outdated packages with python=3.8. Specifically, 
there is only rtabmap_ros<=0.20.14 with python=3.8, which has performances drop compared to rtabmap_ros=0.20.18 with python=3.9.  

One problem with python 3.9 is that, habitat pre-built packages support python<=3.8, which
means that you have to build habitat packages from source. 

```bash
# install ros-noetic
mamba install ros-noetic-desktop

# optional development packages
mamba install libgcc libstdcxx-ng compilers cmake pkg-config make ninja colcon-common-extensions catkin_tools rosdep

# ros utilities packages
mamba install ros-noetic-tf ros-noetic-tf2 ros-noetic-tf2-geometry-msgs ros-noetic-ros-numpy

# install rtabmap packages
mamba install ros-noetic-rtabmap=0.20.18 ros-noetic-rtabmap-ros=0.20.18
# Note that the latest version 0.20.18 not compiled with python=3.8x in robostack
# while habitat not released with python=3.9
```

For more available ROS packages in Robostack channel, you can refer to [this page](https://robostack.github.io/noetic.html).

Make sure you **DO NOT** install any ros packages with apt commands in the system, since this will lead to environment conflicts.



### Install other python dependencies

```bash
# for example, install pytorch 1.9.0 with cuda 11.3
# mamba install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.3 -c pytorch -c conda-forge
cd ~/catkin_ws/src/struct-nav/habitat
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

# Please see https://detectron2.readthedocs.io/en/latest/tutorials/install.html for install instructions. If you follow the same package versions in steps above to install environment, you could also execute following commands to install detectron2.
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
```
### Build ROS packages

TODO: add this denpendency to rosdep
PyKDL: Since PyKDL installed by ros-noetic-tf2-geometry-msgs is outdated and depends on old releases of PyQt5-sip, sip packages, you should rebuild this module from source code:

```bash
cd ~/catkin_ws/src/struct-nav
git clone https://github.com/orocos/orocos_kinematics_dynamics.git
cd orocos_kinematics_dynamics
git submodule update --init
cd ~/catkin_ws
# catkin build python_orocos_kdl
catkin build
source devel/setup.sh
```

## Prepare datasets

If you want to use ground truth semantic sensor in habitat, you should follow the following instructions, otherwise semantic sensor will fail only with `.glb` file.

### Gibson 
We recommand you download generated Gibson semantic annotations from [here]().

If you want to generate annotations on your local machine, please follow steps below: 

#### **Generate Gibson Semantic Annotation from Scratch**

1. You need to download Gibson_tiny split from [Gibson](https://github.com/StanfordVL/GibsonEnv) which contains `mesh.obj` file required for annotation extraction. 

2. To generate semantic annotations from [Gibson](https://github.com/StanfordVL/GibsonEnv) and [3DSceneGraph](https://3dscenegraph.stanford.edu/), you need datatool built from habitat-sim source code.

Since our pipeline is tested on conda-built habitat-sim package, we recommand you build habitat-sim with datatool in a temporary conda environment.

```bash
# conda activate temp_env
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim 
git checkout v0.2.1
python setup.py install --build-datatool
```

3. Follow the annotation preparation instructions in [habitat-sim document](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#gibson-and-3dscenegraph-datasets).

## Trouble Shooting

### Conflicts of Qt and sip

There are some conflicts in dependencies of different packages. Fix conflicts by executing following commands in `ros_env` environment. If you meet unknown conflicts from Qt or sip, run following commands

```bash
pip uninstall PyQt5 PyQt5-sip
pip install PyQt5 PyQt5-sip
```
