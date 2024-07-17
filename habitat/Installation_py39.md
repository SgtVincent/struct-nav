## Installation with RoboStack

### Install miniconda and create environment 

You can follow the instructions below or refer to [Robostack Installation page](https://robostack.github.io/GettingStarted.html) for more details

1. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) or [miniforge](https://github.com/conda-forge/miniforge) from official site
2. Install mamba in your **base** environment. **DO NOT** install mamba in your robostack environment 
```bash
conda install mamba -n base -c conda-forge
```

Alternatively, you can install `mambaforge` from [here](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html), which not only contains `mamba` but also `conda`. Note that you cannot use `mambaforge`, `miniconda` or `anaconda` at the same time, since they will conflict with each other when calling `conda` command.

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

Thus, we recommend you to build habitat-sim and habitat-lab at first, and then install other dependencies.

Since the installation script provided in habitat-sim and habitat-lab will install them into the conda environment, where to place their source code is not important. You can clone them under the `struct_nav` root directory.

```bash
git clone --branch v0.2.1 https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim 
pip install -r requirements.txt    
python setup.py install --with-cuda # if not successfully built, use `./build.sh --with-cuda` instead
cd ..
git clone --branch v0.2.1 https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e .

# if you use ./build.sh to build habitat-sim, add the following line to your ~/.bashrc
export PYTHONPATH=$PYTHONPATH:/path/to/habitat-sim/

```

To test your habitat-sim installation, you can refer to [habitat-sim document](https://github.com/facebookresearch/habitat-sim/tree/v0.2.1?tab=readme-ov-file#testing)

To test your habitat-lab installation, you can run the following command:

```bash
cd habitat-lab
python examples/example.py
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

If you already have ros environment on your machine, please make sure you **DO NOT** source the ros environment variables like
```bash
source /opt/ros/noetic/setup.bash # you need to comment out this line
# and
source /path/to/your/native_ros_ws/devel/setup.bash # you need to comment out this line
```


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
We recommand you download generated Gibson semantic annotations (tiny_split) from [here](https://drive.google.com/file/d/1v71yumz-cRihiTGnW9hzVSU3JuMRp3Uy/view?usp=sharing), and extract the zip file together with the .glb files in the same directory.

From the official download link of Gibson dataset, you can download the Full partition of Gibson dataset in Habitat-sim format. And you don't need to convert it to tiny split in order to run our demo. The official dataset contains `.glb` files and `.navmesh` files for all the scenes.

The folder should look like this with `gibson` folder renamed to `gibson_semantic`:
```
habitat-lab
├── data
│   ├── scene_datasets
│       ├── gibson_semantic
│           ├── Markleeville.glb
|           ├── Markleeville.ids
│           ├── Markleeville.scn
│           ...
```


If you want to generate annotations on your local machine, please follow steps below: 

#### **Generate Gibson Semantic Annotation from Scratch**

1. You need to download Gibson_tiny split from [Gibson](https://github.com/StanfordVL/GibsonEnv) which contains `mesh.obj` file required for annotation extraction. 

2. To generate semantic annotations from [Gibson](https://github.com/StanfordVL/GibsonEnv) and [3DSceneGraph](https://3dscenegraph.stanford.edu/), you need datatool built from habitat-sim source code.

Since our pipeline is tested on conda-built habitat-sim package, we recommend you build habitat-sim with datatool in a temporary conda environment.

```bash
# conda activate temp_env
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim 
git checkout v0.2.1
python setup.py install --build-datatool
```

3. Follow the annotation preparation instructions in [habitat-sim document](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#gibson-and-3dscenegraph-datasets).


Finally, you need to download the gibson objectnav dataset from [here](https://github.com/devendrachaplot/Object-Goal-Navigation#downloading-episode-dataset) and extract it to `struct_nav/habitat/data/objectnav`.

The folder should look like this:
```
habitat
├── data
│   └── objectnav
│       └── gibson
│           └── v1.1
...         ...
```

## Trouble Shooting

### Conflicts of Qt and sip

There are some conflicts in dependencies of different packages. Fix conflicts by executing following commands in `ros_env` environment. If you meet unknown conflicts from Qt or sip, run following commands

```bash
pip uninstall PyQt5 PyQt5-sip
pip install PyQt5 PyQt5-sip
```

### Cannot launch rviz
If you use `rviz` to launch rviz and meet error like this:

```bash
rviz: symbol lookup error: /path/to/mambaforge/envs/env_name/bin/../lib/librviz.so: undefined symbol: _ZTIN4YAML13BadConversionE
```

You can add the following line to ~/.bashrc to fix it:

```bash
export LD_PRELOAD=/path/to/mambaforge/envs/env_name/lib/libyaml-cpp.so
```

### Cannot launch octomap_rviz_plugin

If you run the launch file and meet error like this:

```bash
[ERROR] [1721041341.131853005]: PluginlibFactory: The plugin for class 'octomap_rviz_plugin/ColorOccupancyGrid' failed to load.  Error: According to the loaded plugin descriptions the class octomap_rviz_plugin/ColorOccupancyGrid with base class type rviz::Display does not exist. Declared types are  rtabmap_ros/Info rtabmap_ros/MapCloud rtabmap_ros/MapGraph rviz/AccelStamped rviz/Axes rviz/Camera rviz/DepthCloud rviz/Effort rviz/FluidPressure rviz/Grid rviz/GridCells rviz/Illuminance rviz/Image rviz/InteractiveMarkers rviz/LaserScan rviz/Map rviz/Marker rviz/MarkerArray rviz/Odometry rviz/Path rviz/PointCloud rviz/PointCloud2 rviz/PointStamped rviz/Polygon rviz/Pose rviz/PoseArray rviz/PoseWithCovariance rviz/Range rviz/RelativeHumidity rviz/RobotModel rviz/TF rviz/Temperature rviz/TwistStamped rviz/WrenchStamped rviz_plugin_tutorials/Imu
```

try to build `octomap_rviz_plugins` from [source code](https://github.com/OctoMap/octomap_rviz_plugins) using catkin build.

### Deprecated APIs

Some of the APIs have been deprecated, if you meet errors, you may need to modify the source code of detectron2 and others. Please pay attention to the error description from the command line. For example, you may need to do the following modifications:

```bash
def __init__(self, src_rect, output_size, interp=Image.LINEAR, fill=0):
# Image.LINEAR -> Image.BILINEAR

def get_xyz_points(cloud_array, remove_nans=True, dtype=np.float):
# np.float -> np.float64
```

