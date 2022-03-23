# ROS publisher/subscriber for active SLAM

(1) Download [data](https://drive.google.com/file/d/1E20heV-kLsD7ZiplBOBfsbxmjYG-tHN4/view?usp=sharing) and unzip data under `aslam` folder.

(2) Run rviz by using `roslaunch`:

```bash
$ roslaunch aslam run_rviz.launch
```

(3) Run publiser:

```bash
$ cd active-slam
$ python3 -m aslam.pub_rgbd_and_cloud
```

(4) Subscribe data:

```bash
$ cd active-slam
$ python3 -m aslam.sub_rgbd_and_cloud --base_dir $(rospack find aslam) --config_file aslam/config/rgbd_pub_config.yaml
```

# Dependencies

System: Ubuntu 18.04, ROS melodic, Python 2.7.

Main dependencies:

- opencv
- pcl (for ROS point cloud message type.)
- open3d (for creating point cloud from rgbd images.)

Installation commands:

- Open3D
  ```bash
  pip install open3d
  ```
