#!/bin/sh
rviz -d /home/junting/project_cvl/SceneGraphNav/vis_ros/config/default.rviz &
python -m vis_ros.vis_scene_graph_node

