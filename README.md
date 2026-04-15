# ros_stereo_dso_S1

### Make and Install
The changes to adapt to Ubuntu 24.04 and Pangolin 0.9 have been demonstrated in the code.
Compilation order: 
1. src/stereo_dso
2. src/stereo_dso_ros
3. catkin_make

### Usage Example
1. ```rosbag play --pause ./260401_fly.bag```
2. ```rosrun stereo_dso_ros stereo_dso_ros calib=/home/zml/桌面/code/catkin_ws/src/stereo_dso_ros/examples/livox.txt preset=0 mode=1```
3. ```python rec.py```
4. Play rosbag

### Environments
1. Ubuntu 24.04
2. Pangolin v0.9
3. OpenCV 4.9.0