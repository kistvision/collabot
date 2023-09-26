# Collabot
Objective: To detect if a person has taken a book or not. 

# Prerequisties and Environment
OS: Ubuntu 20.04

GPU: NVIDIA GeForce RTX 2070 SUPER

CUDA version: 11.1

ROS Noetic

# How to use this repository
1. Install anaconda and create your own virtual environment.
````
conda create -n collabot python=3.8.10
````

2. git clone YOLO v5 repository and install requirements.txt.
````
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
````

3. git clone this repository in your workspace
````
cd yolov5
https://github.com/kistvision/collabot.git  # clone
````

# Execution
1. Open Terminal and execute roscore
````
roscore
````

2. Execute detect_book_state.py


