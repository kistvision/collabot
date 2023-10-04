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
git clone https://github.com/kistvision/collabot.git  # clone
````

# Execution
1. Open Terminal and execute roscore
````
roscore
````

2. Execute detect_book_state.py
   
   1. Change the weights file path    
     ````  
     # yolo parameters
       def parser_opt(self):
           parser = argparse.ArgumentParser()
           parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/only_0822/weights/best.pt', help='model path or triton URL')     
     ````
   2. Choose whether to use video or camera. If you use camera, write the camera number. If you use video, write the video path. 
     ````
       # Using Camera
       source = 0
     
       # Using video
       source = '/home/vision/catkin_ws/src/Collabot/datasets/data_0822/video/57.mp4'
     ````
     
   3. Execute detect_book_state.py
     ````
     cd yolov5
     python detect_book_state.py
     ````
      
