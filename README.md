# Parallel-Canny-Edge-Detection
CPU and GPU based parallelism 


Installation requirements and instructions:

sudo apt-get update
sudo apt-get upgrade
sudo apt-get install libopencv-dev


serial compilation command
g++ serial.cpp -o output `pkg-config --cflags --libs opencv` 
