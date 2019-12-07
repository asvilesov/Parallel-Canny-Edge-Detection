# Parallel-Canny-Edge-Detection
CPU and GPU based parallelism 


Installation requirements and instructions:

sudo apt-get update

sudo apt-get upgrade

sudo apt-get install libopencv-dev


SERIAL compilation command

g++ serial.cpp -o output 'pkg-config --cflags --libs opencv' 





CPU code instructions






GPU code instructions

Navigate into gpu directoy: cd gpu

Compile: make

To show an image of what the result of the Canny algorithm is with GPU code execute: ./canny_show_img

To run total time benchmark for unoptimized Canny: ./canny_data_uo

To run total time benchmark for optimized Canny: ./canny_data_o

To run detailed timing analysis for various tages of unoptimized Canny: ./canny_test_analysis_uo

To run detailed timing analysis for various tages of optimized Canny: ./canny_test_analysis_o



WARNING:
A bug in the GPU code that has not been resolved is running optimized canny on 1024x1024 images in a dataset. 



