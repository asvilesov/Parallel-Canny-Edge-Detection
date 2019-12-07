# Parallel-Canny-Edge-Detection
CPU and GPU based parallelism 


Installation requirements and instructions:
`sudo apt-get update
sudo apt-get upgrade
sudo apt-get install libopencv-dev`

CPU code instructions

To compile: make

to run serial
`./output`

to run parallel
`./p_output`



GPU code instructions

Navigate into gpu directoy: cd gpu

To compile: make

To show an image of what the result of the Canny algorithm is with GPU code execute: `./canny_show_img`

To run total time benchmark for unoptimized Canny: `./canny_data_uo`

To run total time benchmark for optimized Canny: `./canny_data_o`

To run detailed timing analysis for various tages of unoptimized Canny: `./canny_test_analysis_uo`

To run detailed timing analysis for various tages of optimized Canny: `./canny_test_analysis_o`



WARNINGs:

The GPU code can only run for images that are of size equal to or less than 1024x1024

The optimized GPU code will not have speedup if one is using more than 16 Streaming Multiprocessors.

A bug in the GPU code that has not been resolved is running optimized canny on 1024x1024 images in a dataset. 



