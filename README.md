# Parallel-Canny-Edge-Detection

### Installation requirements and instructions:
`sudo apt-get update ` </br> ` sudo apt-get upgrade ` </br> ` sudo apt-get install libopencv-dev`

##### CPU code instructions

To compile: ` make`   
To run serial program  `./output`   
To run parallel program `./p_output`



##### GPU code instructions

Navigate into gpu directory: `cd gpu`  
To compile:  ` make` </br>  
To show an image of what the result of the Canny algorithm is with GPU code execute: `./canny_show_img`  
To run total time benchmark for unoptimized Canny: `./canny_data_uo`  
To run total time benchmark for optimized Canny: `./canny_data_o`  
To run detailed timing analysis for various stages of unoptimized Canny: `./canny_test_analysis_uo`  
To run detailed timing analysis for various stages of optimized Canny: `./canny_test_analysis_o`  

##### NOTE:

The GPU code can only run for images that are of size less than or equal to  1024x1024  
The optimized GPU code will not have speedup if one is using more than 16 Streaming Multiprocessors.  
A bug in the GPU code that has not been resolved is running optimized canny on 1024x1024 images in a dataset. 



