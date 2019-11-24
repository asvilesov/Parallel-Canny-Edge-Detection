

#include<string>
#include<iostream> 
#include<stdlib.h>
#include <cublas.h> //CUDA
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;




int main(){

	//timing setup
	// cudaEvent_t start, stop;
	// float time_execute = 0;
	// cudaEventCreate(&start);
	// cudaEventCreate(&stop);

	//image setup
	Mat img = imread("test.png");
	Mat gray_img;
	cvtColor(img, gray_img, CV_BGR2GRAY);

	int height = img.rows;
	int width = img.cols;

	int *gpu_img = new int[height*width];

	int i = 0;
	for(i = 0; i < height*width; i++){
		gpu_img[i] = gray_img.at<uchar>(int(i/width), i%width);
	}

	//Kernel Setup
	int kernel_size = 3;
	int padd = kernel_size/2;

	//GPU setup
	// dim3 dimGrid(height-2*padd);
	// dim3 dimBlock(width-2*padd);

	







	namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Display window", gray_img);                   // Show our image inside it.

    waitKey(0);                                          // Wait for a keystroke in the window
	
    printf("Image size: %i %i \n", height, width);
	printf("%i", img.at<Vec3b>(1,1)[1]);

	return 0;
}