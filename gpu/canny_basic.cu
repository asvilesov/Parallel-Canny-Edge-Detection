

#include<string>
#include<iostream> 
#include<stdlib.h>
#include <cublas.h> //CUDA
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

__global__ void convolution_kernel(int *img, int *conv, int *h, int *w, int *padding){
	int my_x = threadIdx.x;
	int my_y = blockIdx.x*blockDim.x;
	int x_gradient[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
	int y_gradient[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

	int x_mag, y_mag;

	//remove the 0 multiplications... they are useless
	x_mag = img[my_y*(*padding-1)+my_x+*padding-1]*x_gradient[0] +  img[my_y*(*padding-1)+my_x+*padding]*x_gradient[1] + img[my_y*(*padding-1)+my_x+*padding+1]*x_gradient[2] +  
			img[my_y*(*padding)+my_x+*padding-1]*x_gradient[3] +  img[my_y*(*padding)+my_x+*padding]*x_gradient[4] + img[my_y*(*padding)+my_x+*padding+1]*x_gradient[5] + 
			img[my_y*(*padding+1)+my_x+*padding-1]*x_gradient[6] +  img[my_y*(*padding+1)+my_x+*padding]*x_gradient[7] + img[my_y*(*padding+1)+my_x+*padding+1]*x_gradient[8];

	y_mag = img[my_y*(*padding-1)+my_x+*padding-1]*y_gradient[0] +  img[my_y*(*padding-1)+my_x+*padding]*y_gradient[1] + img[my_y*(*padding-1)+my_x+*padding+1]*y_gradient[2] +  
			img[my_y*(*padding)+my_x+*padding-1]*y_gradient[3] +  img[my_y*(*padding)+my_x+*padding]*y_gradient[4] + img[my_y*(*padding)+my_x+*padding+1]*y_gradient[5] + 
			img[my_y*(*padding+1)+my_x+*padding-1]*y_gradient[6] +  img[my_y*(*padding+1)+my_x+*padding]*y_gradient[7] + img[my_y*(*padding+1)+my_x+*padding+1]*y_gradient[8];



	conv[my_y*(*padding) + my_x + *padding] = int(sqrt(double(x_mag*x_mag) + double(y_mag*y_mag))); //CUDA only accpets floats/double in fp operations
}



int main(){

	//timing setup
	cudaEvent_t start, stop;
	float time_execute = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//image setup
	Mat img_cv = imread("test.png");
	Mat gray_img;
	cvtColor(img_cv, gray_img, CV_BGR2GRAY);
	printf("%s", type2str(gray_img.type()).c_str());

	int height = img_cv.rows;
	int width = img_cv.cols;
	int *h_p = &height;
	int *w_p = &width;

	int *img = new int[height*width];
	int *conv_img = new int[height*width];

	int i = 0;
	for(i = 0; i < height*width; i++){
		img[i] = gray_img.at<uchar>(int(i/width), i%width);
	}

	printf("%i\n", conv_img[600]);

	//Kernel Setup
	int kernel_size = 3;
	int padd = kernel_size/2;
	int *kernel_p = &kernel_size;
	int *padd_p = &padd;

	//GPU setup
	dim3 dimGrid(height-2*padd);
	dim3 dimBlock(width-2*padd);

	int *gpu_img, *gpu_conv_img, *gpu_padd, *gpu_h, *gpu_w;
	cudaMalloc((void**)&gpu_img, sizeof(int)*height*width);
	cudaMalloc((void**)&gpu_conv_img, sizeof(int)*height*width);
	cudaMalloc((void**)&gpu_padd, sizeof(int));
	cudaMalloc((void**)&gpu_h, sizeof(int));
	cudaMalloc((void**)&gpu_w, sizeof(int));

	cudaMemcpy(gpu_img, img, sizeof(int)*height*width, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_conv_img, conv_img, sizeof(int)*height*width, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_padd, padd_p, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_h, h_p, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_w, w_p, sizeof(int), cudaMemcpyHostToDevice);

	//invoke Kernel
	convolution_kernel<<<dimGrid, dimBlock>>> (gpu_img, gpu_conv_img, gpu_h, gpu_w, gpu_padd);
	cudaMemcpy(conv_img, gpu_conv_img, sizeof(int)*height*width, cudaMemcpyDeviceToHost);


	printf("%i\n", conv_img[600]);


	namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Display window", gray_img);                   // Show our image inside it.

    Mat conv_img_cv = Mat(1, width, CV_8U, conv_img, sizeof(int)*width);
    //memcpy(conv_img_cv.data, conv_img, height*width*sizeof(int));
    namedWindow( "Convolution Image", WINDOW_AUTOSIZE);
    imshow("Convolution Image", conv_img_cv);

    printf("%s", type2str(conv_img_cv.type()).c_str());

    waitKey(0);                                          // Wait for a keystroke in the window
	
    printf("Image size: %i %i \n", height, width);
	printf("%i", img_cv.at<Vec3b>(1,1)[1]);

	cudaError_t error = cudaGetLastError();
	printf("error: %s\n", cudaGetErrorString(error));


	delete[] img;
	delete[] conv_img;
	cudaFree(gpu_img);
	cudaFree(gpu_conv_img);
	cudaFree(gpu_padd);
	cudaFree(gpu_h);
	cudaFree(gpu_w);

	return 0;
}