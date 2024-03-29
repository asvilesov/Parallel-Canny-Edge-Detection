#include<string>
#include<map>
#include<iostream> 
#include<math.h>
#include<stdlib.h>
#include <cublas.h> //CUDA
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//for benchmarking
// #include <opencv2/core/cuda.hpp>
// #include <opencv2/imgproc.hpp>
// #include <opencv2/cudaimgproc.hpp>

// #include "opencv2\core\core.hpp"
// #include "opencv2\core\cuda.hpp"
// #include "opencv2\core\cuda\filters.hpp"
// #include "opencv2\cudaarithm.hpp"
// #include "opencv2\cudafilters.hpp"
// #include "opencv2\cudaimgproc.hpp"
// #include "opencv2\cudalegacy.hpp"


using namespace cv;
using namespace std;

__device__ __managed__ int flag = 1; // 0 if unchanged, 1 if changed in the Hystersis+thresholding function

//type2str function taken from openCV opensource code
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

__global__ void convolution_kernel(int *img, int *conv, int *phase, int *h, int *w, int *padding){
	int my_x = threadIdx.x;
	int my_y = (blockIdx.x+*padding)*(blockDim.x+2*(*padding));
	int x_gradient[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
	int y_gradient[] = {1,2,1, 0, 0, 0, -1, -2, -1};

	//int y_gradient[] = {0,0,0, 0, 1, 0, 0, 0, 0};

	int x_mag, y_mag;

	*padding = 1;

	//can remove the 0 multiplications... 
	x_mag = img[(my_y-blockDim.x-2*(*padding))+my_x+*padding-1]*x_gradient[0] +  img[(my_y-blockDim.x-2*(*padding))+my_x+*padding]*x_gradient[1] + img[(my_y-blockDim.x-2*(*padding))+my_x+*padding+1]*x_gradient[2] +  
			img[my_y+my_x+*padding-1]*x_gradient[3] +  img[my_y+my_x+*padding]*x_gradient[4] + img[my_y+my_x+*padding+1]*x_gradient[5] + 
			img[(my_y+blockDim.x+2*(*padding))+my_x+*padding-1]*x_gradient[6] +  img[(my_y+blockDim.x+2*(*padding))+my_x+*padding]*x_gradient[7] + img[(my_y+blockDim.x+2*(*padding))+my_x+*padding+1]*x_gradient[8];

	y_mag = img[(my_y-blockDim.x-2*(*padding))+my_x+*padding-1]*y_gradient[0] +  img[(my_y-blockDim.x-2*(*padding))+my_x+*padding]*y_gradient[1] + img[(my_y-blockDim.x-2*(*padding))+my_x+*padding+1]*y_gradient[2] +  
			img[my_y+my_x+*padding-1]*y_gradient[3] +  img[my_y+my_x+*padding]*y_gradient[4] + img[my_y+my_x+*padding+1]*y_gradient[5] + 
			img[(my_y+blockDim.x+2*(*padding))+my_x+*padding-1]*y_gradient[6] +  img[(my_y+blockDim.x+2*(*padding))+my_x+*padding]*y_gradient[7] + img[(my_y+blockDim.x+2*(*padding))+my_x+*padding+1]*y_gradient[8];

	conv[(my_y) + my_x + *padding] = int(sqrt(float(y_mag*y_mag)+ float(x_mag*x_mag))/758*255); //CUDA only accpets floats/double in fp operations
	float phase_angle = atan2(float(y_mag), float(x_mag)) * 180 / (atan(1.0)*4);
	if ( phase_angle < 0){
		phase_angle += 180;
	}
	phase[(my_y) + my_x + *padding] = phase_angle;
}

__global__ void optimized_convolution_filter(int *img, int *conv,  int *phase, int *h, int *w, int *padding){
	int my_x = threadIdx.x;
	int my_y = (blockIdx.x)*((*h/16)*blockDim.x);
	int x_gradient[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
	int y_gradient[] = {1,2,1, 0, 0, 0, -1, -2, -1};
	int x_mag, y_mag;

	__shared__ int s_img[3][2048]; //this is the largest dimension it can be, array parameters have to be static within a Kernel

	if(blockIdx.x == 0){
		s_img[0][threadIdx.x] = img[my_y+my_x];
		s_img[1][threadIdx.x] = img[my_y+blockDim.x+my_x];
		s_img[2][threadIdx.x] = img[my_y+2*blockDim.x+my_x];
	}
	else{
		s_img[0][threadIdx.x] = img[my_y-blockDim.x+my_x];
		s_img[1][threadIdx.x] = img[my_y+my_x];
		s_img[2][threadIdx.x] = img[my_y+blockDim.x+my_x];
	}
	
	if(blockIdx.x == 0){ //if first block
		for(int i = 1; i < (*h/16); i++){
			__syncthreads();
			if(threadIdx.x != 0 && threadIdx.x != blockDim.x -1){
				int x_mag = 	s_img[0][my_x-1]*x_gradient[0] +  s_img[1][my_x]*x_gradient[1] + s_img[2][my_x+1]*x_gradient[2] +  
								s_img[0][my_x-1]*x_gradient[3] +  s_img[1][my_x]*x_gradient[4] + s_img[2][my_x+1]*x_gradient[5] + 
								s_img[0][my_x-1]*x_gradient[6] +  s_img[1][my_x]*x_gradient[7] + s_img[2][my_x+1]*x_gradient[8];
				int y_mag = 	s_img[0][my_x-1]*y_gradient[0] +  s_img[1][my_x]*y_gradient[1] + s_img[2][my_x+1]*y_gradient[2] +  
								s_img[0][my_x-1]*y_gradient[3] +  s_img[1][my_x]*y_gradient[4] + s_img[2][my_x+1]*y_gradient[5] + 
								s_img[0][my_x-1]*y_gradient[6] +  s_img[1][my_x]*y_gradient[7] + s_img[2][my_x+1]*y_gradient[8];
				conv[my_y + (i)*blockDim.x + my_x] = int(sqrt(float(y_mag*y_mag)+ float(x_mag*x_mag))/758*255);
				float phase_angle = atan2(float(y_mag), float(x_mag)) * 180 / (atan(1.0)*4);
				if ( phase_angle < 0){
					phase_angle += 180;
				}
				phase[my_y + (i)*blockDim.x + my_x] = phase_angle;
			}
			__syncthreads();
			s_img[0][threadIdx.x] = s_img[1][threadIdx.x];
			s_img[1][threadIdx.x] = s_img[2][threadIdx.x];
			s_img[2][threadIdx.x] = img[my_y+(1+i)*blockDim.x+my_x]; //dependent
		}
	}
	else if(blockIdx.x == 15){ //if last block
		for(int i = 0; i < (*h/16)-1; i++){
			__syncthreads();
			if(threadIdx.x != 0 && threadIdx.x != blockDim.x -1){
				int x_mag = 	s_img[0][my_x-1]*x_gradient[0] +  s_img[1][my_x]*x_gradient[1] + s_img[2][my_x+1]*x_gradient[2] +  
								s_img[0][my_x-1]*x_gradient[3] +  s_img[1][my_x]*x_gradient[4] + s_img[2][my_x+1]*x_gradient[5] + 
								s_img[0][my_x-1]*x_gradient[6] +  s_img[1][my_x]*x_gradient[7] + s_img[2][my_x+1]*x_gradient[8];
				int y_mag = 	s_img[0][my_x-1]*y_gradient[0] +  s_img[1][my_x]*y_gradient[1] + s_img[2][my_x+1]*y_gradient[2] +  
								s_img[0][my_x-1]*y_gradient[3] +  s_img[1][my_x]*y_gradient[4] + s_img[2][my_x+1]*y_gradient[5] + 
								s_img[0][my_x-1]*y_gradient[6] +  s_img[1][my_x]*y_gradient[7] + s_img[2][my_x+1]*y_gradient[8];
				conv[my_y + (i)*blockDim.x + my_x] = int(sqrt(float(y_mag*y_mag)+ float(x_mag*x_mag))/758*255);
				float phase_angle = atan2(float(y_mag), float(x_mag)) * 180 / (atan(1.0)*4);
				if ( phase_angle < 0){
					phase_angle += 180;
				}
				phase[my_y + (i)*blockDim.x + my_x] = phase_angle;
			}
			__syncthreads();
			s_img[0][threadIdx.x] = s_img[1][threadIdx.x];
			s_img[1][threadIdx.x] = s_img[2][threadIdx.x];
			s_img[2][threadIdx.x] = img[my_y+(2+i)*blockDim.x+my_x]; //dependent
		}
	}
	else{ //if any other block
		for(int i = 0; i < (*h/16); i++){
			__syncthreads();
			if(threadIdx.x != 0 && threadIdx.x != blockDim.x -1){
				int x_mag = 	s_img[0][my_x-1]*x_gradient[0] +  s_img[1][my_x]*x_gradient[1] + s_img[2][my_x+1]*x_gradient[2] +  
								s_img[0][my_x-1]*x_gradient[3] +  s_img[1][my_x]*x_gradient[4] + s_img[2][my_x+1]*x_gradient[5] + 
								s_img[0][my_x-1]*x_gradient[6] +  s_img[1][my_x]*x_gradient[7] + s_img[2][my_x+1]*x_gradient[8];
				int y_mag = 	s_img[0][my_x-1]*y_gradient[0] +  s_img[1][my_x]*y_gradient[1] + s_img[2][my_x+1]*y_gradient[2] +  
								s_img[0][my_x-1]*y_gradient[3] +  s_img[1][my_x]*y_gradient[4] + s_img[2][my_x+1]*y_gradient[5] + 
								s_img[0][my_x-1]*y_gradient[6] +  s_img[1][my_x]*y_gradient[7] + s_img[2][my_x+1]*y_gradient[8];
				conv[my_y + (i)*blockDim.x + my_x] = int(sqrt(float(y_mag*y_mag)+ float(x_mag*x_mag))/758*255);
				float phase_angle = atan2(float(y_mag), float(x_mag)) * 180 / (atan(1.0)*4);
				if ( phase_angle < 0){
					phase_angle += 180;
				}
				phase[my_y + (i)*blockDim.x + my_x] = phase_angle;
			}
			__syncthreads();
			s_img[0][threadIdx.x] = s_img[1][threadIdx.x];
			s_img[1][threadIdx.x] = s_img[2][threadIdx.x];
			s_img[2][threadIdx.x] = img[my_y+(2+i)*blockDim.x+my_x]; //dependent
		}
	} 

}

//3x3 Kernel hard-coded with 1-sigma SD
__global__ void gaussian_filter(int *img, int *conv, int *padding){
	int my_x = threadIdx.x;
	int my_y = (blockIdx.x+*padding)*(blockDim.x+2*(*padding));
	
	//std 1
	float gauss[] = {0.077847,	0.123317,	0.077847,
					0.123317,	0.195346,	0.123317,
					0.077847,	0.123317,	0.077847};

	//std 2
	// float gauss[] = {0.102059,	0.115349,	0.102059,
	// 				0.115349,	0.130371,	0.115349,
	// 				0.102059,	0.115349,	0.102059};

	float gauss_val = 	img[(my_y-blockDim.x-2*(*padding))+my_x+*padding-1]*gauss[0] +  img[(my_y-blockDim.x-2*(*padding))+my_x+*padding]*gauss[1] + img[(my_y-blockDim.x-2*(*padding))+my_x+*padding+1]*gauss[2] +  
						img[my_y+my_x+*padding-1]*gauss[3] +  img[my_y+my_x+*padding]*gauss[4] + img[my_y+my_x+*padding+1]*gauss[5] + 
						img[(my_y+blockDim.x+2*(*padding))+my_x+*padding-1]*gauss[6] +  img[(my_y+blockDim.x+2*(*padding))+my_x+*padding]*gauss[7] + img[(my_y+blockDim.x+2*(*padding))+my_x+*padding+1]*gauss[8];

	conv[(my_y) + my_x + *padding] = int(gauss_val); 

}


__global__ void optimized_gaussian_filter(int *img, int *conv, int *h, int *w, int *padding){
	int my_x = threadIdx.x;
	int my_y = (blockIdx.x)*((*h/16)*blockDim.x);

	__shared__ int s_img[3][2048]; //this is the largest dimension it can be

	if(blockIdx.x == 0){
		s_img[0][threadIdx.x] = img[my_y+my_x];
		s_img[1][threadIdx.x] = img[my_y+blockDim.x+my_x];
		s_img[2][threadIdx.x] = img[my_y+2*blockDim.x+my_x];
	}
	else{
		s_img[0][threadIdx.x] = img[my_y-blockDim.x+my_x];
		s_img[1][threadIdx.x] = img[my_y+my_x];
		s_img[2][threadIdx.x] = img[my_y+blockDim.x+my_x];
	}

	//std 1
	float gauss[] = {0.077847,	0.123317,	0.077847,
					0.123317,	0.195346,	0.123317,
					0.077847,	0.123317,	0.077847};

	//std 2
	// float gauss[] = {0.102059,	0.115349,	0.102059,
	// 				0.115349,	0.130371,	0.115349,
	// 				0.102059,	0.115349,	0.102059};
	
	if(blockIdx.x == 0){ //if first block
		for(int i = 1; i < (*h/16); i++){
			__syncthreads();
			if(threadIdx.x != 0 && threadIdx.x != blockDim.x -1){
				float gauss_val = 	s_img[0][my_x-1]*gauss[0] +  s_img[1][my_x]*gauss[1] + s_img[2][my_x+1]*gauss[2] +  
								s_img[0][my_x-1]*gauss[3] +  s_img[1][my_x]*gauss[4] + s_img[2][my_x+1]*gauss[5] + 
								s_img[0][my_x-1]*gauss[6] +  s_img[1][my_x]*gauss[7] + s_img[2][my_x+1]*gauss[8];

				conv[my_y + (i)*blockDim.x + my_x] = int(gauss_val);
			}
			__syncthreads();
			s_img[0][threadIdx.x] = s_img[1][threadIdx.x];
			s_img[1][threadIdx.x] = s_img[2][threadIdx.x];
			s_img[2][threadIdx.x] = img[my_y+(1+i)*blockDim.x+my_x]; //dependent
		}
	}
	else if(blockIdx.x == 15){ //if last block
		for(int i = 0; i < (*h/16)-1; i++){
			__syncthreads();
			if(threadIdx.x != 0 && threadIdx.x != blockDim.x -1){
				float gauss_val = 	s_img[0][my_x-1]*gauss[0] +  s_img[1][my_x]*gauss[1] + s_img[2][my_x+1]*gauss[2] +  
								s_img[0][my_x-1]*gauss[3] +  s_img[1][my_x]*gauss[4] + s_img[2][my_x+1]*gauss[5] + 
								s_img[0][my_x-1]*gauss[6] +  s_img[1][my_x]*gauss[7] + s_img[2][my_x+1]*gauss[8];
				conv[my_y + (i)*blockDim.x + my_x] = int(gauss_val);
			}
			__syncthreads();
			s_img[0][threadIdx.x] = s_img[1][threadIdx.x];
			s_img[1][threadIdx.x] = s_img[2][threadIdx.x];
			s_img[2][threadIdx.x] = img[my_y+(2+i)*blockDim.x+my_x]; //dependent
		}
	}
	else{ //if any other block
		for(int i = 0; i < (*h/16); i++){
			__syncthreads();
			if(threadIdx.x != 0 && threadIdx.x != blockDim.x -1){
				float gauss_val = 	s_img[0][my_x-1]*gauss[0] +  s_img[1][my_x]*gauss[1] + s_img[2][my_x+1]*gauss[2] +  
								s_img[0][my_x-1]*gauss[3] +  s_img[1][my_x]*gauss[4] + s_img[2][my_x+1]*gauss[5] + 
								s_img[0][my_x-1]*gauss[6] +  s_img[1][my_x]*gauss[7] + s_img[2][my_x+1]*gauss[8];
				conv[my_y + (i)*blockDim.x + my_x] = int(gauss_val);
			}
			__syncthreads();
			s_img[0][threadIdx.x] = s_img[1][threadIdx.x];
			s_img[1][threadIdx.x] = s_img[2][threadIdx.x];
			s_img[2][threadIdx.x] = img[my_y+(2+i)*blockDim.x+my_x]; //dependent
		}
	} 

}

__global__ void non_max_suppression(int *img, int *output, int *phase, int *padding){ //this might have an error, if it is changing in place... should be reading from steady state image
	int my_x = threadIdx.x;
	int my_y = (blockIdx.x+*padding)*(blockDim.x+2*(*padding));

	int phase_to_pix[9][4] = {{0,1,0,-1} , {-1,1,1,-1}, {-1,1,1,-1}, {1,0,-1,0}, {1,0,-1,0}, {1,1,-1,-1}, {1,1,-1,-1}, {0,1,0,-1}, {0,1,0,-1}}; //{(x1,y1),(x2,y2)} for each possible direction

	int compare_value = img[my_y+my_x+*padding];
	float i = phase[my_y+my_x+*padding];
	int val = (int)(i/22.5);

	if( (compare_value < img[(my_y+(blockDim.x+2*(*padding))*phase_to_pix[val][1])+my_x+*padding + phase_to_pix[val][0]]) || (compare_value < img[(my_y+(blockDim.x+2*(*padding))*phase_to_pix[val][3])+my_x+*padding + phase_to_pix[val][2]])){
		output[my_y+my_x+*padding] = 0;
	}
}

__global__ void thresholding(int *img, int *padding, int *high, int *low){
	int my_x = threadIdx.x;
	int my_y = (blockIdx.x+*padding)*(blockDim.x+2*(*padding));

	if(img[my_y+my_x+*padding] > *high)
		img[my_y+my_x+*padding] = 255;
	else if(img[my_y+my_x+*padding] < *low)
		img[my_y+my_x+*padding] = 0;
	else
		img[my_y+my_x+*padding] = 100;
}

__global__ void hystersis(int*img, int*padding){
	int my_x = threadIdx.x;
	int my_y = (blockIdx.x+*padding)*(blockDim.x+2*(*padding));

	if(img[my_y+my_x+*padding] == 100){ //if weak
		if( (img[(my_y-blockDim.x-2*(*padding))+my_x+*padding-1] == 255) || (img[(my_y-blockDim.x-2*(*padding))+my_x+*padding]== 255) || 
			(img[(my_y-blockDim.x-2*(*padding))+my_x+*padding+1] == 255) || (img[my_y+my_x+*padding-1] == 255) ||
			(img[(my_y+blockDim.x+2*(*padding))+my_x+*padding-1] == 255) || (img[my_y+my_x+*padding+1] == 255) ||
			(img[(my_y+blockDim.x+2*(*padding))+my_x+*padding] == 255) || (img[(my_y+blockDim.x+2*(*padding))+my_x+*padding+1]== 255)  ){

			img[my_y+my_x+*padding] = 255; //if near strong, change pixel to strong too
			flag = 1;
		}

		//Switch commented code to have short hysteresis BFS search like in CPU code
		if( (img[(my_y-blockDim.x-2*(*padding))+my_x+*padding-1] == 0) && (img[(my_y-blockDim.x-2*(*padding))+my_x+*padding]== 0) && 
			(img[(my_y-blockDim.x-2*(*padding))+my_x+*padding+1] == 0) && (img[my_y+my_x+*padding-1] == 0) &&
			(img[(my_y+blockDim.x+2*(*padding))+my_x+*padding-1] == 0) && (img[my_y+my_x+*padding+1] == 0) &&
			(img[(my_y+blockDim.x+2*(*padding))+my_x+*padding] == 0) && (img[(my_y+blockDim.x+2*(*padding))+my_x+*padding+1]== 0)){
		// if( (img[(my_y-blockDim.x-2*(*padding))+my_x+*padding-1] == 0) || (img[(my_y-blockDim.x-2*(*padding))+my_x+*padding]== 0) || 
		// 	(img[(my_y-blockDim.x-2*(*padding))+my_x+*padding+1] == 0) || (img[my_y+my_x+*padding-1] == 0) ||
		// 	(img[(my_y+blockDim.x+2*(*padding))+my_x+*padding-1] == 0) || (img[my_y+my_x+*padding+1] == 0) ||
		// 	(img[(my_y+blockDim.x+2*(*padding))+my_x+*padding] == 0) || (img[(my_y+blockDim.x+2*(*padding))+my_x+*padding+1]== 0)){

			img[my_y+my_x+*padding] = 0;
			flag = 1;
		}

	}
}

__global__ void clean_up(int *img, int* padding){
	int my_x = threadIdx.x;
	int my_y = (blockIdx.x+*padding)*(blockDim.x+2*(*padding));

	if(img[my_y+my_x+*padding] == 100){ //if weak
		img[my_y+my_x+*padding] = 0;
	}
}

float canny_edge_detector_benchmark(Mat gray_img, bool optimize, bool show_pic, bool debug){
	int height = gray_img.rows;
	int width = gray_img.cols;
	int *height_p = &height;
	int *width_p = &width;
	//Canny Edge Filter Parameters
	int high = 70;
	int low = 30;
	int *h_p = &high;
	int *l_p = &low;

	int *img = new int[height*width];
	int *conv_img = new int[height*width];
	int *phase_img = new int[height*width];

	//Convert to 1D array
	int i = 0;
	for(i = 0; i < height*width; i++){
		img[i] = gray_img.at<uchar>(int(i/width), i%width);
	}

	//Kernel Setup
	int kernel_size = 3;
	int padd = (kernel_size-1)/2;
	int *kernel_p = &kernel_size;
	int *padd_p = &padd;

	//Malloc
	int *gpu_img, *gpu_conv_img, *gpu_phase_img, *gpu_padd, *gpu_h, *gpu_w, *gpu_height, *gpu_width;
	cudaMalloc((void**)&gpu_img, sizeof(int)*height*width);
	cudaMalloc((void**)&gpu_conv_img, sizeof(int)*height*width);
	cudaMalloc((void**)&gpu_phase_img, sizeof(int)*height*width);
	cudaMalloc((void**)&gpu_padd, sizeof(int));
	cudaMalloc((void**)&gpu_h, sizeof(int));
	cudaMalloc((void**)&gpu_w, sizeof(int));
	cudaMalloc((void**)&gpu_height, sizeof(int));
	cudaMalloc((void**)&gpu_width, sizeof(int));

	//Send to GPU
	cudaMemcpy(gpu_img, img, sizeof(int)*height*width, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_conv_img, img, sizeof(int)*height*width, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_phase_img, phase_img, sizeof(int)*height*width, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_padd, padd_p, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_h, h_p, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_w, l_p, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_height, height_p, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_width, width_p, sizeof(int), cudaMemcpyHostToDevice);

	//timing setup
	cudaEvent_t start, stop;
	float time_execute = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	//start timer
	cudaEventRecord(start, 0);

	//----------- Start Canny Algorithm -----------
	
	//GPU setup
	dim3 dimGrid2(height-2*padd);
	dim3 dimBlock2(width-2*padd);

	if(optimize){

		dim3 dimGrid(16);
		dim3 dimBlock(width);

		//invoke Gauss Kernel
		optimized_gaussian_filter<<<dimGrid, dimBlock>>>(gpu_conv_img, gpu_img, gpu_height, gpu_width, gpu_padd);
		//invoke Sobel Kernel
		optimized_convolution_filter<<<dimGrid, dimBlock>>> (gpu_conv_img, gpu_img, gpu_phase_img, gpu_height, gpu_width, gpu_padd);

	}
	else{
		//invoke Gauss Kernel
		gaussian_filter<<<dimGrid2, dimBlock2>>> (gpu_img, gpu_conv_img, gpu_padd);
		//invoke Sobel Kernel
		convolution_kernel<<<dimGrid2, dimBlock2>>> (gpu_conv_img, gpu_img, gpu_phase_img, gpu_h, gpu_w, gpu_padd);
	}
	
	cudaMemcpy(conv_img, gpu_img, sizeof(int)*height*width, cudaMemcpyDeviceToHost);

	// invoke non-max suppression
	int *gpu_new_img;
	cudaMalloc((void**)&gpu_new_img, sizeof(int)*height*width);
	cudaMemcpy(gpu_new_img, conv_img, sizeof(int)*height*width, cudaMemcpyHostToDevice);
	non_max_suppression<<<dimGrid2, dimBlock2>>> (gpu_img, gpu_new_img, gpu_phase_img, gpu_padd);

	// invoke thresholding
	thresholding<<<dimGrid2, dimBlock2>>> (gpu_new_img, gpu_padd, gpu_h, gpu_w);
	cudaMemcpy(conv_img, gpu_new_img, sizeof(int)*height*width, cudaMemcpyDeviceToHost);

	// invoke hysteresis
	int count = 0;
	flag = 1;
	while(flag == 1){
		count++;
		flag = 0;
		hystersis<<<dimGrid2, dimBlock2>>> (gpu_new_img, gpu_padd);
		cudaDeviceSynchronize();
	}

	// cleanup Image
	clean_up<<<dimGrid2, dimBlock2>>> (gpu_new_img, gpu_padd);
	cudaMemcpy(conv_img, gpu_new_img, sizeof(int)*height*width, cudaMemcpyDeviceToHost);

	//----------- End Canny Algorithm -----------

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time_execute, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	if(show_pic){
		namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
	    imshow( "Display window", gray_img);                   // Show our image inside it.

	    for(i = 0; i < height*width; i++){
			gray_img.at<uchar>(int(i/width), i%width) = conv_img[i];
		}

	    namedWindow( "Canny Image", WINDOW_AUTOSIZE);
	    imshow("Canny Image", gray_img);

	    waitKey(0); // Wait for a keystroke in the window
	}

	if(debug){
	    cudaError_t error = cudaGetLastError();
		printf("error: %s\n", cudaGetErrorString(error));
	}

    //Deallocate all memory
	delete[] img;
	delete[] conv_img;
	delete[] phase_img;
	cudaFree(gpu_img);
	cudaFree(gpu_conv_img);
	cudaFree(gpu_padd);
	cudaFree(gpu_h);
	cudaFree(gpu_w);
	cudaFree(gpu_phase_img);
	cudaFree(gpu_new_img);

	return time_execute;
}

float canny_edge_detector_analysis(Mat gray_img, float *gauss_time, float *gradient_time, float *non_max_suppression_time, float *thresholding_time, float *hystersis_time, int *depth_bfs, bool optimize, bool show_pic, bool debug){

	int height = gray_img.rows;
	int width = gray_img.cols;
	int *height_p = &height;
	int *width_p = &width;
	//Canny Edge Filter Parameters
	int high = 70;
	int low = 30;
	int *h_p = &high;
	int *l_p = &low;

	int *img = new int[height*width];
	int *conv_img = new int[height*width];
	int *phase_img = new int[height*width];

	//Convert to 1D array
	int i = 0;
	for(i = 0; i < height*width; i++){
		img[i] = gray_img.at<uchar>(int(i/width), i%width);
	}

	//Kernel Setup
	int kernel_size = 3;
	int padd = (kernel_size-1)/2;
	int *kernel_p = &kernel_size;
	int *padd_p = &padd;

	//Malloc
	int *gpu_img, *gpu_conv_img, *gpu_phase_img, *gpu_padd, *gpu_h, *gpu_w, *gpu_height, *gpu_width;
	cudaMalloc((void**)&gpu_img, sizeof(int)*height*width);
	cudaMalloc((void**)&gpu_conv_img, sizeof(int)*height*width);
	cudaMalloc((void**)&gpu_phase_img, sizeof(int)*height*width);
	cudaMalloc((void**)&gpu_padd, sizeof(int));
	cudaMalloc((void**)&gpu_h, sizeof(int));
	cudaMalloc((void**)&gpu_w, sizeof(int));
	cudaMalloc((void**)&gpu_height, sizeof(int));
	cudaMalloc((void**)&gpu_width, sizeof(int));

	//Send to GPU
	cudaMemcpy(gpu_img, img, sizeof(int)*height*width, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_conv_img, img, sizeof(int)*height*width, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_phase_img, phase_img, sizeof(int)*height*width, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_padd, padd_p, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_h, h_p, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_w, l_p, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_height, height_p, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_width, width_p, sizeof(int), cudaMemcpyHostToDevice);

	//timing setup
	cudaEvent_t start, stop;
	float time_execute = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//start timer
	cudaEventRecord(start, 0);

	//----------- Start Canny Algorithm -----------
	
	//GPU setup
	dim3 dimGrid2(height-2*padd);
	dim3 dimBlock2(width-2*padd);

	if(optimize){

		dim3 dimGrid(16);
		dim3 dimBlock(width);

		//invoke Gauss Kernel
		optimized_gaussian_filter<<<dimGrid, dimBlock>>>(gpu_conv_img, gpu_img, gpu_height, gpu_width, gpu_padd);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time_execute, start, stop);
		*gauss_time += time_execute;
		//invoke Sobel Kernel
		cudaEventRecord(start, 0);
		optimized_convolution_filter<<<dimGrid, dimBlock>>> (gpu_conv_img, gpu_img, gpu_phase_img, gpu_height, gpu_width, gpu_padd);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time_execute, start, stop);
		*gradient_time += time_execute;

	}
	else{
		//invoke Gauss Kernel
		gaussian_filter<<<dimGrid2, dimBlock2>>> (gpu_img, gpu_conv_img, gpu_padd);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time_execute, start, stop);
		*gauss_time += time_execute;
		//invoke Sobel Kernel
		cudaEventRecord(start, 0);
		convolution_kernel<<<dimGrid2, dimBlock2>>> (gpu_conv_img, gpu_img, gpu_phase_img, gpu_h, gpu_w, gpu_padd);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time_execute, start, stop);
		*gradient_time += time_execute;
	}
	
	cudaEventRecord(start, 0);
	cudaMemcpy(conv_img, gpu_img, sizeof(int)*height*width, cudaMemcpyDeviceToHost);

	// invoke non-max suppression
	int *gpu_new_img;
	cudaMalloc((void**)&gpu_new_img, sizeof(int)*height*width);
	cudaMemcpy(gpu_new_img, conv_img, sizeof(int)*height*width, cudaMemcpyHostToDevice);
	non_max_suppression<<<dimGrid2, dimBlock2>>> (gpu_img, gpu_new_img, gpu_phase_img, gpu_padd);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time_execute, start, stop);
	*non_max_suppression_time += time_execute;

	// invoke thresholding
	cudaEventRecord(start, 0);
	thresholding<<<dimGrid2, dimBlock2>>> (gpu_new_img, gpu_padd, gpu_h, gpu_w);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time_execute, start, stop);
	cudaMemcpy(conv_img, gpu_new_img, sizeof(int)*height*width, cudaMemcpyDeviceToHost);
	*thresholding_time += time_execute;

	// invoke hysteresis
	cudaEventRecord(start, 0);
	int count = 0;
	//printf("The flag is: %i\n", flag );
	flag = 1;
	while(flag == 1){
		count++;
		flag = 0;
		hystersis<<<dimGrid2, dimBlock2>>> (gpu_new_img, gpu_padd);
		cudaDeviceSynchronize();
	}
	*depth_bfs += count;
	//printf("The count for the hysteresis function is: %i\n", count);

	// cleanup Image
	clean_up<<<dimGrid2, dimBlock2>>> (gpu_new_img, gpu_padd);
	cudaMemcpy(conv_img, gpu_new_img, sizeof(int)*height*width, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time_execute, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	*hystersis_time += time_execute;

	//----------- End Canny Algorithm -----------

	if(show_pic){
		namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
	    imshow( "Display window", gray_img);                   // Show our image inside it.
	}
	
	for(i = 0; i < height*width; i++){
			gray_img.at<uchar>(int(i/width), i%width) = conv_img[i];
		}
	
	if(show_pic){
	    namedWindow( "Canny Image", WINDOW_AUTOSIZE);
	    imshow("Canny Image", gray_img);

	    waitKey(0); // Wait for a keystroke in the window
	}


	if(debug){
	    cudaError_t error = cudaGetLastError();
		printf("error: %s\n", cudaGetErrorString(error));
	}

    //Deallocate all memory
	delete[] img;
	delete[] conv_img;
	delete[] phase_img;
	cudaFree(gpu_img);
	cudaFree(gpu_conv_img);
	cudaFree(gpu_padd);
	cudaFree(gpu_h);
	cudaFree(gpu_w);
	cudaFree(gpu_phase_img);
	cudaFree(gpu_new_img);

	return time_execute;
}
