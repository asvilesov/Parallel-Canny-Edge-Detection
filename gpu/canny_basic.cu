

#include<string>
#include<map>
#include<iostream> 
#include<math.h>
#include<stdlib.h>
#include <cublas.h> //CUDA
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


using namespace cv;
using namespace std;

__device__ __managed__ int flag = 1; // 0 if unchanged, 1 if changed in the Hystersis+thresholding function

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
	int y_gradient[] = {-1,- 2, -1, 0, 0, 0, +1, +2, +1};

	//int y_gradient[] = {0,0,0, 0, 1, 0, 0, 0, 0};

	int x_mag, y_mag;

	*padding = 1;

	//remove the 0 multiplications... they are useless
	// x_mag = img[my_y*(*padding-1)+my_x+*padding-1]*x_gradient[0] +  img[my_y*(*padding-1)+my_x+*padding]*x_gradient[1] + img[my_y*(*padding-1)+my_x+*padding+1]*x_gradient[2] +  
	// 		img[my_y*(*padding)+my_x+*padding-1]*x_gradient[3] +  img[my_y*(*padding)+my_x+*padding]*x_gradient[4] + img[my_y*(*padding)+my_x+*padding+1]*x_gradient[5] + 
	// 		img[my_y*(*padding+1)+my_x+*padding-1]*x_gradient[6] +  img[my_y*(*padding+1)+my_x+*padding]*x_gradient[7] + img[my_y*(*padding+1)+my_x+*padding+1]*x_gradient[8];
	x_mag = img[(my_y-blockDim.x+2*(*padding))+my_x+*padding-1]*x_gradient[0] +  img[(my_y-blockDim.x+2*(*padding))+my_x+*padding]*x_gradient[1] + img[(my_y-blockDim.x+2*(*padding))+my_x+*padding+1]*x_gradient[2] +  
			img[my_y+my_x+*padding-1]*x_gradient[3] +  img[my_y+my_x+*padding]*x_gradient[4] + img[my_y+my_x+*padding+1]*x_gradient[5] + 
			img[(my_y+blockDim.x+2*(*padding))+my_x+*padding-1]*x_gradient[6] +  img[(my_y+blockDim.x+2*(*padding))+my_x+*padding]*x_gradient[7] + img[(my_y+blockDim.x+2*(*padding))+my_x+*padding+1]*x_gradient[8];

	y_mag = img[(my_y-blockDim.x+2*(*padding))+my_x+*padding-1]*y_gradient[0] +  img[(my_y-blockDim.x+2*(*padding))+my_x+*padding]*y_gradient[1] + img[(my_y-blockDim.x+2*(*padding))+my_x+*padding+1]*y_gradient[2] +  
			img[my_y+my_x+*padding-1]*y_gradient[3] +  img[my_y+my_x+*padding]*y_gradient[4] + img[my_y+my_x+*padding+1]*y_gradient[5] + 
			img[(my_y+blockDim.x+2*(*padding))+my_x+*padding-1]*y_gradient[6] +  img[(my_y+blockDim.x+2*(*padding))+my_x+*padding]*y_gradient[7] + img[(my_y+blockDim.x+2*(*padding))+my_x+*padding+1]*y_gradient[8];

	// if (int(sqrt(float(y_mag*y_mag))) > 255){
	// 	y_mag = 0;
	// }

	conv[(my_y) + my_x + *padding] = int(sqrt(float(y_mag*y_mag)+ float(x_mag*x_mag))/758*255); //CUDA only accpets floats/double in fp operations
	float phase_angle = atan2(float(y_mag), float(x_mag)) * 180 / (atan(1.0)*4);
	if ( phase_angle < 0){
		phase_angle += 180;
	}
	phase[(my_y) + my_x + *padding] = phase_angle;
}

//3x3 Kernel hard-coded with 1-sigma SD
__global__ void gaussian_filter(int *img, int *conv, int *padding){
	int my_x = threadIdx.x;
	int my_y = (blockIdx.x+*padding)*(blockDim.x+2*(*padding));
	
	float gauss[] = {0.077847,	0.123317,	0.077847,
					0.123317,	0.195346,	0.123317,
					0.077847,	0.123317,	0.077847};

	float gauss_val = 	img[(my_y-blockDim.x+2*(*padding))+my_x+*padding-1]*gauss[0] +  img[(my_y-blockDim.x+2*(*padding))+my_x+*padding]*gauss[1] + img[(my_y-blockDim.x+2*(*padding))+my_x+*padding+1]*gauss[2] +  
						img[my_y+my_x+*padding-1]*gauss[3] +  img[my_y+my_x+*padding]*gauss[4] + img[my_y+my_x+*padding+1]*gauss[5] + 
						img[(my_y+blockDim.x+2*(*padding))+my_x+*padding-1]*gauss[6] +  img[(my_y+blockDim.x+2*(*padding))+my_x+*padding]*gauss[7] + img[(my_y+blockDim.x+2*(*padding))+my_x+*padding+1]*gauss[8];

	conv[(my_y) + my_x + *padding] = int(gauss_val); 

}

__global__ void non_max_suppression(int *img, int *output, int *phase, int *padding){ //this might have an error, if it is changing in place... should be reading from steady_state image
	int my_x = threadIdx.x;
	int my_y = (blockIdx.x+*padding)*(blockDim.x+2*(*padding));

	int phase_to_pix[8][4] = {{0,1,0,-1} , {-1,1,1,-1}, {-1,1,1,-1}, {1,0,-1,0}, {1,0,-1,0}, {1,1,-1,-1}, {1,1,-1,-1}, {0,1,0,-1}}; //{(x1,y1),(x2,y2)} for each possible direction

	int compare_value = img[my_y+my_x+*padding];
	float i = phase[my_y+my_x+*padding];
	int val = int(i/22.5);

	if( (compare_value < img[(my_y+blockDim.x*phase_to_pix[val][1])+my_x+*padding + phase_to_pix[val][0]]) || (compare_value < img[(my_y+blockDim.x*phase_to_pix[val][2])+my_x+*padding + phase_to_pix[val][3]])){
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

	if(img[my_y+my_x+*padding] == 100){
		if( (img[(my_y-blockDim.x+2*(*padding))+my_x+*padding-1] == 255) || (img[(my_y-blockDim.x+2*(*padding))+my_x+*padding]== 255) || 
			(img[(my_y-blockDim.x+2*(*padding))+my_x+*padding+1] == 255) || (img[my_y+my_x+*padding-1] == 255) ||
			(img[(my_y+blockDim.x+2*(*padding))+my_x+*padding-1] == 255) || (img[my_y+my_x+*padding+1] == 255) ||
			(img[(my_y+blockDim.x+2*(*padding))+my_x+*padding] == 255) || (img[(my_y+blockDim.x+2*(*padding))+my_x+*padding+1]== 255)  ){

			img[my_y+my_x+*padding] = 255;
			flag = 1;
		}

		if( (img[(my_y-blockDim.x+2*(*padding))+my_x+*padding-1] == 0) || (img[(my_y-blockDim.x+2*(*padding))+my_x+*padding]== 0) || 
			(img[(my_y-blockDim.x+2*(*padding))+my_x+*padding+1] == 0) || (img[my_y+my_x+*padding-1] == 0) ||
			(img[(my_y+blockDim.x+2*(*padding))+my_x+*padding-1] == 0) || (img[my_y+my_x+*padding+1] == 0) ||
			(img[(my_y+blockDim.x+2*(*padding))+my_x+*padding] == 0) || (img[(my_y+blockDim.x+2*(*padding))+my_x+*padding+1]== 0)  ){

			img[my_y+my_x+*padding] = 0;
			flag = 1;
		}

	}
}




// }

struct pixel_angle{
	int pixel_loc[4];
};


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
	//Canny Edge Filter Parameters
	int high = 50;
	int low = 10;
	int *h_p = &high;
	int *l_p = &low;

	int *img = new int[height*width];
	int *conv_img = new int[height*width];
	int *phase_img = new int[height*width];

	int i = 0;
	for(i = 0; i < height*width; i++){
		img[i] = gray_img.at<uchar>(int(i/width), i%width);
	}

	//Kernel Setup
	int kernel_size = 3;
	int padd = kernel_size/2;
	int *kernel_p = &kernel_size;
	int *padd_p = &padd;
	printf("%i\n", *padd_p);

	//GPU setup
	dim3 dimGrid(height-2*padd);
	dim3 dimBlock(width-2*padd);

	int *gpu_img, *gpu_conv_img, *gpu_phase_img, *gpu_padd, *gpu_h, *gpu_w;
	cudaMalloc((void**)&gpu_img, sizeof(int)*height*width);
	cudaMalloc((void**)&gpu_conv_img, sizeof(int)*height*width);
	cudaMalloc((void**)&gpu_phase_img, sizeof(int)*height*width);
	cudaMalloc((void**)&gpu_padd, sizeof(int));
	cudaMalloc((void**)&gpu_h, sizeof(int));
	cudaMalloc((void**)&gpu_w, sizeof(int));
	// map<int,int*> gpu_angle_pix;
	// map<int,int*> angle_to_pixel_loc;
	// int NS[] {0,1,0,-1};
	// angle_to_pixel_loc[0] = NS;
	// cudaMalloc((void**)&gpu_angle_pix, sizeof(angle_to_pixel_loc));


	cudaMemcpy(gpu_img, img, sizeof(int)*height*width, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_conv_img, conv_img, sizeof(int)*height*width, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_phase_img, phase_img, sizeof(int)*height*width, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_padd, padd_p, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_h, h_p, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_w, l_p, sizeof(int), cudaMemcpyHostToDevice);

	//invoke Gauss Kernel
	gaussian_filter<<<dimGrid, dimBlock>>> (gpu_img, gpu_conv_img, gpu_padd);
	cudaMemcpy(conv_img, gpu_conv_img, sizeof(int)*height*width, cudaMemcpyDeviceToHost);

	// //invoke Sobel Kernel
	convolution_kernel<<<dimGrid, dimBlock>>> (gpu_conv_img, gpu_img, gpu_phase_img, gpu_h, gpu_w, gpu_padd);
	cudaMemcpy(conv_img, gpu_img, sizeof(int)*height*width, cudaMemcpyDeviceToHost);
	cudaMemcpy(phase_img, gpu_phase_img, sizeof(int)*height*width, cudaMemcpyDeviceToHost);

	// // // invoke non-max suppression
	int *gpu_new_img;
	cudaMalloc((void**)&gpu_new_img, sizeof(int)*height*width);
	cudaMemcpy(gpu_new_img, conv_img, sizeof(int)*height*width, cudaMemcpyHostToDevice);
	non_max_suppression<<<dimGrid, dimBlock>>> (gpu_img, gpu_new_img, gpu_phase_img, gpu_padd);
	cudaMemcpy(conv_img, gpu_new_img, sizeof(int)*height*width, cudaMemcpyDeviceToHost);

	// // // invoke thresholding
	thresholding<<<dimGrid, dimBlock>>> (gpu_new_img, gpu_padd, gpu_h, gpu_w);
	cudaMemcpy(conv_img, gpu_new_img, sizeof(int)*height*width, cudaMemcpyDeviceToHost);

	// invoke hysteresis
	int count = 0;
	while(flag == 1){
		count++;
		flag = 0;
		hystersis<<<dimGrid, dimBlock>>> (gpu_new_img, gpu_padd);
		cudaDeviceSynchronize();
		cudaMemcpy(conv_img, gpu_new_img, sizeof(int)*height*width, cudaMemcpyDeviceToHost);
	}

	printf("Flag is: %i and count is %i\n", flag, count);


	printf("%i\n", phase_img[1000]);






	//find max element
	printf("%i\n", *max_element(conv_img, conv_img + height*width));


	//printf("%i\n", conv_img[1]);


	namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Display window", gray_img);                   // Show our image inside it.

    //Mat conv_img_cv = Mat(1, width, CV_8U, conv_img, sizeof(int)*width);
    //memcpy(conv_img_cv.data, conv_img, height*width*sizeof(int));
    for(i = 0; i < height*width; i++){
		gray_img.at<uchar>(int(i/width), i%width) = conv_img[i];
	}
    namedWindow( "Convolution Image", WINDOW_AUTOSIZE);
    imshow("Convolution Image", gray_img);

    //printf("%s", type2str(conv_img.type()).c_str());

    waitKey(0);                                          // Wait for a keystroke in the window
	
    printf("Image size: %i %i \n", height, width);
	printf("%i", img_cv.at<Vec3b>(0,1)[1]);

	cudaError_t error = cudaGetLastError();
	printf("error: %s\n", cudaGetErrorString(error));


	delete[] img;
	delete[] conv_img;
	delete[] phase_img;
	cudaFree(gpu_img);
	cudaFree(gpu_conv_img);
	cudaFree(gpu_padd);
	cudaFree(gpu_h);
	cudaFree(gpu_w);
	cudaFree(gpu_phase_img);
	// cudaFree(gpu_angle_pix);

	return 0;
}