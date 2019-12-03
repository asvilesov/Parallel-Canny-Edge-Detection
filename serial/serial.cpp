#include<string>
#include<iostream> 
#include<stdlib.h>
// #include <cublas.h> //CUDA
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>

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


void gaussian_kernel(float** gauss, int padd, float sigma){
    //x, y = np.mgrid[-size:size+1, -size:size+1]
	float coeff = 2 * pow(sigma, 2);
	int k_size = 1 + 2*padd;
	float sum = 0;
	// gauss = np.exp(-(x**2 + y**2) / coeff
    // technically not a true gaussian kernel, but sum(ker) == 1 is more important than closer approx. of gaus. dist. 
    for(int i = 0; i < k_size; ++i){
	    for(int j = 0; j < k_size; ++j){
	        gauss[i][j] = exp(-((i-padd)*(i-padd)+(j-padd)*(j-padd)) / coeff);
	        //printf("%f\n", gauss[i][j]);
	        sum += gauss[i][j];
	    }
	}
    //g /= np.abs(g).sum()
	for(size_t i = 0; i < k_size; ++i){
	    for(size_t j = 0; j < k_size; ++j){
			gauss[i][j] = gauss[i][j] / sum;
	    }
	}
    return;
}


int main(){


	//image setup
	Mat img = imread("test.png");
	Mat gray_img;
	cvtColor(img, gray_img, CV_BGR2GRAY);

	int height = img.rows;
	int width = img.cols;

	//Kernel Setup
	int kernel_size = 3;
	int padd = kernel_size/2;

	namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Display window", gray_img);                   // Show our image inside it.

    waitKey(0);                                          // Wait for a keystroke in the window
	
    printf("Image size: %i %i \n", height, width);
    string ty1 =  type2str( gray_img.type() );
	cout << "type: " << ty1 << endl;	
	// printf("%i \n", img.at<Vec3b>(1,1)[1]);

	Mat blurred_img(height, width, CV_8U);

    printf("Image size: %i %i \n", blurred_img.rows, blurred_img.cols);
    string ty2 =  type2str( blurred_img.type() );
	cout << "type: " << ty2 << endl;	

    float ** gauss = new float*[kernel_size];
    for(size_t i = 0; i < kernel_size; ++i){
    	gauss[i] = new float[kernel_size];
    }

    gaussian_kernel(gauss, padd, 1.4);
    for(int i = 0; i < kernel_size; ++i){
	    for(int j = 0; j < kernel_size; ++j){
	    	printf("%f ", gauss[i][j]);
	    	// printf("gauss[%i][%i]=%f ", i, j, gauss[i][j]);
		}
	    printf("\n");
	}

	for(size_t i = 0; i < kernel_size; ++i){
		delete gauss[i];
	}
	delete gauss;


	return 0;
}