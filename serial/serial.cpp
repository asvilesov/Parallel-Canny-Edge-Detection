#include<string>
#include<iostream> 
#include<stdlib.h>
// #include <cublas.h> //CUDA
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



	return 0;
}