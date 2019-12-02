#include "canny_algorithm.h"

int main(){
	
	printf("hi\n");


	//image setup
	//Mat img_cv = imread("test.png");
	Mat img_cv = imread("4.jpg");
	Mat gray_img;
	cvtColor(img_cv, gray_img, CV_BGR2GRAY);


	printf("Total time of execution: %f\n", canny_edge_detector(gray_img));
	return 0;
}