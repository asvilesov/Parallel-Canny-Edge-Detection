#include "canny_algorithm.h"

int main(){

	//image setup
	//Mat img_cv = imread("test.png");
	Mat img_cv = imread("gg.jpg");
	Mat gray_img;
	cvtColor(img_cv, gray_img, CV_BGR2GRAY);

	bool optimize = false;
    bool show_picture = true;
    bool debug = false;


	printf("Total time of execution: %f\n", canny_edge_detector_benchmark(gray_img, optimize, show_picture, debug));
	return 0;
}