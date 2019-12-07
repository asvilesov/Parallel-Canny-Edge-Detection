#include<string>
#include<iostream> 
#include<stdlib.h>
#include <cublas.h> //CUDA
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


// #include "opencv2\core\cuda.hpp"
// #include "opencv2\core\cuda\filters.hpp"
// #include "opencv2\cudaarithm.hpp"
// #include "opencv2\cudafilters.hpp"
// #include "opencv2\cudaimgproc.hpp"
// #include "opencv2\cudalegacy.hpp"

#include "canny_algorithm.h"


using namespace std;
using namespace cv;

//Should take the folder path of the images and of the folder to save the grayscale photos into
void test_canny_img_dataset(String folderpath, string saveFolder) {		//void for now, idk what it should return or if we will just pass the data by calling another function

                                            // That allows us to test on different folders and not hardcode it
    vector<String> filenames;
    cv::glob(folderpath, filenames);

    float total_time = 0;

    for (size_t i = 0; i < filenames.size(); i++)
    { 
        //Read image in to program in color
        Mat im = imread(filenames[i]);//, 1;

        //Matrix to hold the images 
        Mat grayscaleImage;
        //Convert to grayscale
        cvtColor(im, grayscaleImage, CV_BGR2GRAY);

        String save = saveFolder + "/" + filenames[i] + "_grey";  

        //Our GPU function
        bool optimize = true;
        bool show_picture = false;
        bool debug = false;

        total_time += canny_edge_detector_benchmark(grayscaleImage, optimize, show_picture, debug); 

        //OpenCV Benchmark GPU Function // currently unoperational
        // Ptr<cv::cuda::CannyEdgeDetector> canny=cv::cuda::createCannyEdgeDetector(50,100);
        // cv::cuda::GpuMat edge;
        // cv::cuda::GpuMat src(grayscaleImage);
        // canny->detect(src,edge);

        //Uncomment if you would like to save grayscale canny images
        //imwrite(save, grayscaleImage); 
    }

    int fps = (int)1000/(total_time/filenames.size());

    printf("The average time to process the canny operation is: %f ms\n The average FPS is: %f\n", total_time/filenames.size(), 1000/(total_time/filenames.size()));

}

int main(){
    printf("Running Total Time Benchmark for Canny Algorithm Optimized:\n");
    printf("\nFor 64x64 images:\n");
    test_canny_img_dataset("../images/64x64/*.jpg", "testing");
    printf("\nFor 128x128 images:\n");
    test_canny_img_dataset("../images/128x128/*.jpg", "testing");
    printf("\nFor 256x256 images:\n");
    test_canny_img_dataset("../images/256x256/*.jpg", "testing");
    printf("\nFor 512x512 images:\n");
    test_canny_img_dataset("../images/512x512/*.jpg", "testing");
    printf("\nDone.\n\n");

}