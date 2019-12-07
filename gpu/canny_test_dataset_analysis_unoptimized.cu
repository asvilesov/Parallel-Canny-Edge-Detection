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
void test_canny_img_dataset(String folderpath, string saveFolder) {     //void for now, idk what it should return or if we will just pass the data by calling another function

                                            // That allows us to test on different folders and not hardcode it
    vector<String> filenames;
    cv::glob(folderpath, filenames);

    float total_time = 0;

    float gauss_time = 0;
    float gradient_time = 0;
    float non_max_suppression_time = 0; 
    float thresholding_time = 0;
    float hystersis_time =0;
    int depth_bfs_tree = 0;
    int initial_tree_size = 0;
    int final_tree_size = 0;

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
        bool optimize = false;
        bool show_picture = false;
        bool debug = false;
        //printf("Total time of execution: %f\n", canny_edge_detector(grayscaleImage, optimize, show_picture, debug));
        //total_time += canny_edge_detector_benchmark(grayscaleImage, optimize, show_picture, debug); 
        total_time += canny_edge_detector_analysis(grayscaleImage, &gauss_time, &gradient_time, &non_max_suppression_time, &thresholding_time, &hystersis_time, &depth_bfs_tree, optimize, show_picture, debug);
        //

        int height = grayscaleImage.rows;
        int width = grayscaleImage.cols;
        for(int j = 0; j < height*width; j++){
            if(grayscaleImage.at<uchar>(int(j/width), j%width) == 255){ 
                final_tree_size++;
            }
        }

        //OpenCV Benchmark GPU Function // currently unoperational
        // Ptr<cv::cuda::CannyEdgeDetector> canny=cv::cuda::createCannyEdgeDetector(50,100);
        // cv::cuda::GpuMat edge;
        // cv::cuda::GpuMat src(grayscaleImage);
        // canny->detect(src,edge);

        //Uncomment if you would like to save grayscale canny images
        //imwrite(save, grayscaleImage); 
    }

    int fps = (int)1000/(total_time/filenames.size());
    int depth = (int)depth_bfs_tree/filenames.size();
    int final = (int)final_tree_size/filenames.size();
    int initial = (int)final_tree_size/filenames.size();

    printf("The average time to process the gaussian operation is: %f ms\n", gauss_time/filenames.size()); //Optimized 0.0190, 0.0288, 0.0512, 0.1279, (1024 doesnt work)
    printf("The average time to process the gradient operation is: %f ms\n", gradient_time/filenames.size());
    printf("The average time to process the non_max_suppression operation is: %f ms\n", non_max_suppression_time/filenames.size());
    printf("The average time to process the thresholding operation is: %f ms\n", thresholding_time/filenames.size());
    printf("The average time to process the hysteresis operation is: %f ms\n", hystersis_time/filenames.size());
    printf("The average depth of the BFS tree in hysteresis is: %i levels\n", depth); // 8,12,16,20,29
    printf("The average final size of the BFS tree in hysteresis is %i\n", final); // 248, 726, 1922, 5689, 25570 
    //printf("The average initial size of the BFS tree in hysteresis is %i\n", initial); // 158, 446, 1142, 3332, 14880
}

int main(){
    printf("Running a detailed analysis for stages of Canny Algorithm Unoptimized:\n");
    printf("\nFor 64x64 images:\n");
    test_canny_img_dataset("../images/64x64/*.jpg", "testing");
    printf("\nFor 128x128 images:\n");
    test_canny_img_dataset("../images/128x128/*.jpg", "testing");
    printf("\nFor 256x256 images:\n");
    test_canny_img_dataset("../images/256x256/*.jpg", "testing");
    printf("\nFor 512x512 images:\n");
    test_canny_img_dataset("../images/512x512/*.jpg", "testing");
    printf("\nFor 1024x1024 images:\n");
    test_canny_img_dataset("../images/1024x1024/*.jpg", "testing");
    printf("\nDone.\n\n");

}