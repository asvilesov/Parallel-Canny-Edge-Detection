
#include<string>
#include<iostream> 
#include<stdlib.h>
#include <cublas.h> //CUDA
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "canny_algorithm.h"

using namespace std;
using namespace cv;

//Should take the folder path of the images and of the folder to save the grayscale photos into
void convertToOpenCV(String folderpath, string saveFolder) {		//void for now, idk what it should return or if we will just pass the data by calling another function

    folderpath = "../images/512x512/*.jpg";  // This is temporary, whatever calls the function should give the folder
                                            // That allows us to test on different folders and not hardcode it
    vector<String> filenames;
    cv::glob(folderpath, filenames);

    for (size_t i = 0; i < filenames.size(); i++)
    {
        //Read image in to program in color
        Mat im = imread(filenames[i]);//, 1;

        //Matrix to hold the images
        Mat grayscaleImage;
        //Convert to grayscale
        cvtColor(im, grayscaleImage, CV_BGR2GRAY);

        String save = saveFolder + "/" + filenames[i] + "_grey";

        //Can run function on the image here
        //printf("Total time of execution: %f\n", canny_edge_detector(gray_img));

        imwrite(save, grayscaleImage);
    }

}

int main(){

}