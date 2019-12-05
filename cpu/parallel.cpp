#include<string>
#include<iostream> 
#include<stdio.h>
#include <iostream>       // cout, endl
#include <thread>         // this_thread::sleep_for
#include <chrono>         // chrono::seconds
// #include <cublas.h> //CUDA
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include <time.h>
#include <omp.h>


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
    for(int i = 0; i < k_size; ++i){
        for(int j = 0; j < k_size; ++j){
            gauss[i][j] = gauss[i][j] / sum;
        }
    }
    return;
}


void gaussian_blur(float **img, float **blur_img, float **gauss, int h, int w, int k_size){
    int padd = k_size/2;
    omp_set_num_threads(8); // Use 4 threads for all consecutive parallel regions
    #pragma omp parallel for
    for (int i=padd; i<h-padd; i++){
        for (int j=padd; j<w-padd; j++){
            for(int k_i=-padd; k_i<=padd; k_i++){
                for(int k_j=-padd; k_j<=padd; k_j++){
                    blur_img[i][j] += img[i+k_i][j+k_j]*gauss[k_i+padd][k_j+padd];
                }
            }
        }
    }
    return;
}


void sobel_filter(float **img, float **sobel_img, float **theta, int h, int w){
    int padd = 1; 
    float sobel_x[3][3] =
    {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    float sobel_y[3][3] =
    {
        {1, 2, 1},
        {0, 0, 0},
        {-1, -2, -1}
    };

    omp_set_num_threads(8); // Use 4 threads for all consecutive parallel regions
    #pragma omp parallel for
    for(int i=padd; i<h-padd; i++){
        for(int j=padd; j<w-padd; j++){
            float grad_x = 0;
            float grad_y = 0;
            for(int k_i=-padd; k_i<=padd; k_i++){
                for(int k_j=-padd; k_j<=padd; k_j++){
                    grad_x += img[i+k_i][j+k_j]*sobel_x[k_i+padd][k_j+padd]; 
                    grad_y += img[i+k_i][j+k_j]*sobel_y[k_i+padd][k_j+padd];
                }
            }
            sobel_img[i][j] = sqrt(pow(grad_x,2) + pow(grad_y,2));///758*255;
            theta[i][j] = atan2(grad_y, grad_x);
        }
    }
    //#sobel_img = sobel_img / sobel_img.max() * 255
    //return sobel_img, theta
    return;
}

void non_max_suppression(float **img, float **Z, float**theta, int h, int w){
    int padd = 1;
    omp_set_num_threads(8); // Use 4 threads for all consecutive parallel regions
    #pragma omp parallel for
    for(int i=padd; i<h-padd; i++){
        for(int j=padd; j<w-padd; j++){
            theta[i][j] = theta[i][j] * 180 / M_PI;
            if(theta[i][j] < 0){
                theta[i][j] += 180;
            } 

            float q = 255;
            float r = 255;
            if (0 <= theta[i][j] < 22.5 || 157.5 <= theta[i][j] <= 180){
                q = img[i][j+1];
                r = img[i][j-1];               
            }
            else if(22.5 <= theta[i][j] < 67.5){
                q = img[i+1][j-1];
                r = img[i-1][j+1];
            }
            else if(67.5 <= theta[i][j] < 112.5){
                q = img[i+1][j];
                r = img[i-1][j];
            }
            else if(112.5 <= theta[i][j] < 157.5){
                q = img[i-1][j-1];
                r = img[i+1][j+1];
            }

            if (img[i][j] >= q && img[i][j] >= r){
                Z[i][j] = img[i][j];
            }
            else{
                Z[i][j] = 0;
            }
        }
    }
    return;
}

void double_threshold(float **img, float **out_img, float**theta, int h, int w, float low_thres = 50, float high_thres = 100){
    int padd = 1;
    omp_set_num_threads(8); // Use 4 threads for all consecutive parallel regions
    #pragma omp parallel for
    for(int i=padd; i<h-padd; i++){
        for(int j=padd; j<w-padd; j++){
            if(img[i][j] >= high_thres){
                out_img[i][j] = high_thres;
            }
            else if(low_thres <= img[i][j] && img[i][j] <= high_thres){
                out_img[i][j] = low_thres;
            }
            else{
                out_img[i][j] = 0;
            }
        }
    }
    return;
}

void hysteresis(float **out_img, int h, int w, float low_thres = 50, float high_thres = 100){
    bool changed = true;
    while(changed){
        changed = false;
        int padd = 1;
        omp_set_num_threads(8); // Use 4 threads for all consecutive parallel regions
        #pragma omp parallel for shared(changed)
        for(int i=padd; i<h-padd; i++){
            for(int j=padd; j<w-padd; j++){
                if (out_img[i][j] == low_thres){
                    changed = true;
                    if ((out_img[i+1][j-1] == high_thres) || (out_img[i+1][j] == high_thres) || (out_img[i+1][j+1] == high_thres)
                        || (out_img[i][j-1] == high_thres) || (out_img[i][j+1] == high_thres)
                        || (out_img[i-1][j-1] == high_thres) || (out_img[i-1][j] == high_thres) || (out_img[i-1][j+1] == high_thres)){
                        out_img[i][j] = 255;
                    }
                    else{
                        out_img[i][j] = 0;
                    }
                }
            }
        }
    }
    return;
}


void visualize(float ** raw_img, int height, int width){
    // //create new img
    Mat img(height, width, CV_8U);
    uint8_t *myData = img.data;
 //    printf("Image size: %i %i \n", blurred_img.rows, blurred_img.cols);
 //    string ty2 =  type2str( blurred_img.type() );
    // cout << "type: " << ty2 << endl; 
    omp_set_num_threads(8); // Use 4 threads for all consecutive parallel regions
    #pragma omp parallel for
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            if(i==0 || j==0 || i==height-1 || j==width-1){
                myData[i*height + j] = 0;
            }
            myData[i*height + j] = (int) raw_img[i][j]; 
            //printf("%i\n", myData[i*height + j]);
        }
    }
    namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Display window", img);                   // Show our image inside it.
    waitKey(0);                                          // Wait for a keystroke in the window
    return;
}

int main(){


    //image setup
    Mat cv_img = imread("test.png");
    Mat gray_img;
    cvtColor(cv_img, gray_img, CV_BGR2GRAY);

    int height = cv_img.rows;
    int width = cv_img.cols;

    //Kernel Setup
    int kernel_size = 5;
    int padd = kernel_size/2;

    // namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
 //    imshow( "Display window", gray_img);                   // Show our image inside it.

 //    waitKey(0);                                          // Wait for a keystroke in the window
    
    printf("Image size: %i %i \n", height, width);
    string ty1 =  type2str( gray_img.type() );
    cout << "type: " << ty1 << endl;    
    // printf("%i \n", img.at<Vec3b>(1,1)[1]);

// create kernel
    float ** gauss = new float*[kernel_size];
    for(int i = 0; i < kernel_size; ++i){
        gauss[i] = new float[kernel_size];
    }
    gaussian_kernel(gauss, padd, 1.4);
    // for(int i = 0; i < kernel_size; ++i){
    //     for(int j = 0; j < kernel_size; ++j){
    //         printf("%f ", gauss[i][j]);
    //         // printf("gauss[%i][%i]=%f ", i, j, gauss[i][j]);
    //     }
    //     printf("\n");
    // }

//create 2D array
    float ** img = new float*[height];
    float **img2 = new float*[height];
    float **theta = new float*[height];

    for(int i = 0; i < height; ++i){
        img[i] = new float[width];
        img2[i] = new float[width];
        theta[i] = new float[width];
    }


    omp_set_num_threads(8); // Use 4 threads for all consecutive parallel regions
    #pragma omp parallel for
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            img[i][j] = gray_img.at<uchar>(i, j);
            img2[i][j] = 0;
        }
    }

    struct timespec start, stop; 
    double time;


    if( clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime");}

    gaussian_blur(img, img2, gauss, height, width, kernel_size);
//  visualize(img2, height, width);
    if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror("clock gettime");}   
    time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;    
    printf("Blur Execution time = %f sec\n", time);    
    
    if( clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime");}
    sobel_filter(img2, img, theta, height, width);
//  visualize(img, height, width);
    if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror("clock gettime");}   
    time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;    
    printf("Sobel Execution time = %f sec\n", time);    

    if( clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime");}
    non_max_suppression(img, img2, theta, height, width);
//  visualize(img2, height, width);
    if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror("clock gettime");}   
    time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;    
    printf("Non max Execution time = %f sec\n", time);    
    
    if( clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime");}
    double_threshold(img2, img, theta, height, width);
    hysteresis(img, height, width);
//  visualize(img, height, width);    
    if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror("clock gettime");}   
    time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;    
    printf("Double+hysteresis Execution time = %f sec\n", time);    


    omp_set_num_threads(8); // Use 4 threads for all consecutive parallel regions
    #pragma omp parallel for
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            img[i][j] = gray_img.at<uchar>(i, j);
            img2[i][j] = 0;
        }
    }


    // if( clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime");}
    // this_thread::sleep_for(chrono::seconds(1));
    // if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror("clock gettime");}   
    // time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;    
    // printf("Sanity check= %f sec\n", time);    

//  visualize(img, height, width);
//start timer
    if( clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime");}
    gaussian_blur(img, img2, gauss, height, width, kernel_size);
//  visualize(img2, height, width);
    sobel_filter(img2, img, theta, height, width);
//  visualize(img, height, width);
    non_max_suppression(img, img2, theta, height, width);
//  visualize(img2, height, width);
    double_threshold(img2, img, theta, height, width);
    hysteresis(img, height, width);
//end timer
    if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror("clock gettime");}   
    time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;    
    printf("Canny total Execution time = %f sec\n", time);        

    visualize(img, height, width);

    for(int i = 0; i < height; ++i){
        delete img[i];
        delete img2[i]; 
        delete theta[i];
    }

    delete img;
    delete img2;
    delete theta;

//delete kernel
    for(int i = 0; i < kernel_size; ++i){
        delete gauss[i];
    }
    delete gauss;


}