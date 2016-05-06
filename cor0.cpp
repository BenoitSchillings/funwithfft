#include <stdio.h>
#include <cv.h>
#include <highgui.h>
#include <ctime>
#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/cudafilters.hpp"

#include <opencv2/cuda.hpp>
#include <opencv2/cudaarithm.hpp>

using namespace std;
using namespace cv;

Mat psf_file = imread("PSF.tif", CV_LOAD_IMAGE_ANYDEPTH);

int fwhm_x = 340;
int fwhm_y = 170;

int alpha = 100;
int bright = 150.0;

char *name1;
Mat input_file;



Point build_psf(float ffwhm_x)
{
    Mat kernelX = getGaussianKernel(180, ffwhm_x);
    Mat kernelY = getGaussianKernel(180, ffwhm_x);
    Mat kernel = kernelX * kernelY.t(); 

    Mat tmp;

    kernel.convertTo(kernel, CV_32F);	
  
    matchTemplate(input_file, kernel, tmp, CV_TM_CCORR_NORMED);

    //normalize(tmp, tmp, 0, 1, NORM_MINMAX, CV_32F);

    double min, max;
    Point loc1, loc2;
    cv::minMaxLoc(tmp, &min, &max, &loc1, &loc2);

    int x,y;

    x = loc2.x;
    y = loc2.y;
    
    Mat tmpb = tmp(Rect(x,y,1, 1));
    tmpb = Scalar(2);
    //imshow("cor", 2*tmp - 0.3);

    printf("%d %d %f\n", x, y, max);
    x += 90; y += 90;
    Mat tmpa = input_file(Rect(x-90, y-90, 180, 180));
    tmpa -= kernel * 13;
    return loc2;
    Mat k0;

    normalize(kernel, k0, 0, 1, NORM_MINMAX, CV_32F); 
    //imshow("kernel", k0); 
    //waitKey(1); 
    return loc2;
}


int run(){
    
    double ffwhm_x = 0.0 + fwhm_x / 100.0;
    double ffwhm_y = 0.0 + fwhm_y / 100.0;

    
    for (int i =0 ; i < 100000; i++) {
	build_psf(ffwhm_x * 2.0);
	if (i % 20 == 0) {
		//input_file = 1.0 - input_file;
	}	
   	if (i % 40 == 0) {
		imshow("o1", input_file);
		waitKey(1);	
	} 
    } 
    return 0;
}

void on_trackbar(int v)
{
    run();
}


void track()
{
    char *name =  "param";
    cvNamedWindow(name, 1);
    cvResizeWindow(name, 500, 50);
    cvCreateTrackbar("fwhm_x", name, &fwhm_x, 1000, &on_trackbar );
    cvCreateTrackbar("fwhm_y", name, &fwhm_y, 1000, &on_trackbar );
  
    cvCreateTrackbar("alpha", name, &alpha, 10000, &on_trackbar );
    cvCreateTrackbar("bright", name, &bright, 1000, &on_trackbar );
    run();
}



int main( int argc, char *argv[] ){
  
    name1 = argv[1];
    input_file = imread(name1, CV_LOAD_IMAGE_ANYDEPTH);
    
    imshow("Original", input_file);
    input_file.convertTo(input_file, CV_32F); 
    normalize(input_file, input_file, 0, 1, NORM_MINMAX, CV_32F);
    //input_file = input_file(Rect(550, 550, 300, 300)).clone(); 
    copyMakeBorder(input_file, input_file,
                   10,
                   10,
                   10, 
                   10,
                   BORDER_CONSTANT, Scalar::all(0.0));


    track();
    do {
        cvWaitKey(0);
    } while(1);
}
