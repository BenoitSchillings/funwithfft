#include <stdio.h>
#include <cv.h>
#include <highgui.h>
#include <ctime>
#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/photo.hpp>

using namespace std;
using namespace cv;

Mat psf_file = imread("PSF.tif", CV_LOAD_IMAGE_ANYDEPTH);
Mat bg;

int fwhm_x = 120;
int fwhm_y = 120;

int iter = 600;
int bright = 150.0;

char *name1;
Mat input_file;

char image_inited = 0;

Mat build_psf0(float ffwhm_x)
{
    Mat kernel;
	
    ffwhm_x *= 15;
    resize(psf_file, kernel, Size(0, 0), 1.0/ffwhm_x, 1.0/ffwhm_x, INTER_AREA);
 
    Mat tmp;
 
    imshow("kernel", kernel * 235);
    waitKey(1);
    kernel.convertTo(kernel, CV_32F);	
   
    return kernel; 
}

Mat build_psf(float ffwhm_x)
{
    Mat kernelX = getGaussianKernel(13, ffwhm_x);
    Mat kernelY = getGaussianKernel(13, ffwhm_x);
    Mat kernel = kernelX * kernelY.t(); 

    Mat tmp;

    kernel.convertTo(kernel, CV_32F);	
  
    return kernel;
}

char inited = 0;


Mat conv(Mat m1, Mat psf)
{
	Mat	tmp;

	filter2D(m1, tmp, -1 , psf , Point( -1, -1 ), 0, BORDER_DEFAULT );
	return tmp;
}

void denoise(Mat m)
{
    int     i;
    Mat	    origin;

    origin = m.clone();

    for (int k = 0; k < 3; k++) 
    for (i = 0; i < 100000000; i++) {
       	if (i % 5000000 == 0) {
		printf("%d\n", i);	
		imshow("out_denoise", m * 0.3);	
		cvWaitKey(1);	
	} 
	int x = 1 + rand() % (m.cols - 2);
   	int y = 1 + rand() % (m.rows - 2);

	float v1 = m.at<float>(y, x); 
   	int direction = rand() % 4;

	int x1, y1;

	if (direction == 0) {
		x1 = x;
		y1 = y + 1;
	}
	if (direction == 1) {
		x1 = x;
		y1 = y - 1;	
	} 
	if (direction == 2) {
		x1 = x + 1;
		y1 = y;	
	}
   	if (direction == 3) {
		x1 = x - 1;
		y1 = y;
	}

	float v2 = m.at<float>(y1, x1);

	float div = (20.0);
	float delta = (v2 - v1) / div;

	v2 = v2 - delta;
	v1 = v1 + delta; 
   	
	float dist1 = fabs(v1 / origin.at<float>(y, x));
	float dist2 = fabs(v2 / origin.at<float>(y1, x1));
	
	if (dist1 > 1.0) dist1 = 1.0/dist1;
	if (dist2 > 1.0) dist2 = 1.0/dist2;

	float max = 0.975;

	if (dist1 > max && dist2 > max) {
		m.at<float>(y1, x1) = v2;
		m.at<float>(y, x) = v1;	
	} 
  }
}

int run(){
    
    float fw = 1.0;
 
    double ffwhm_x = 0.0 + fwhm_x / 100.0;
    int height = input_file.rows;
    int width = input_file.cols;
  
 
    double t;

    Mat psf = build_psf(ffwhm_x);
    Mat psfa = build_psf(1.2); 
    Mat model = input_file.clone();  
 
    Mat m0;
    Mat input0;
	
    input0 = input_file.clone();
    
    model = input0.clone();
 
   for (int i = 0; i < iter; i++) {
    	m0 = conv(input0/(conv(model, psf)), psfa);
	
	if (i == 130) {
		psf = build_psf(1.0);	
	}
	if (i == 250) {
		psf = build_psf(0.87);
	}	
	model = model.mul((m0 + Scalar(30.0))/31.0); 
  	printf("%d\n", i); 
   	imshow("out", model * 0.3);
        cvWaitKey(1);
	
	input0 = model;	
    }
   
    char buf[512];
    sprintf(buf, "./tmp/axout%d.png", time(0));
 
    imwrite(buf, model * 180); 
 
    imshow("out", model * 0.3);
    
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
  
    cvCreateTrackbar("iter", name, &iter, 500, &on_trackbar );
    cvCreateTrackbar("bright", name, &bright, 1000, &on_trackbar );
    
    run();
}






int main( int argc, char *argv[] ){
  
    cv::theRNG().state = time(0); 
    name1 = argv[1];
    input_file = imread(name1, CV_LOAD_IMAGE_ANYDEPTH);
    
    //imshow("Original", 60 * input_file);
    //moveWindow("Original", 100, 100);
    input_file.convertTo(input_file, CV_32F); 
    normalize(input_file, input_file, 0, 128.0, NORM_MINMAX, CV_32F);
    

    Mat cur;
    Mat f1;
    Mat input = input_file.clone();
 
    for (int i = 0; i < 10; i++) {
    	medianBlur(input, bg, 5); 
	input = bg;	
	cur = input_file - bg;
    	//threshold(cur, f1, 0, 32000, THRESH_TOZERO); 
    
    	//Mat over = (f1 - cur);   
       	//Mat over1; 
	//over1 = over.mul(over);	
	//over1 /= 20.0;	
	//input = input - over1;

	//imshow("input", input);
	//cvWaitKey(1);
    } 

    medianBlur(bg, f1, 5);
    bg = f1;

    //input_file = input_file - bg * 0.8;
    input_file = input_file / 60.0;
    input_file(cv::Rect(500,300,600,600)).copyTo(input_file);
    //bg(cv::Rect(900,400,600,600)).copyTo(bg);
    imshow("baseline", input_file * 0.3); 
    //denoise(input_file); 
    //input_file = input_file - bg * 0.8; 
    track(); 
    //denoise(input_file); 
    do {
        cvWaitKey(0);
    } while(1);


}
