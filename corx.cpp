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

int valpha = 100;
int bright = 150.0;

char *name1;
Mat input_file;
FILE *data;
//--------------------------------------------------------------------

Mat kernel;
float cur_fwhm;
int m = 180;

//--------------------------------------------------------------------

float calc_dev(Mat mat)
{
	int	x,y;
   	float	sum = 0;
 
	int w = mat.size().width;
    	int h = mat.size().height;

	for (y = 1; y < (h-1); y++) {
    		float* row_ptr1 = mat.ptr<float>(y);
		float* row_ptr2 = mat.ptr<float>(y+1);
	
                row_ptr1 += 1;
		row_ptr2 += 1;	
		for (x = 1; x < (w-1); x++) {
			float v0 = *row_ptr1;
			row_ptr1++;
			float v1 = *row_ptr1;
			float v2 = *row_ptr2++;
			sum += (v1-v0)*(v1-v0);
			sum += (v2-v0)*(v2-v0);
		}
	}
	return sum; 
}

//--------------------------------------------------------------------

char sub_psf(float ffwhm_x, float alpha)
{
    //if (cur_fwhm != ffwhm_x) { 
    	m = ffwhm_x * 9;	
	Mat kernelX = getGaussianKernel(m, ffwhm_x);
    	Mat kernelY = getGaussianKernel(m, ffwhm_x);
    	kernel = kernelX * kernelY.t(); 

    	Mat tmp;

    	kernel.convertTo(kernel, CV_32F);
	cur_fwhm = ffwhm_x;	
        pow(kernel, 2, kernel); 
        normalize(kernel, kernel, 0, 1, NORM_MINMAX, CV_32F);
 	//imshow("psf", kernel);	
	kernel *= alpha;
   //}
 

    int x,y;
    int w,h;
 
    w = input_file.size().width;
    h = input_file.size().height;

    //Mat tmp;

    matchTemplate(input_file, kernel, tmp, CV_TM_CCORR_NORMED);
    
    normalize(tmp, tmp, 0, 1, NORM_MINMAX, CV_32F);
    double min, max;
    Point loc1, loc2;
    cv::minMaxLoc(tmp, &min, &max, &loc1, &loc2);

    x= loc2.x + m/2;
    y= loc2.y + m/2;
    Mat tmpa = input_file(Rect(x-(m/2), y-(m/2), m, m));
    
    float dev0 = calc_dev(input_file);
	
    Mat i0 = input_file.clone();

    tmpa -= kernel;
    
    Scalar mean, stddev;
    float dev1 = calc_dev(input_file);
    
    if (dev1 > dev0) {
   	input_file = i0; 
   	return 0; 
    }
    else {
	cout << "better " << dev0 << " " <<  dev1 << " alpha = " << alpha << " fwhm= " << ffwhm_x << "\n";
   	return 1; 
    }
}



char opt_psf(float ffwhm_x, float alpha)
{
    	m = ffwhm_x * 7;	
	Mat kernelX = getGaussianKernel(m, ffwhm_x);
    	Mat kernelY = getGaussianKernel(m, ffwhm_x);
    	kernel = kernelX * kernelY.t(); 

    	Mat tmp;

    	kernel.convertTo(kernel, CV_32F);
	cur_fwhm = ffwhm_x;	
        //pow(kernel, 2, kernel); 
        normalize(kernel, kernel, 0, 1, NORM_MINMAX, CV_32F);
 

    int x,y;
    int w,h;
 
    w = input_file.size().width;
    h = input_file.size().height;

    int  bx = 0;
    int  by = 0;
    float bmul = 0; 
    float mdev = 0;
    float mul;
 
    for (int u = 0; u < 50000; u++) {
      
      if (u < 20000 || (bmul == 0)) {
      	x= (m) + rand()%(w-(m)*2);
      	y= (m) + rand()%(h-(m)*2);
      
        mul = (rand()%32000)/32000.0;
        mul += 0.02;
      } else {
	x = bx + -2 + rand()%5;
	y = by + -2 + rand()%5;
        mul = bmul + ((rand()%32768)/32768.0 - 0.5) * 0.05;	
        if (x < m) x = m;
	if (y < m) y = m;
	if (x> (w-m)) x = (w-m);
	if (y> (h-m)) y = (h-m); 
      } 

      float mul = (rand()%32000)/32000.0;
      
      //mul += 0.02;
      
      if (rand() & 1) mul = -mul; 
      Mat tmpa = input_file(Rect(x-(m/2), y-(m/2), m, m));
   
      float dev2 = calc_dev(tmpa);
      tmpa -= kernel * mul;
    
      float dev1 = calc_dev(tmpa);
      tmpa += kernel * mul;
 
      if (u%1220 == 0) 
      	waitKey(1); 
      
      if ((dev2-dev1) > mdev) {
	mdev = (dev2-dev1);
	bx = x;
	by = y;
 	bmul = mul;	
	//printf("better <%d> %d %d dev= %f, mul = %f\n", u, x, y, mdev, mul);
    }
    }
    
    if (mdev > 0.002) {
        Mat tmpa = input_file(Rect(bx-(m/2), by-(m/2), m, m));
        tmpa -= kernel * bmul;
    	
	fprintf(data, "%d,%d,%f,%f\n", bx, by, bmul, ffwhm_x);
	fflush(data);	
	return 1; 
     }
     return 0;
}


int run(){
    
    double ffwhm_x = 0.0 + fwhm_x / 100.0;
    double ffwhm_y = 0.0 + fwhm_y / 100.0;


    double psf = 20.0;
    double alpha = 0.2;
    
    Mat Clone = input_file.clone();
    float ratio; 
    for (int i =0 ; i < 10000000; i++) {
	int match = 0;
        float v0 = calc_dev(input_file(Rect(220, 220, input_file.size().width-440, input_file.size().height-440)));  
        do {	
	        psf = rand()%256;
		psf = psf / 256;
		psf = psf * psf;
		psf = 10.0 + psf * 30; 
		match = opt_psf(psf, alpha);	
                float v1 = calc_dev(input_file(Rect(220, 220, input_file.size().width-440, input_file.size().height-440))); 
		imshow("o1", input_file);
                Mat x = (Clone-input_file);
                normalize(x, x, 0, 1, NORM_MINMAX, CV_32F);

                imshow("diff", x);
                waitKey(1);
		ratio = v0/v1;	
		printf("ratio %f, %f-%f, fwhm=%f\n", ratio, v0, v1, psf);
		v0 = v1;	
	} while (match);
        //printf("reduce %f\n", psf); 
        //psf *= 0.85;	
   	if (psf < 2.5) psf = 2.5; 
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
  
    cvCreateTrackbar("alpha", name, &valpha, 10000, &on_trackbar );
    cvCreateTrackbar("bright", name, &bright, 1000, &on_trackbar );
    run();
}



int main( int argc, char *argv[] ){
    data = fopen("./log.txt", "w"); 
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
