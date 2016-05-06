#include <stdio.h>
#include <cv.h>
#include <highgui.h>
#include <ctime>
#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <opencv2/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace std;
using namespace cv;


int main( int argc, char *argv[] ){
    int max_offset = 150;
    int cmp_width = 200;
 
    char *name1 = argv[1];
 
    VideoCapture cap;

    cap.open(name1);
    if(!cap.isOpened())  // check if we succeeded
        return -1;
    Mat mat;
    
    int nframes = cap.get(CV_CAP_PROP_FRAME_COUNT);
   
    for (int i = 0; i < 335; i++) { 
	cap >> mat;
    } 
    int sy, sx;
    
    sy = mat.size().height;
    sx = mat.size().width;
 
   
    Mat m0 = mat(Rect((sx/2) - cmp_width,
		     (sy/2) - cmp_width,
		     cmp_width * 2,
		     cmp_width * 2)); 
    Mat sum;
 
    m0.convertTo(m0, CV_32F); 
    
    normalize(m0, m0, 0, 1, NORM_MINMAX, CV_32F);
 
    for (int i = 0; i < nframes - 5; i++) {
	cap >> mat;
        mat.convertTo(mat, CV_32F); 
	
	normalize(mat, mat, 0, 1, NORM_MINMAX, CV_32F);

	Mat sm = mat;
	
	Mat out;
        matchTemplate(m0, mat, out, CV_TM_SQDIFF);
        double min, max;
  	Point loc1, loc2;	
	cv::minMaxLoc(out, &min, &max, &loc1, &loc2);	
        out -= min;
        out /= (max-min);
       
	int dx = loc1.x + 0.5;
	int dy = loc1.y + 0.5;
	dx = dx - out.size().width / 2;
	dy = dy - out.size().height / 2;
	if (dx <= -max_offset) dx = -max_offset;
	if (dy <= -max_offset) dy = -max_offset; 
	if (dx >= max_offset) dx = max_offset;
	if (dy >= max_offset) dy = max_offset;	
	printf("dx = %d, dy = %d\n", dx, dy);	
	
	copyMakeBorder(mat, mat,
                       max_offset,
                       max_offset,
                       max_offset,
                       max_offset,
                       BORDER_CONSTANT, Scalar::all(0.0));

	Mat mx =  mat(Rect(max_offset + dx,
			   max_offset + dy,
			   sx, sy)); 
	
	normalize(mx, mx, 0, 1, NORM_MINMAX, CV_32F);
	
        imshow("out", mx);	
        imshow("out1", out);	
	cvWaitKey(1);
	
/*
	if (sum.size().height == 0) {
		sum = mx;
	}
	else
		sum = sum + mx;
	imshow("sum", (sum) / (i + 1.0)); 
*/
 
 
    	m0 = sm(Rect((sx/2) - cmp_width,
                     (sy/2) - cmp_width,
                     cmp_width * 2,
                     cmp_width * 2));


    	normalize(m0, m0, 0, 1, NORM_MINMAX, CV_32F);

    };
    imshow("sum", sum/nframes);
    cvWaitKey(0);
}
