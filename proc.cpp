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
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

//-----------------------------------------------------------

Mat GetFrame(VideoCapture c)
{
  Mat    t;
  static Mat flat; 
/*
  if (flat.size().height == 0) {
 	flat = imread("./flat.pgm", CV_LOAD_IMAGE_ANYDEPTH);
 	flat.convertTo(flat, CV_32F);
  	//flat /= 1300.0; 
 	flat += Scalar(10); 
  } 
*/ 
  c >> t;
  cvtColor(t, t, COLOR_RGB2GRAY);
  t.convertTo(t, CV_32F); 
  //cout << t.depth() << " " <<  t.channels(); 
  t.convertTo(t, CV_32F); 
  //flat.convertTo(flat, CV_32F); 
  //t = t / flat;
  normalize(t, t, 0, 1, NORM_MINMAX, CV_32F);
  resize(t, t, cvSize(0, 0), 1.0, 1.0, INTER_NEAREST); 

  //imshow("out", t); 
  return t;
}

//-----------------------------------------------------------


Point match(Mat m0, Mat mat)
{
	Mat out;
       
	matchTemplate(m0, mat, out, CV_TM_SQDIFF);
        double min, max;
  	Point loc1, loc2;	

	GaussianBlur(out, out, Size(17, 17), 3);

	cv::minMaxLoc(out, &min, &max, &loc1, &loc2);

        loc1.x += 0.5;
        loc1.y += 0.5;

        loc1.x = loc1.x - out.size().width / 2;
        loc1.y = loc1.y - out.size().height / 2;


	return loc1;
}

//-----------------------------------------------------------


int main( int argc, char *argv[] ){
    int max_offset = 70;
    int max_diff = 30; 
    int cmp_width = 180;
 
    char *name1 = argv[1];
 
    VideoCapture cap;

    cap.open(name1);
    if(!cap.isOpened())  // check if we succeeded
        return -1;
    Mat mat;
   
    int nframes = cap.get(CV_CAP_PROP_FRAME_COUNT);
   
    for (int i = 0; i < 5; i++) { 
	mat = GetFrame(cap);
    } 
    int sy, sx;
    
    sy = mat.size().height;
    sx = mat.size().width;
 
   
    Mat m0 = mat(Rect((sx/2) - cmp_width,
		     (sy/2) - cmp_width,
		     cmp_width * 2,
		     cmp_width * 2)); 
    
    Mat sum = mat.clone();
    sum = Scalar(0);

    float cnt = 0;
 
    for (int i = 0; i < nframes - 5; i++) {
	mat = GetFrame(cap);
	
	Point loc1 = match(m0, mat);
     
	int dx = loc1.x;
	int dy = loc1.y;
 
	char skip = 0;
	
	if (abs(dx) > max_diff || abs(dy) > max_diff) {
		skip = 1;
	}

	if (dx <= -max_offset) dx = -max_offset;
	if (dy <= -max_offset) dy = -max_offset; 
	if (dx >= max_offset) dx = max_offset;
	if (dy >= max_offset) dy = max_offset;	
	printf("dx = %d, dy = %d\n", dx, dy);	

	copyMakeBorder(mat, mat,
                       abs(dy),
                       abs(dy),
                       abs(dx),
                       abs(dx),
                       BORDER_CONSTANT, Scalar::all(0.0));

	Mat mx =  mat(Rect(abs(dx) + dx,
			   abs(dy) + dy,
			   sx, sy)); 
	
	normalize(mx, mx, 0, 1, NORM_MINMAX, CV_32F);
	
        imshow("out", mx);	
	
	cvWaitKey(1);
	
	if (skip == 0) {
		sum = sum + mx;
		cnt += 1;	
	}
     /* 
	Mat diff;

	diff = sum.clone();
	diff = diff / cnt;
	diff = diff - mx;  
	diff = diff.mul(diff);	
	normalize(diff, diff, 0, 1, NORM_MINMAX, CV_32F);	
	imshow("diff", diff); 
	*/	
	/* 
	m0 = sum(Rect((sx/2) - cmp_width,
		          (sy/2) - cmp_width,
		          cmp_width * 2,
		          cmp_width * 2)).clone(); 
	
	normalize(m0, m0, 0, 1, NORM_MINMAX, CV_32F);
   	imshow("m0", mat);
	*/  
    };
    imshow("sum", sum/nframes);
    cvWaitKey(0);

    normalize(sum, sum, 0, 65535, NORM_MINMAX, CV_32F);
    sum.convertTo(sum, CV_16U);

    imwrite("./out.pgm", sum);
}
