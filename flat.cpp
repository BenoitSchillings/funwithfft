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



//-----------------------------------------------------------

Mat GetFrame(VideoCapture c)
{
  Mat    t;
 
  c >> t;
 
  t.convertTo(t, CV_32F); 
  //normalize(t, t, 0, 1, NORM_MINMAX, CV_32F);
  resize(t, t, cvSize(0, 0), 1, 1, INTER_NEAREST); 
  return t;
}


//-----------------------------------------------------------


int main( int argc, char *argv[] ){
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
   
    Mat sum = mat.clone();
    sum = Scalar(0);
 
    for (int i = 0; i < nframes - 5; i++) {
	mat = GetFrame(cap);
	imshow("mat", mat);	
        sum = sum + mat;
	  
    };
    imshow("sum", sum/nframes);
    cvWaitKey(0);

    sum /= (nframes/10);
    //normalize(sum, sum, 0, 65535, NORM_MINMAX, CV_32F);
    sum.convertTo(sum, CV_16U);

    imwrite("./flat.pgm", sum);
}
