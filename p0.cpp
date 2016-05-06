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

using namespace std;
using namespace cv;


int main( int argc, char *argv[] ){
  
    char *name1 = argv[1];
    printf("%s\n", name1);
 
    VideoCapture cap;

    cap.open("pollen1.avi");

    Mat mat(480, 640, CV_8UC4);
    int nframes = cap.get(CV_CAP_PROP_FRAME_COUNT);

    printf("%d\n", nframes);
    do {
	cap.read(mat);
	imshow("frame", mat);
    } while(1);

}
