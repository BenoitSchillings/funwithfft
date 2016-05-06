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
  static Mat flat; 

  if (flat.size().height == 0) {
 	flat = imread("./flat.pgm", CV_LOAD_IMAGE_ANYDEPTH);
 	flat.convertTo(flat, CV_32F);
 	flat = flat / 1856.0;
  } 
 
  c >> t;
  cvtColor(t, t, COLOR_RGB2GRAY);
  t.convertTo(t, CV_32F); 
  //cout << t.depth() << " " <<  t.channels(); 
  t.convertTo(t, CV_32F); 
  flat.convertTo(flat, CV_32F); 
  //t = t.mul(flat);
  t = t - flat; 
  normalize(t, t, 0, 1, NORM_MINMAX, CV_32F);
  resize(t, t, cvSize(0, 0), 1.0, 1.0, INTER_NEAREST); 
  return t;
}

Mat xGetFrame(VideoCapture c)
{
  Mat    t;
  static Mat flat; 

  if (flat.size().height == 0) {
 	flat = imread("./flat.pgm", CV_LOAD_IMAGE_ANYDEPTH);
 	flat.convertTo(flat, CV_32F);
  } 
 
  c >> t;
  cvtColor(t, t, COLOR_RGB2GRAY);
  t.convertTo(t, CV_32F); 
  //cout << t.depth() << " " <<  t.channels(); 
  t.convertTo(t, CV_32F); 
  flat.convertTo(flat, CV_32F); 
  t = t.mul(flat);
  normalize(t, t, 0, 1, NORM_MINMAX, CV_32F);
  resize(t, t, cvSize(0, 0), 1.0, 1.0, INTER_NEAREST); 
 
  return t;
}

//-----------------------------------------------------------


Point match(Mat m0, Mat mat)
{
	Mat out;

	//imshow("m0", mat);	
	matchTemplate(mat, m0, out, CV_TM_SQDIFF);
	double min, max;
  	Point loc1, loc2;	
	cv::minMaxLoc(out, &min, &max, &loc1, &loc2);	
	
	//imshow("out", out/max);
	loc1.x += 0.5;
	loc1.y += 0.5;	

        loc1.x = loc1.x - out.size().width / 2;
        loc1.y = loc1.y - out.size().height / 2;

	return loc1;
        Mat c1 = mat.clone() - Scalar(0.5);
       
 
        dft(c1, c1, 0, c1.rows);

	Mat c2 = m0.clone()- Scalar(0.5);

	dft(c2, c2, 0, c2.rows);

	mulSpectrums(c1, c2, c2, 0);

	idft(c2, c2, DFT_SCALE, c1.rows + c2.rows - 1);

  	normalize(c2, c2, 0, 1, NORM_MINMAX, CV_32F);

	pow(c2, 3, c2);	
	imshow("dft", c2);
	
	cv::minMaxLoc(c2, &min, &max, &loc1, &loc2);

        loc1.x += 0.5;
        loc1.y += 0.5;

        //loc1.x = loc1.x - out.size().width / 2;
        //loc1.y = loc1.y - out.size().height / 2;

	return loc1;
}

//-----------------------------------------------------------


int main( int argc, char *argv[] ){
    int max_offset = 280;
    int cmp_width;
 
    char *name1 = argv[1];
 
    VideoCapture cap;
    VideoWriter out;
  
    cap.open(name1);

    int ex = static_cast<int>(cap.get(CV_CAP_PROP_FOURCC));
    out.open("out1.avi",
	     ex,
	     60,
	     Size(1280, 960),
	     false);
    char s[5];
    s[4] = 0;
    memcpy(s, &ex, 4);
    printf("%s\n", s); 
 
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
    cmp_width = sx/4; 
  
    Mat m0 = mat(Rect(300, 300, sx-600, sy-600)).clone();

    Mat sum = mat.clone();
    sum = Scalar(0);

    double t = clock();
    int cnt = 0;
 
    for (int i = 0; i < nframes - 5; i++) {

	mat = GetFrame(cap);

    	m0 = mat(Rect(300, 300, sx-600, sy-600)).clone();

	Point loc1 = match(m0, mat);
        
	//cout << loc1; 
	int dx = loc1.x;
	int dy = loc1.y;
 
	char skip = 0;
	
	if (abs(dx) > max_offset || abs(dy) > max_offset) {
		skip = 1;
	}

	if (dx <= -max_offset) dx = -max_offset;
	if (dy <= -max_offset) dy = -max_offset; 
	if (dx >= max_offset) dx = max_offset;
	if (dy >= max_offset) dy = max_offset;	
	
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
	
      
	cvWaitKey(1);
	
	if (skip == 0) {
		sum = sum + mx;
	}
        
	cnt++;
	printf("%d\n", i);	
	if (cnt == 25) {
		normalize(sum, sum, 0, 1, NORM_MINMAX, CV_32F);	
 		//m0 = sum(Rect(300, 300, sx-600, sy-600)).clone();

		imshow("m0", sum);
		cnt = 0;
		Mat tmp;

		normalize(sum, tmp, 0, 255, NORM_MINMAX, CV_32F);
		tmp.convertTo(tmp, CV_8U);
		//imshow("tmp",tmp);	
		out << tmp;
		sum = Scalar(0);
	}
    };
}
