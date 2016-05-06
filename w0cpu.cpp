#include <stdio.h>
#include <cv.h>
#include <highgui.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <opencv2/cuda.hpp>
#include <opencv2/cudaarithm.hpp>

using namespace std;
using namespace cv;
//using namespace cv::cuda;


int fwhm = 100;
int fwhm1 = 100;
int alpha = 100;
int sub = 0.0;

char *name1;
char *name2;
Mat input_file;
Mat psf_file;

cuda::GpuMat gp;

int run(){
    
    double snr = (alpha+1.0) / 50000.0;
    double ffwhm = 1.0 + fwhm / 200.0;
    double ffwhm1 = 1.0 + fwhm1 / 200.0;
    
    Mat ImageR;
    Mat ImageI;
    
    int height = input_file.rows;
    int width = input_file.cols;
    ImageR = input_file;
    
    
    
    ImageR.convertTo(ImageR, CV_32F);
    ImageI = Mat::zeros(ImageR.size(), CV_32F);
    
    Mat rand_mat =  Mat::zeros(ImageR.size(), CV_32F);
    randn(rand_mat, 0, 1);
    //ImageR = ImageR + rand_mat;
    
    Mat im_planes[] = {ImageR, ImageI};
    Mat complexImage;
    
    merge(im_planes, 2, complexImage);
    dft(complexImage, complexImage, DFT_COMPLEX_OUTPUT, 0);
    split(complexImage, im_planes);

    
    Mat PSFR;
    Mat PSFI;


    PSFR = Mat::zeros(ImageR.size(), CV_32F);
   
 
    Mat kernelX = getGaussianKernel(30, ffwhm);
    Mat kernelY = getGaussianKernel(30, ffwhm);
    Mat kernel = kernelX * kernelY.t(); 

    kernelX = getGaussianKernel(30, ffwhm1);
    kernelY = getGaussianKernel(30, ffwhm1);
    Mat kernel1 = kernelX * kernelY.t(); 

    kernel = kernel + kernel1 * 0.1;
   
  
    int dy = height - 30;
    int dx = width - 30;

    copyMakeBorder(kernel, PSFR,
		   dy/2,
		   dy-(dy/2),
		   dx/2,
		   dx-(dx/2),
		   BORDER_CONSTANT, Scalar::all(0.0));

    Scalar psum = sum(PSFR);
    PSFR /= psum[0]; 
 
    PSFR.convertTo(PSFR, CV_32F);
    
    PSFI = Mat::zeros(PSFR.size(), CV_32F);
 
    Mat psf_planes[] = {PSFR, PSFI};
    
    Mat complexPSF;
    
    merge(psf_planes, 2, complexPSF);
    gp.upload(complexPSF);
    cuda::dft(gp, gp, complexPSF.size());
    gp.download(complexPSF); 
    split(complexPSF, psf_planes);
    
    Mat outR;
    Mat outI;
    
    Mat sum = PSFR.mul(PSFR) + PSFI.mul(PSFI) + snr;
    
    outR = ImageR.mul(PSFR) + ImageI.mul(PSFI);
    outI = ImageI.mul(PSFR) - ImageR.mul(PSFI);
    outR = outR / sum;
    outI = outI / sum;
    
    Mat out_planes[] = {outR, outI};
  

    merge(out_planes, 2, complexImage);
    gp.upload(complexImage);
    cuda::dft(gp, gp, complexImage.size(), DFT_INVERSE); 
    //dft(complexImage, complexImage, DFT_COMPLEX_OUTPUT + DFT_INVERSE + DFT_SCALE, 0 );
    gp.download(complexImage); 
    split(complexImage, out_planes);
 
    int cx = outR.cols/2;
    int cy = outR.rows/2;

    Mat q0(outR, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(outR, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(outR, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(outR, Rect(cx, cy, cx, cy)); // Bottom-Right

    Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);

    double min, max;
    
    minMaxLoc(outR, &min, &max);
    
    printf("%f %f\n", min, max);
    imshow("out", outR / (max/1.25));
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
    cvCreateTrackbar("fwhm", name, &fwhm, 1000, &on_trackbar );
    cvCreateTrackbar("fwhm1", name, &fwhm1, 1000, &on_trackbar );
 
    cvCreateTrackbar("alpha", name, &alpha, 10000, &on_trackbar );
    cvCreateTrackbar("sub", name, &sub, 1000, &on_trackbar );
    run();
}



int main( int argc, char *argv[] ){
  
    name1 = argv[1];
    name2 = argv[2];
    input_file = imread(name1, CV_LOAD_IMAGE_ANYDEPTH);
    //Rect myROI(0, 0, 384, 384); 
    //input_file = input_file(myROI); 
    psf_file = imread(name2, CV_LOAD_IMAGE_GRAYSCALE);
    imshow("Original", input_file);
    track();
    do {
        cvWaitKey(0);
    } while(1);
}
