#include <stdio.h>
#include <cv.h>
#include <highgui.h>
#include <ctime>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <opencv2/cuda.hpp>
#include <opencv2/cudaarithm.hpp>

using namespace std;
using namespace cv;


int fwhm = 100;
int fwhm1 = 100;
int alpha = 100;
int sub = 0.0;

char *name1;
char *name2;
Mat input_file;
Mat psf_file;
Mat complexImage;

char image_inited = 0;

Mat ImageR;
Mat ImageI;

Mat PSFR;
Mat PSFI;

cuda::GpuMat gPSFR;
cuda::GpuMat gPSFI;


void init_image_fft()
{
    if (image_inited == 0) {
        image_inited = 1;
        ImageR = input_file;
        ImageR.convertTo(ImageR, CV_32F);
        ImageI = Mat::zeros(ImageR.size(), CV_32F);
        Mat im_planes[] = {ImageR, ImageI};
        Mat complexImage;
        merge(im_planes, 2, complexImage);
        dft(complexImage, complexImage, DFT_COMPLEX_OUTPUT, 0);
        split(complexImage, im_planes);
    }
}


void build_psf_fft(int width, int height, float ffwhm)
{
    PSFR = Mat::zeros(ImageR.size(), CV_32F);
   
 
    Mat kernelX = getGaussianKernel(30, ffwhm);
    Mat kernelY = getGaussianKernel(30, ffwhm);
    Mat kernel = kernelX * kernelY.t(); 

    int dy = height - 30;
    int dx = width - 30;

    copyMakeBorder(kernel, PSFR,
		   dy/2,
		   dy-(dy/2),
		   dx/2,
		   dx-(dx/2),
		   BORDER_CONSTANT, Scalar::all(0.0));

    Scalar psum = cv::sum(PSFR);
    PSFR /= psum[0]; 
 
    PSFR.convertTo(PSFR, CV_32F);
    
    PSFI = Mat::zeros(PSFR.size(), CV_32F);
 
    Mat psf_planes[] = {PSFR, PSFI};
   
    cuda::GpuMat psf_planes_gpu[] = {gPSFR, gPSFI};
 
    Mat complexPSF;
   
    cuda::GpuMat gpPSF;
 
    merge(psf_planes, 2, complexPSF);
    gpPSF.upload(complexPSF);
    cuda::dft(gpPSF, gpPSF, complexPSF.size());
    gpPSF.download(complexPSF); 
    split(complexPSF, psf_planes);
    split(gpPSF, psf_planes_gpu); 
}

int run(){
    
    double snr = (alpha+1.0) / 50000.0;
    double ffwhm = 1.0 + fwhm / 200.0;
    
    int height = input_file.rows;
    int width = input_file.cols;

    init_image_fft();
    
 
    double t;

    t = clock();
    
    build_psf_fft(width, height, ffwhm);

    Mat outR;
    Mat outI;
    
    Mat sum = PSFR.mul(PSFR) + PSFI.mul(PSFI) + snr;
    
    outR = ImageR.mul(PSFR) + ImageI.mul(PSFI);
    outI = ImageI.mul(PSFR) - ImageR.mul(PSFI);
    outR = outR / sum;
    outI = outI / sum;
    
    Mat out_planes[] = {outR, outI};
 
  

    cuda::GpuMat gpImage; 
    
    merge(out_planes, 2, complexImage);
    gpImage.upload(complexImage);
    cuda::dft(gpImage, gpImage, complexImage.size(), DFT_INVERSE); 
    gpImage.download(complexImage); 
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
  
    cv::minMaxLoc(outR, &min, &max);
    
    //printf("%f %f\n", min, max);
    imshow("out", outR / (max/1.25));
    printf("dt %f\n", (clock()-t)/(double)CLOCKS_PER_SEC);
  
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
    psf_file = imread(name2, CV_LOAD_IMAGE_GRAYSCALE);
    imshow("Original", input_file);
    track();
    do {
        cvWaitKey(0);
    } while(1);
}
