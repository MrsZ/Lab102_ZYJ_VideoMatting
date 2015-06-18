#ifndef VIDEOMATTING_H
#define VIDEOMATTING_H
#include <opencv2\opencv.hpp>
#include "sharedmatting.h"
using namespace cv;
class VideoMatting{
public:
	Mat leftImage;
	Mat disparityMap;
	Mat outputAlphaImage;
	Mat outputPNGImage;
	VideoMatting();
	VideoMatting(Mat leftImage, Mat disparityMap, int canny_t1 = 10, int canny_t2 = 30);
	~VideoMatting();

	int run(Mat leftImage = Mat(), Mat disparityMap = Mat());

private:
	//Mat leftImage_in_process;
	//Mat disparityMap_in_process;
	Rect validArea;
	//Mat Contour;//contour of human. binary mat.

	int canny_t1;
	int canny_t2;

	int calculateGrayPeak(int &peak, int &loDiff, int &, int _th = 1);//Calculate Gray Peak in disparity map.
	int removeSmallComponent(Mat &image, int removeColor, int diviseur = 1000);
	int findSeedPoint(int &_x, int &_y, int peak);
	int reduceContour(Mat &_contour, int dSize = 5);
	int expansionContour(Mat &_contour, int dSize = 7);
	int calcContourFromDepth(Mat &contour_get_from_depth);
	void bgdTransparent();
	int generateImageForMatting(Mat &_image);

	int removeEdges(Mat &edges, const Mat &_contour);
	//int accurateContour(Mat &result, const Mat &edges, const Mat &_contour);
	int accurateContour2(Mat &result, const Mat &edges, const Mat &_contour);

	int removeSmallComponentForContour(Mat &contour);


	int removeSomeOutliers(Mat &_process_contour, const Mat &_left_image_edge);
	int removeMostOutliers(Mat &_process_contour);
	int removeAloner(Mat &_process_contour);
	int allChangeToWhite(Mat &_process_contour);
};





#endif