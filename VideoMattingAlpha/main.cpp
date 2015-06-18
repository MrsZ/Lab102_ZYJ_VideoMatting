#include <opencv2\opencv.hpp>
#include "VideoMatting.h"
#include "sharedmatting.h"
using namespace cv;

int main(){
	//setUseOptimized(true);
	//setNumThreads(8);
	Mat leftImage;
	Mat disparityMap;
	
	string head_file_name = "TestImage3_zyj//";
	
	int64 t0 = getTickCount();
	for (int i = 0; i < 5; i++)
	{
		//cout << "----------------" << i << "------------------" << endl;
		string fileName;
		stringstream ss;
		int readIndex = i;
		ss << head_file_name << readIndex << "leftremap.png";
		ss >> fileName;
		leftImage = imread(fileName);
		ss.clear();

		ss << head_file_name << readIndex << "disparity.png";
		ss >> fileName;
		disparityMap = imread(fileName, 0);
		ss.clear();
		VideoMatting videoMatting(leftImage, disparityMap, 10, 30);
		videoMatting.run();

		ss << head_file_name << readIndex << "foreground.png";
		ss >> fileName;
		ss.clear();
		imwrite(fileName, videoMatting.outputPNGImage);


		ss << head_file_name << readIndex << "alpha.png";
		ss >> fileName;
		ss.clear();
		imwrite(fileName, videoMatting.outputAlphaImage);

		leftImage.release();
		disparityMap.release();

		//if (i == 273)
		//{
		//	i = 16;
		//	cout << "In total, it takes " << (getTickCount() - t0) * 1000 / getTickFrequency() << "ms" << endl;
		//}
			
	}
	

	
}
