#include "VideoMatting.h"
#include <vector>
#include <iostream>
#include "3x3_norm.h"

//#define SHOWIMAGE
//#define SHOWUSINGTIME
//#define JAC_DEBUG

using namespace std;
VideoMatting::VideoMatting(Mat leftImage, Mat disparityMap, int t1, int t2){
	this->leftImage = leftImage;
	this->disparityMap = disparityMap;
	this->canny_t1 = t1;
	this->canny_t2 = t2;
}
VideoMatting::~VideoMatting(){
	outputAlphaImage.release();
	leftImage.release();
	disparityMap.release();
	outputPNGImage.release();
}

int VideoMatting::calculateGrayPeak(int &peak, int &loDiff, int &upDiff, int times){
	int histSize = 255;
	Mat hist;
	float range[] = { 0, 255 };
	const float* histRange = { range };
	calcHist(&disparityMap, 1, 0, Mat(), hist, 1, &histSize, &histRange);
	int total = disparityMap.rows * disparityMap.cols;

	vector<uchar> peaks;
	vector<uchar> troughs;
	float* _data_hist = (float*)hist.ptr(0);
	total = total - (int)_data_hist[0];

	for (int i = 150; i > 4; i--)
	{
		if (_data_hist[i] > _data_hist[i - 1] && _data_hist[i] > _data_hist[i + 1])
		{
			if (_data_hist[i] < total / 100)
				continue;

			int interval_total = +_data_hist[i - 4] + _data_hist[i-3] + _data_hist[i - 2] + _data_hist[i - 1] + _data_hist[i] +
				_data_hist[i + 1] + _data_hist[i + 2] + _data_hist[i + 3] + _data_hist[i + 4];
			if (interval_total > total / 10)
			{
				//if (times > 1)
				//{
				//	times--;
				//	continue;
				//}
				peak = i;
				break;
			}
		}
	}

	//for (int _upDiff = 0; ; _upDiff++)
	//{
	//	int index = peak + _upDiff;
	//	if (_data_hist[index] < 100 && _data_hist[index + 1] < 100 && _data_hist[index+2] < 100)
	//	{
	//		upDiff = _upDiff+2;
	//		break;
	//	}
	//}
	upDiff = 50;


	//find second peak.
	int second_peak = 0;
	bool concesstion = true;
	for (int i = peak - 4; i > 4; i--)
	{
		if (concesstion)
		{
			int interval_total = +_data_hist[i - 4] + _data_hist[i - 3] + _data_hist[i - 2] + _data_hist[i - 1] + _data_hist[i] +
				_data_hist[i + 1] + _data_hist[i + 2] + _data_hist[i + 3] + _data_hist[i + 4];

			if (interval_total > total / 10)
			{
				continue;
			}
			else
				concesstion = false;

		}
		else
		{
			if (_data_hist[i] > _data_hist[i - 1] && _data_hist[i] > _data_hist[i + 1])
			{
				if (_data_hist[i] < total / 100)
					continue;

				int interval_total = +_data_hist[i - 4] + _data_hist[i - 3] + _data_hist[i - 2] + _data_hist[i - 1] + _data_hist[i] +
					_data_hist[i + 1] + _data_hist[i + 2] + _data_hist[i + 3] + _data_hist[i + 4];
				if (interval_total > total / 10)
				{
					second_peak = i;
					break;
				}
			}
		}
	}

	//if no second_peak
	int _loDiff = 0;
	if (second_peak == 0)
	{
		int one_peak_total = 0;
		for (int i = 0; i <= upDiff; i++)
		{
			one_peak_total += _data_hist[i + peak];
		}
		while (one_peak_total < 0.7 * total)
		{
			_loDiff++;
			one_peak_total += _data_hist[peak - _loDiff];
		}
		while (true)
		{
			_loDiff++;
			if (_data_hist[peak - _loDiff] < 100 && _data_hist[peak - _loDiff + 1] < 100 && _data_hist[peak - _loDiff + 2] < 100)
				break;
		}
		loDiff = _loDiff;
		return 1;
	}


	//if no second_peak
	float _temp = _data_hist[peak];
	for (int i = second_peak; i < peak; i++)
	{
		if (_data_hist[i] < _temp)
		{
			_temp = _data_hist[i];
			_loDiff = i;
		}
	}
	loDiff = peak - _loDiff + 3;

	return 1;
}

int VideoMatting::findSeedPoint(int &_x, int&_y, int peak){
	uchar* _data_imageGray = disparityMap.ptr<uchar>(0);
	int total = disparityMap.cols * disparityMap.rows;

	RNG rng(getTickCount());
	for (;;){
		int index = rng.uniform(0, total);
		if ((int)(_data_imageGray[index]) == peak)
		{
			_y = index / disparityMap.cols;
			_x = index % disparityMap.cols;
			break;
		}
	}

	return 1;
}

int VideoMatting::calcContourFromDepth(Mat &contour_get_from_depth){
	int total = disparityMap.cols * disparityMap.rows;
	Rect rect;
	int label_count = 256;
	Mat label_image;
	int _x = -1, _y = -1;

	int times = 1;
	int peak, loDiff, upDiff;
	for (; rect.area() < total / 10;)
	{
		calculateGrayPeak(peak, loDiff, upDiff, times++);
		int have_tried = 0;
		do{
			have_tried++;
			findSeedPoint(_x, _y, peak);
			disparityMap.convertTo(label_image, CV_32SC1);
			floodFill(label_image, Point(_x, _y), label_count, &rect, loDiff, upDiff);
			if (have_tried > 20)
				break;
		} while (rect.area() < total / 10);
	}
	Mat _Contour = Mat(disparityMap.rows, disparityMap.cols, CV_8UC1, Scalar::all(0));
	for (int _row = 0; _row < rect.height; _row++)
	{
		int _index_row = rect.y + _row;
		int* _row_label_data = label_image.ptr<int>(_index_row);
		uchar* _row_disparity_data = disparityMap.ptr<uchar>(_index_row);
		uchar* _row_contour_data = _Contour.ptr<uchar>(_index_row);
		for (int _col = 0; _col < rect.width; _col++)
		{
			int _index_col = rect.x + _col;
			if (_row_label_data[_index_col] == label_count
				&&_row_disparity_data[_index_col] >= peak - loDiff
				&& _row_disparity_data[_index_col] <= peak + upDiff)
			{
				_row_contour_data[_index_col] = 255;
			}
		}
	}
	removeSmallComponent(_Contour, 255, 100);
	removeSmallComponentForContour(_Contour);
	_Contour.copyTo(contour_get_from_depth);
	_Contour.release();
	return 1;
}


int VideoMatting::removeSmallComponentForContour(Mat &binary_image){
	Mat label_image;
	binary_image.convertTo(label_image, CV_32SC1);


	vector<int> labels;
	vector<Rect> rects;
	int max_rect_label = -1;
	Rect max_rect;
	int area_total = binary_image.rows * binary_image.cols;
	int label_count = 256;
	for (int _row = 0; _row < binary_image.rows; _row++)
	{
		int _index_row = _row;
		Rect _rect;
		int* _label_row_data = label_image.ptr<int>(_index_row);
		uchar* _binary_image_row_data = binary_image.ptr<uchar>(_index_row);
		for (int _col = 0; _col < binary_image.cols; _col++)
		{
			int _index_col = _col;
			if (_binary_image_row_data[_index_col] != 255)
			{
				continue;
			}
			if (_label_row_data[_index_col] > 255)
				continue;

			floodFill(label_image, cv::Point(_index_col, _index_row), label_count, &_rect, 0, 0, 8);

			if (_rect.area() > 0.25 * area_total)
			{
				max_rect = _rect;
				max_rect_label = label_count;
				label_count++;
				continue;
			}

			labels.push_back(label_count);
			rects.push_back(_rect);

			label_count++;
		}
	}

	if (max_rect_label == -1)
	{
		int area = 0;
		for (int i_rect = 0; i_rect < rects.size(); i_rect++)
		{
			if (rects[i_rect].area() > area)
			{
				area = rects[i_rect].area();
				max_rect_label = labels[i_rect];
				max_rect = rects[i_rect];
			}
		}
	}

	for (int i_rect = 0; i_rect < rects.size(); i_rect++)
	{
		if (max_rect_label == labels[i_rect])
		{
			rects.erase(rects.begin() + i_rect);
			labels.erase(labels.begin() + i_rect);
			break;
		}
	}

	for (int i_rect = 0; i_rect < rects.size(); i_rect++)
	{
		//Add more contraints.
		if (max_rect.contains(rects[i_rect].tl()) && max_rect.contains(rects[i_rect].br()))
		{
			rects.erase(rects.begin() + i_rect);
			labels.erase(labels.begin() + i_rect);
			i_rect--;
		}
	}

	for (int i_rect = 0; i_rect < rects.size(); i_rect++)
	{
		//cout << "rects: " << rects[i_rect] << endl;
		Rect _rect = rects[i_rect];
		int label = labels[i_rect];
		for (int _row = 0; _row < _rect.height; _row++){
			int _index_row_2 = _rect.y + _row;
			int* _row_label_data = label_image.ptr<int>(_index_row_2);
			uchar* _binary_image_row_data = binary_image.ptr<uchar>(_index_row_2);
			for (int _col = 0; _col < _rect.width; _col++){
				int _index_col_2 = _rect.x + _col;
				if (_row_label_data[_index_col_2] != label) {
					continue;
				}
				_binary_image_row_data[_index_col_2] = 0;
			}
		}
	}

	return 1;
}


int VideoMatting::expansionContour(Mat &_contour, int dSize){
	Mat edge;
	int area_total = _contour.rows * _contour.cols;
	Canny(_contour, edge, 1, 3);
	uchar *_contour_data = _contour.ptr<uchar>(0);
	uchar *_edge_data = edge.ptr<uchar>(0);
	for (int _row = 0; _row < edge.rows; _row++)
	{
		for (int _col = 0; _col < edge.cols; _col++)
		{
			int index = _row * edge.cols + _col;
			if (_edge_data[index] == 0)
				continue;

			for (int _c = -dSize; _c < dSize; _c++)
			{
				for (int _r = -dSize; _r < dSize; _r++)
				{
					int changeIndex = index + _c + _r*edge.cols;
					if (changeIndex >= 0 && changeIndex < area_total && _contour_data[changeIndex] == 0)
					{
						_contour_data[changeIndex] = 255;
					}
				}
			}
		}
	}
	return 1;
}

int VideoMatting::reduceContour(Mat &_contour, int dSize){
	Mat edge;
	Canny(_contour, edge, 1, 3);

	int totalEdge = edge.rows * edge.cols;
	uchar *_contourData = _contour.ptr<uchar>(0);
	uchar *colData = edge.ptr<uchar>(0);

	for (int _row = 0; _row < edge.rows; _row++)
	{
		for (int _col = 0; _col < edge.cols; _col++)
		{
			int index = _row * edge.cols + _col;
			if (colData[index] == 0)
				continue;

			for (int _c = -dSize; _c < dSize; _c++)
			{
				for (int _r = -dSize; _r < dSize; _r++)
				{
					int changeIndex = index + _c + _r*edge.cols;
					if (changeIndex >= 0 && changeIndex < totalEdge && _contourData[changeIndex] == 255)
					{
						_contourData[changeIndex] = 0;
					}
				}
			}
		}
	}

	return 1;

}


int VideoMatting::removeEdges(Mat &edges, const Mat &_contour){
	Mat orgineContour;
	_contour.copyTo(orgineContour);
	int cols = edges.cols;
	int rows = edges.rows;
	int area_total = rows * cols;

	//---------------------------------------
	//remove edges which are easily removed.
	//---------------------------------------
	Mat label_image;
	edges.convertTo(label_image, CV_32SC1);
	int label_count = 256;
	for (int _row = 0; _row < edges.rows; _row++)
	{
		Rect _rect;
		int* _row_label_data = label_image.ptr<int>(_row);
		uchar* _row_edges_data = edges.ptr<uchar>(_row);
		const uchar* _data_contour = orgineContour.ptr<uchar>(_row);
		for (int _col = 0; _col < edges.cols; _col++)
		{
			if (_row_label_data[_col] > 255 || _row_edges_data[_col] == 0)
				continue;

			floodFill(label_image, cv::Point(_col, _row), label_count, &_rect, 0, 0, 8);

			int inliers = 0, outliers = 0;
			for (int _row = 0; _row < _rect.height; _row++){
				int _index_row_2 = _rect.y + _row;
				int* _row_label_data = label_image.ptr<int>(_index_row_2);
				const uchar* _data_contour_2 = orgineContour.ptr<uchar>(_index_row_2);
				for (int _col = 0; _col < _rect.width; _col++){
					int _index_col_2 = _rect.x + _col;
					if (_row_label_data[_index_col_2] != label_count) {
						continue;
					}
					if (_data_contour_2[_index_col_2] == 255)
						inliers++;
					else
						outliers++;
				}
			}
			if (inliers >= outliers)
			{
				label_count++;
				continue;
			}
			if (inliers == 0 || outliers / inliers > 10)
			{
				for (int _row = 0; _row < _rect.height; _row++){
					int _index_row_2 = _rect.y + _row;
					int* _row_label_data_2 = label_image.ptr<int>(_index_row_2);
					uchar* _row_edges_data_2 = edges.ptr<uchar>(_index_row_2);
					for (int _col = 0; _col < _rect.width; _col++){
						int _index_col_2 = _rect.x + _col;
						if (_row_label_data_2[_index_col_2] != label_count) {
							continue;
						}
						_row_edges_data_2[_index_col_2] = 0;
					}
				}
			}
			label_count++;
		}
	}

	//-----------------------
	//expansion of contour
	//-----------------------
	expansionContour(orgineContour, 7);

	//-----------------------------------------
	//remove edges which are outside contour
	//-----------------------------------------
	const uchar *_contour_data = orgineContour.ptr<uchar>(0);
	uchar* _edges_data = edges.ptr<uchar>(0);
	for (int i = 0; i < area_total; i++)
	{
		if (_contour_data[i] == 0)
			_edges_data[i] = 0;
	}

	orgineContour.release();
	return 1;
}

int VideoMatting::accurateContour2(Mat &result, const Mat &edges, const Mat &_contour){
	int rows = edges.rows;
	int cols = edges.cols;
	int area_total = rows * cols;
	Mat contour_process;
	_contour.copyTo(contour_process);
	expansionContour(contour_process, 3);

	removeSomeOutliers(contour_process, edges);

#ifdef SHOWIMAGE
	imshow("removeSomeOutliers", contour_process);
	imwrite("removeSomeOutliers.png", contour_process);
	waitKey(0);
#endif
	removeMostOutliers(contour_process);
	removeAloner(contour_process);
#ifdef SHOWIMAGE
	imshow("removeMostOutliers", contour_process);
	waitKey(0);
#endif
	allChangeToWhite(contour_process);

	removeSmallComponent(contour_process, 255);

	contour_process.copyTo(result);
	contour_process.release();
	return 1;
}

int VideoMatting::allChangeToWhite(Mat &_contour_process)
{
	int area_total = _contour_process.rows * _contour_process.cols;
	uchar *_contour_process_data = _contour_process.ptr<uchar>(0);
	for (int i = 0; i < area_total; i++)
	{
		if (_contour_process_data[i] != 0)
		{
			_contour_process_data[i] = 255;
		}
	}
	return 1;
}

int VideoMatting::removeSomeOutliers(Mat &contour_process, const Mat &edges){
	int rows = edges.rows;
	int cols = edges.cols;
	int area_total = rows * cols;

	const uchar* _edges_data = edges.ptr<uchar>(0);
	uchar* _contour_process_data = contour_process.ptr<uchar>(0);
	for (int i = 0; i < area_total; i++)
	{
		if (_edges_data[i] == 255)
			_contour_process_data[i] = 128;
	}
	removeSmallComponent(contour_process, 0, 1000);
	Mat contour_au_debut;
	contour_process.copyTo(contour_au_debut);
	const uchar* _contour_au_debut_data = contour_au_debut.ptr<uchar>(0);
#ifdef SHOWIMAGE
	imshow("removeSomeOutliers0.1", contour_process);
	waitKey(0);
#endif
	//-------
	//3x3
	//-------
	/*
		direction = 
		{
		1, (-1,0)
		2, (-1, -1)
		3, (0,-1)
		4, (1, -1)
		5, (1,0)
		6, (1,1)
		7, (0,1)
		8  (-1,1)
		}
	*/

	int windowSize_n = 1;
	int windowSize_2n_1 = 2 * windowSize_n + 1;

	//begin from 1 ends with 8
	int directions[9] = {
		0, -1, -1 - cols, 
		-cols, -cols + 1, 1, 
		1 + cols, cols, cols - 1
	};


	for (int row = windowSize_n; row < rows - windowSize_n; row++)
	{
		const uchar* row_data_0 = contour_au_debut.ptr<uchar>(row - 1);
		const uchar* row_data_1 = contour_au_debut.ptr<uchar>(row);
		const uchar* row_data_2 = contour_au_debut.ptr<uchar>(row + 1);
		
		for (int col = windowSize_n; col < cols - windowSize_n; col++)
		{
			if (row_data_1[col] != 128)
			{
				continue;
			}
			int direct_1 = -1;
			int direct_2 = -1;

			int _ = row_data_0[col - 1] == 128;
			_ += 2 * (row_data_0[col] == 128);
			_ += 4 * (row_data_0[col+1] == 128);
			_ += 8 * (row_data_1[col-1] == 128);
			_ += 16 * (row_data_1[col] == 128);
			_ += 32 * (row_data_1[col+1] == 128);
			_ += 64 * (row_data_2[col-1] == 128);
			_ += 128 * (row_data_2[col] == 128);
			_ += 256 * (row_data_2[col+1] == 128);

			direct_1 = pre_def_3_directions_1[_];
			direct_2 = pre_def_3_directions_2[_];

			if (direct_1 == -1)
				continue;

			int avance_threshold = 20;
			int index_piexl = row * cols + col;
			int avance_1 = 2;
			int step_avance = directions[direct_1];
			if (direct_2 != -1)
			{
				int norm_width_1 = 0;
				int norm_width_2 = 0;
				for (int i = 2; ; i++)
				{
					if (i > 80 || index_piexl + i * step_avance > area_total)
					{
						norm_width_1 = i;
						break;
					}
					uchar val = _contour_au_debut_data[index_piexl + i * step_avance];
					if (val == 0)
					{
						norm_width_1 = i;
						break;
					}
				}
				step_avance = directions[direct_2];
				for (int i = 2;; i++)
				{
					if (i > 80 || index_piexl + i * step_avance > area_total)
					{
						norm_width_1 = i;
						break;
					}
					uchar val = _contour_au_debut_data[index_piexl + i * step_avance];
					if (val == 0)
					{
						norm_width_2 = i;
						break;
					}
				}
				if (norm_width_1 + norm_width_2 > 80)
					avance_threshold = 50;
			}
			
			step_avance = directions[direct_1];
			uchar change_result = 0;
			int border_avance_1 = 0;
			bool key_border_1 = false;
			for (; avance_1 < avance_threshold; avance_1++)
			{
				uchar val = _contour_au_debut_data[index_piexl + avance_1 * step_avance];
				uchar val_1 = _contour_au_debut_data[index_piexl + avance_1 * step_avance - 1];
				uchar val_2 = _contour_au_debut_data[index_piexl + avance_1 * step_avance + 1];
				if (val == 128 || val_1 == 128 || val_2 == 128)
					break;
				if (val == 255)
					continue;
				if (val == 0)
				{
					key_border_1 = true;
					break;
				}
			}
			if (false == key_border_1 && direct_2 == -1)
				continue;

			if (key_border_1 && direct_2 == -1)
			{
				for (int i = 1; i <= avance_1; i++)
				{
					_contour_process_data[index_piexl + i * step_avance] = change_result;
				}
				continue;
			}
			
			step_avance = directions[direct_2];
			bool key_border_2 = false;
			int avance_2 = 2;
			for (; avance_2 < avance_threshold; avance_2++)
			{
				uchar val = _contour_au_debut_data[index_piexl + avance_2 * step_avance];
				uchar val_1 = _contour_au_debut_data[index_piexl + avance_2 * step_avance - 1];
				uchar val_2 = _contour_au_debut_data[index_piexl + avance_2 * step_avance + 1];
				if (val == 128 || val_1 == 128 || val_2 == 128)
					break;
				if (val == 255)
					continue;
				if (val == 0)
				{
					key_border_2 = true;
					break;
				}
			}

			if (false == key_border_1 && false == key_border_2)
			{
				continue;
			}
			else if (false == key_border_1 && key_border_2)
			{
				for (int i = 1; i <= avance_2; i++)
				{
					_contour_process_data[index_piexl + i * step_avance] = change_result;
				}
				continue;
			}
			else if (key_border_1 && false == key_border_2)
			{
				step_avance = directions[direct_1];
				for (int i = 1; i <= avance_1; i++)
				{
					_contour_process_data[index_piexl + i * step_avance] = change_result;
				}
				continue;
			}
			else
			{
				int avance = avance_2;
				if (avance_2 > avance_1)
				{
					avance = avance_1;
					step_avance = directions[direct_1];
				}
					
				for (int i = 1; i <= avance; i++)
				{
					_contour_process_data[index_piexl + i * step_avance] = change_result;
				}
			}

		}
	}
	contour_au_debut.release();

	return 1;
}

int VideoMatting::removeMostOutliers(Mat &contour_process)
{
	int result = 0;
	Mat label_image;
	contour_process.convertTo(label_image, CV_32SC1);

	int rows = label_image.rows;
	int cols = label_image.cols;
	int area_total = rows * cols;
	int label_count = 256;
	int* _label_data = label_image.ptr<int>(0);
	uchar* _contour_data = contour_process.ptr<uchar>(0);
	for (int _row = 0; _row < rows; _row++)
	{
		int _index_row = _row;
		Rect _rect;
		int* _row_label_data = label_image.ptr<int>(_index_row);
		uchar* _row_contour_data = contour_process.ptr<uchar>(_index_row);
		for (int _col = 0; _col < cols; _col++)
		{
			int _index_col = _col;
			if (_row_contour_data[_index_col] != 255 || _row_label_data[_index_col] > 255)
			{
				continue;
			}

			floodFill(label_image, cv::Point(_index_col, _index_row), label_count, &_rect, 0, 0, 4);
			label_count++;

			if (_rect.area() > area_total / 10)
			{
				continue;
			}

			//--------
			//check
			//--------
			int this_label = label_count - 1;
			int black_count = 0;
			int edge_count = 0;
			Mat notes = Mat::zeros(rows, cols, CV_8UC1);
			uchar *_notes_data = notes.ptr<uchar>(0);
			if (_rect.tl() == Point(450, 321))
				cout << _rect;
			for (int _row_2 = 0; _row_2 < _rect.height; _row_2++)
			{
				int _index_row_2 = _rect.y + _row_2;
				for (int _col_2 = 0; _col_2 < _rect.width; _col_2++)
				{
					int _index_2 = cols * _index_row_2 + _rect.x + _col_2;
					if (_label_data[_index_2] != this_label)
						continue;

					uchar val = _contour_data[_index_2 - 1];
					if (_notes_data[_index_2 - 1])
						val = 255;
					if (val == 128){
						edge_count++;
						_notes_data[_index_2 - 1] = 1;
					}
					else if (val == 0)
					{
						black_count++;
						_notes_data[_index_2 - 1] = 1;
					}
						
					val = _contour_data[_index_2 + 1];
					if (_notes_data[_index_2 + 1])
						val = 255;
					if (val == 128)
					{
						edge_count++;
						_notes_data[_index_2 + 1] = 1;
					}
					else if (val == 0)
					{
						black_count++;
						_notes_data[_index_2 + 1] = 1;
					}
						
					val = _contour_data[_index_2 - cols];
					if (_notes_data[_index_2 - cols])
						val = 255;
					if (val == 128)
					{
						edge_count++;
						_notes_data[_index_2 - cols] = 1;
					}
					else if (val == 0)
					{
						black_count++;
						_notes_data[_index_2 - cols] = 1;
					}



					val = _contour_data[_index_2 + cols];
					if (_notes_data[_index_2 + cols])
						val = 255;
					if (val == 128)
					{
						edge_count++;
						_notes_data[_index_2 + cols] = 1;
					}
					else if (val == 0)
					{
						black_count++;
						_notes_data[_index_2 + cols] = 1;
					}
				}
			}
			if (black_count == 0)
				continue;
			int _max = max(_rect.width, _rect.height);
			if (edge_count > 2 * black_count)
				continue;
			if (edge_count > black_count && black_count < _max)
				continue;

			//----------
			//remove
			//----------
			for (int _row_2 = 0; _row_2 < _rect.height; _row_2++)
			{
				int _index_row_2 = _rect.y + _row_2;
				for (int _col_2 = 0; _col_2 < _rect.width; _col_2++)
				{
					int _index_2 = cols * _index_row_2 + _rect.x + _col_2;
					if (_label_data[_index_2] != this_label)
						continue;
					_contour_data[_index_2] = 0;
				}
			}
		}
	}

	return 1;
}

int VideoMatting::removeAloner(Mat &contour_process)
{
	int rows = contour_process.rows;
	int cols = contour_process.cols;
	int area_total = rows * cols;

	uchar *_contour_process_data = contour_process.ptr<uchar>(0);
	for (int row = 1; row < rows - 1; row++)
	{
		for (int col = 1; col < cols - 1; col++)
		{
			int index = row * cols + col;
			if (_contour_process_data[index] != 128)
				continue;
			int count = 0;
			if (_contour_process_data[index - 1] == 0)
				count++;
			if (_contour_process_data[index + 1] == 0)
			{
				count++;
				if (_contour_process_data[index - 1] == 0)
					count++;
			}
				
			if (_contour_process_data[index - cols] == 0)
				count++;
			if (_contour_process_data[index + cols] == 0)
			{
				count++;
				if (_contour_process_data[index - cols] == 0)
					count++;
			}

			if (count < 3)
				continue;

			_contour_process_data[index] = 0;
		}
	}
	
	return 1;
}

int VideoMatting::generateImageForMatting(Mat &_image){
	//plan 1: leftImage
	leftImage.copyTo(_image);
	return 1;

	//plan 2:
	//Mat image[3];
	//cvtColor(leftImage, image[2], CV_BGR2GRAY);
	//fgbgMask.copyTo(image[1]);
	//disparityMap.copyTo(image[0]);
	//merge(image, 3, _image);

	return 1;
}

int VideoMatting::removeSmallComponent(Mat &image, int removeColor, int diviseur){
	int result = 0;
	if (removeColor == 0)
		result = 255;

	Mat label_image;
	image.convertTo(label_image, CV_32SC1);

	int area_total = image.rows * image.cols;
	int label_count = 256;
	for (int _row = 0; _row < image.rows; _row++)
	{
		int _index_row = _row;
		Rect _rect;
		int* _row_label_data = label_image.ptr<int>(_index_row);
		uchar* _row_contour_data = image.ptr<uchar>(_index_row);
		for (int _col = 0; _col < image.cols; _col++)
		{
			int _index_col = _col;
			if (_row_contour_data[_index_col] != removeColor)
			{
				continue;
			}
			if (_row_label_data[_index_col] > 255)
				continue;

			floodFill(label_image, cv::Point(_index_col, _index_row), label_count, &_rect, 0, 0, 4);

			if (_rect.area() < area_total / diviseur)
			{
				_row--;
				break;
			}

			label_count++;
		}

		for (int _row = 0; _row < _rect.height; _row++){
			int _index_row_2 = _rect.y + _row;
			int* _row_label_data = label_image.ptr<int>(_index_row_2);
			for (int _col = 0; _col < _rect.width; _col++){
				int _index_col_2 = _rect.x + _col;
				if (_row_label_data[_index_col_2] != label_count) {
					continue;
				}
				image.at<uchar>(_index_row_2, _index_col_2) = result;
			}
		}
	}

	return 1;
}

void VideoMatting::bgdTransparent(){
	outputPNGImage = Mat(leftImage.rows, leftImage.cols, CV_8UC4, Scalar::all(0));

	int rows = leftImage.rows;
	int cols = leftImage.cols;
	for (int r = 0; r < rows; r++)
	{
		Vec3b *_image_row_data = leftImage.ptr<Vec3b>(r);
		Vec4b *_output_row_image_data = outputPNGImage.ptr<Vec4b>(r);
		uchar *_alpha_row_data = outputAlphaImage.ptr<uchar>(r);
		for (int c = 0; c < cols; c++)
		{
			_output_row_image_data[c] = Vec4b(_image_row_data[c][0], _image_row_data[c][1], 
				_image_row_data[c][2], _alpha_row_data[c]);
		}
	}
}


int VideoMatting::run(Mat _leftImage, Mat _disparityMap)
{

#ifdef SHOWUSINGTIME
	using namespace std;
	double timeBegin = getTickCount();
#endif

	if (!_leftImage.data) _leftImage = leftImage;
	else _leftImage.copyTo(leftImage);
	if (!_disparityMap.data) _disparityMap = disparityMap;
	else _disparityMap.copyTo(disparityMap);

#ifdef SHOWUSINGTIME
	double time1 = getTickCount();
	cout << "distribution of memory takes " << (time1 - timeBegin) * 1000 / getTickFrequency() << "ms" << endl;
#endif
	//------------------------------------
	//Step 1:
	//Extract contour of human from depth.
	//Pre-process this contour.
	//------------------------------------
	Mat contour_in_process;
	calcContourFromDepth(contour_in_process);

#ifdef SHOWUSINGTIME
	double time2 = getTickCount();
	cout << "calcContourFromDepth takes " << (time2 - time1) * 1000 / getTickFrequency() << "ms" << endl;
#endif

#ifdef SHOWIMAGE
	imshow("contour_from_depth", contour_in_process);
	waitKey(0);
#endif

	//----------------------------
	//Step 2:
	//Get canny of leftImage
	//----------------------------
	Mat edge_leftImage;

	//adaptiveBilateralFilter(leftImage, edge_leftImage, Size(5, 5), 10);
	blur(leftImage, edge_leftImage, Size(3, 3));
	Canny(edge_leftImage, edge_leftImage, canny_t1, canny_t2, 5);

#ifdef JAC_DEBUG
	cout << "step2 finished!" << endl;
#endif

#ifdef SHOWIMAGE
	imshow("edge_leftImage", edge_leftImage);
	waitKey(0);
#endif


	//-------------------------------------------
	//Step 3:
	//Remove useless edges in canny of leftImage 
	//-------------------------------------------
	int rows = contour_in_process.rows;
	int cols = contour_in_process.cols;
	int total = rows * cols;
	uchar *_edge_leftImage_data = edge_leftImage.ptr<uchar>(0);
	const uchar *_contour_data = contour_in_process.ptr<uchar>(0);
	removeEdges(edge_leftImage, contour_in_process);

#ifdef JAC_DEBUG
	cout << "step3 finished!" << endl;
#endif

#ifdef SHOWIMAGE
	imshow("edge_leftImage2", edge_leftImage);
	waitKey(0);
#endif


	//-----------------------------------------------
	//Step 4:
	//Get contour from canny edges and disparity map
	//-----------------------------------------------
	Mat contourResult;
	accurateContour2(contourResult, edge_leftImage, contour_in_process);
#ifdef JAC_DEBUG
	cout << "step4 finished!" << endl;
#endif

	//----------------
	//Step 5:
	//Generate trimap
	//----------------
	Mat trimap;
	contourResult.copyTo(trimap);
	Canny(contourResult, contourResult, 1, 3);
	int dSize = 4;
	uchar *_edge_data = contourResult.ptr<uchar>(0);
	uchar* _trimap_data = trimap.ptr<uchar>(0);
	for (int i_edge = 0; i_edge < total; i_edge++)
	{
		if (_edge_data[i_edge] == 0)
			continue;

		for (int _c = -dSize; _c < dSize; _c++)
		{
			for (int _r = -dSize; _r < dSize; _r++)
			{
				int changeIndex = i_edge + _c + _r*cols;
				if (changeIndex >= 0 && changeIndex < total)
				{
					_trimap_data[changeIndex] = 128;// the value has effets on shared matting.
				}
			}
		}
	}
#ifdef JAC_DEBUG
	cout << "step5 finished!" << endl;
#endif
	cvtColor(trimap, trimap, CV_GRAY2BGR);


#ifdef SHOWUSINGTIME
	double time3 = getTickCount();
	cout << "generateTrimap takes " << (time3 - time2) * 1000 / getTickFrequency() << "ms" << endl;
#endif

	//--------------------------------------------------
	//Step 6:
	//Using the trimap, matte with SharedMatting.
	//Then generate background and alpha image.
	//--------------------------------------------------
	SharedMatting sm;

	Mat imageForMatting;
	generateImageForMatting(imageForMatting);

	sm.loadImage(imageForMatting);
	sm.loadTrimap(trimap);
	sm.solveAlpha();
#ifdef SHOWUSINGTIME
	double time4 = getTickCount();
	cout << "SharedMatting takes " << (time4 - time3) * 1000 / getTickFrequency() << "ms" << endl;
#endif
	//sm.save("result.png");//Save the alpha result.
	sm.getMatte(outputAlphaImage);
	bgdTransparent();


	trimap.release();
	imageForMatting.release();
#ifdef SHOWUSINGTIME
	cout << "In total, it takes " << (getTickCount() - timeBegin) * 1000 / getTickFrequency() << "ms" << endl;
#endif
	return 1;
}
