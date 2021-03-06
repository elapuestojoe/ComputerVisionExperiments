// ImageSegmentation.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp> 
#include <time.h>
#include <stack>
using namespace cv;
using namespace std;

//Convierte imagen a binaria usando un umbral
void ImageThreshold(const Mat &sourceImage, Mat &outImage, double umbral);
// k-means algorithm
double kmean(const Mat &sourceImage, Mat &labelImage, int k, std::vector<Vec3d> &start_colors);
void stackRegionGrow(Mat &image, Mat &outImage, Point p);
void recursiveRegionGrow(Mat &image, Mat &outImage, Point p);
void recursiveRegionGrowRGB(Mat &image, Mat &outR, Mat &outG, Mat &outB, Point p);
void IsoData(Mat &segmentedImage, Mat &labeledImage, double umbral);
int main()
{

	//for (int i = 0; i < 10; i++) {
	//	string filename = "C:\\Users\\Joe\\Pictures\\GTA" + std::to_string(i) + ".jpg";
	//	Mat image = imread(filename);
	//	Mat t;
	//	vector<Vec3d> v(0);
	//	kmean(image, t, 8, v);

	//	string outname = "C:\\Users\\Joe\\Pictures\\GTAKMEANS" + std::to_string(i) + ".jpg";
	//	imwrite(outname, t);
	//}

	string filename = "C:\\Users\\Joe\\Pictures\\GTA16.jpg";
	Mat image = imread(filename);
	Mat t;
	vector<Vec3d> v(0);
	kmean(image, t, 8, v);

	string outname = "C:\\Users\\Joe\\Pictures\\GTAKMEANS16.jpg";
	imwrite(outname, t);


	/*Mat image, gray, grayStack, recursiveGray; 
	
	image = imread("C:\\Users\\Joe\\Pictures\\fruits.jpg", CV_LOAD_IMAGE_COLOR);*/
	//

	//if (!image.data) {
	//	std::cout << "Could not find the image" << std::endl;
	//	return -1;
	//}
	//cv::resize(image, image, Size(), 0.25, 0.25, INTER_NEAREST);

	//cvtColor(image, gray, CV_RGB2GRAY);
	//stack<Point> stack, stackR;

	//while (stack.size() <= 64) {
	//	int x = rand() % gray.cols;
	//	int y = rand() % gray.rows;

	//	if (gray.at<uchar>(y, x) > 150) {
	//		stack.push(Point(x, y));
	//		stackR.push(Point(x, y));
	//	}
	//}

	//namedWindow("Original", WINDOW_AUTOSIZE);
	//cv::imshow("Original", image);

	////Growing-region
	//cvtColor(image, grayStack, CV_RGB2GRAY);
	//Mat outImage = Mat(gray.rows, gray.cols, gray.type(), Scalar(0));
	//while (stack.size() > 0) {
	//	Point currentPoint = stack.top();
	//	stack.pop();
	//	stackRegionGrow(grayStack, outImage, currentPoint);
	//}
	//imshow("Stack GR", outImage);
	////imshow("StackOr", grayStack);

	//////Recursive-growing region
	//cvtColor(image, recursiveGray, CV_RGB2GRAY);
	//Mat outImageRecursive = Mat(recursiveGray.rows, recursiveGray.cols, recursiveGray.type(), Scalar(0));

	//while (stackR.size() > 0) {
	//	Point currentPoint = stackR.top();
	//	stackR.pop();
	//	recursiveRegionGrow(recursiveGray, outImageRecursive, currentPoint);
	//}
	//cv::imshow("orRec", recursiveGray);
	//cv::imshow("Recursive GR", outImageRecursive);

	//imwrite("C:\\Users\\Joe\\Pictures\\lenaRec.jpg", outImageRecursive);
	//imwrite("C:\\Users\\Joe\\Pictures\\lenaStack.jpg", outImage);
	//
	waitKey(0);

	return 0;
}

void ImageThreshold(const Mat &sourceImage, Mat &outImage, double umbral)
{
	if (outImage.empty())
		outImage = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type());
	MatConstIterator_<Vec3b> it_in = sourceImage.begin<Vec3b>(), it_in_end = sourceImage.end<Vec3b>();
	MatIterator_<Vec3b> it_out = outImage.begin<Vec3b>();
	bool t;
	for (; it_in != it_in_end; ++it_in, ++it_out)
	{
		(*it_out)[0] = 0;
		(*it_out)[1] = 0;
		(*it_out)[2] = 0;
		t = (*it_in)[0] > umbral;
		t |= (*it_in)[1] > umbral;
		t |= (*it_in)[2] > umbral;
		if (t)
		{
			(*it_out)[0] = 255;
			(*it_out)[1] = 255;
			(*it_out)[2] = 255;
		}
	}
}

// k segmentation of a color image
double kmean(const Mat &sourceImage, Mat &labelImage, int k, std::vector<Vec3d> &start_colors)
{
	srand(time(NULL));
	if (labelImage.empty())
		labelImage = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type());
	Mat sampImage;
	Mat labImage;
	int j, n, changes, currentClass, kclass;
	double distance, mindistance;
	cv::resize(sourceImage, sampImage, Size(), 0.125, 0.125, INTER_NEAREST);
	//    imshow("Sampled", sampImage);
	labImage = sampImage.clone();
	//  imshow("Small", sampImage);
	MatConstIterator_<Vec3b> it_in = sampImage.begin<Vec3b>(), it_in_end = sampImage.end<Vec3b>();
	MatIterator_<Vec3b> it_label = labImage.begin<Vec3b>();
	Vec3d red(0.5, 0, 0);
	Vec3d green(0, 0.5, 0);
	Vec3d blue(0, 0, 0.5);
	Vec3d black(0.0, 0.0, 0.0);
	Vec3d white(1.0, 1.0, 1.0);
	Vec3d dis;
	std::vector<Vec3d> mean_colors, newmean_colors, five_colors;
	std::vector<int> size_colors;
	mean_colors.resize(k, black);
	newmean_colors.resize(k, black);
	size_colors.resize(k, 0);
	five_colors.resize(5, 0);
	five_colors[0] = red;
	five_colors[1] = green;
	five_colors[2] = blue;
	five_colors[3] = black;
	five_colors[4] = white;
	if (start_colors.size() == (unsigned int)k)
	{
		for (n = 0; n < k; n++)
		{
			mean_colors[n] = start_colors[n];
		}
	}
	else
	{
		start_colors.resize(k, 0);
		if (k == 5)
		{
			mean_colors[0] = red;
			mean_colors[1] = green;
			mean_colors[2] = blue;
			mean_colors[3] = black;
			mean_colors[4] = white;
		}
		else
		{
			for (n = 0; n < k; n++)
			{
				for (j = 0; j < 3; j++) mean_colors[n][j] = (double)rand() / RAND_MAX;
			}
		}
	}
	changes = 0;
	int loops = 0;
	do
	{
		//        printf("%d Changes=%d  (%d, %8.4f),(%d, %8.4f),(%d, %8.4f),(%d, %8.4f),(%d, %8.4f) \n",loops,changes,size_colors[0],mean_colors[0][0],size_colors[1],mean_colors[1][1],size_colors[2],mean_colors[2][2],size_colors[3],mean_colors[3][0],size_colors[4],mean_colors[4][0]);
		changes = 0;
		kclass = 0;
		for (n = 0; n < k; n++)
		{
			size_colors[n] = 0;
			newmean_colors[n][0] = 0;
			newmean_colors[n][1] = 0;
			newmean_colors[n][2] = 0;
		}
		for (it_in = sampImage.begin<Vec3b>(), it_label = labImage.begin<Vec3b>(); it_in != it_in_end; ++it_in, ++it_label)
		{
			currentClass = (*it_label)[0] % k;
			kclass = 0;
			mindistance = 0;
			for (j = 0; j < 3; j++)
			{
				dis[j] = (double)((*it_in)[j]) / 255.0 - mean_colors[0][j];
				mindistance += dis[j] * dis[j];
			}
			for (n = 1; n < k; n++)
			{
				distance = 0;
				for (j = 0; j < 3; j++)
				{
					dis[j] = (double)((*it_in)[j]) / 255.0 - mean_colors[n][j];
					distance += dis[j] * dis[j];
				}
				if (distance < mindistance)
				{
					mindistance = distance;
					kclass = n;
				}
			}
			changes += (currentClass != kclass);
			(*it_label)[0] = kclass;
			++size_colors[kclass];
			for (j = 0; j < 3; j++)
			{
				newmean_colors[kclass][j] += (double)((*it_in)[j]) / 255.0;
			}
		}
		for (n = 0; n < k; n++)
		{
			if (size_colors[n] > 0)
			{
				for (j = 0; j < 3; j++)
				{
					mean_colors[n][j] = newmean_colors[n][j] / size_colors[n];
				}
			}
		}
		//        imshow("SmallLabels", 20*labImage);
		//        printf("%d Changes=%d  (%d, %8.4f),(%d, %8.4f),(%d, %8.4f),(%d, %8.4f),(%d, %8.4f) \n",loops,changes,size_colors[0],mean_colors[0][0],size_colors[1],mean_colors[1][1],size_colors[2],mean_colors[2][2],size_colors[3],mean_colors[3][0],size_colors[4],mean_colors[4][0]);
		++loops;
	} while ((changes > 100) && (loops < 100));
	MatConstIterator_<Vec3b> it_inSource = sourceImage.begin<Vec3b>(), it_in_endSource = sourceImage.end<Vec3b>();
	MatIterator_<Vec3b> it_inLabel = labelImage.begin<Vec3b>();
	for (it_inSource = sourceImage.begin<Vec3b>(), it_inLabel = labelImage.begin<Vec3b>(); it_inSource != it_in_endSource; ++it_inSource, ++it_inLabel)
	{
		currentClass = 0;
		kclass = 0;
		mindistance = 0;
		for (j = 0; j < 3; j++)
		{
			dis[j] = (double)((*it_inSource)[j]) / 255.0 - mean_colors[0][j];
			mindistance += dis[j] * dis[j];
		}
		for (n = 1; n < k; n++)
		{
			distance = 0;
			for (j = 0; j < 3; j++)
			{
				dis[j] = (double)((*it_inSource)[j]) / 255.0 - mean_colors[n][j];
				distance += dis[j] * dis[j];
			}
			if (distance < mindistance)
			{
				mindistance = distance;
				kclass = n;
			}
		}
		if (k != 5)
		{
			(*it_inLabel)[0] = mean_colors[kclass][0] * 255;
			(*it_inLabel)[1] = mean_colors[kclass][1] * 255;
			(*it_inLabel)[2] = mean_colors[kclass][2] * 255;
		}
		//            (*it_inLabel)[0]= kclass*50;
		//            (*it_inLabel)[1]= kclass*50;
		//            (*it_inLabel)[2]= kclass*50;
		else
		{
			(*it_inLabel)[0] = five_colors[kclass][0] * 255;
			(*it_inLabel)[1] = five_colors[kclass][1] * 255;
			(*it_inLabel)[2] = five_colors[kclass][2] * 255;
		}
	}
	for (n = 0; n < k; n++)
	{
		start_colors[n] = mean_colors[n];
	}
	return 0;
}

void stackRegionGrow(Mat &image, Mat &outImage, Point p) {
	stack<Point> stack;
	stack.push(p);
	float seedValue = image.at<uchar>(p.y, p.x);

	float threshValue = 50;

	float minVal = 50;
	if (seedValue - threshValue > minVal) {
		minVal = seedValue - threshValue;
	}
	float maxVal = 255;
	if (seedValue + threshValue < maxVal) {
		maxVal = seedValue + threshValue;
	}

	while (!stack.empty()) {

		Point currentP = stack.top();
		stack.pop();

		float pixelValue = image.at<uchar>(currentP.y, currentP.x);
		if (pixelValue > minVal && pixelValue < maxVal) {
			outImage.at<uchar>(currentP.y, currentP.x) = 255;
			image.at<uchar>(currentP.y, currentP.x) = 0;

			//Add neighbors
			if (currentP.x > 0) {
				stack.push(Point(currentP.x - 1, currentP.y));
			}
			if (currentP.y > 0) {
				stack.push(Point(currentP.x, currentP.y - 1));
			}
			if (currentP.x < image.cols - 1) {
				stack.push(Point(currentP.x + 1, currentP.y));
			}
			if (currentP.y < image.rows - 1) {
				stack.push(Point(currentP.x, currentP.y + 1));
			}
		}

	}
}


void recursiveRegionGrow(Mat &image, Mat &outImage, Point p) {

	if (p.x > 0 && p.x < (image.cols - 1) && p.y > 0 && p.y < (image.rows - 1)) {
		
		if (image.at<uchar>(p.y, p.x) > 150) {
			outImage.at<uchar>(p.y, p.x) = 255;
			image.at<uchar>(p.y, p.x) = 0;
			recursiveRegionGrow(image, outImage, Point(p.x, p.y + 1));
			recursiveRegionGrow(image, outImage, Point(p.x + 1, p.y));
			recursiveRegionGrow(image, outImage, Point(p.x, p.y - 1));
			recursiveRegionGrow(image, outImage, Point(p.x - 1, p.y));
		}
		image.at<uchar>(p.y, p.x) = 0;	
	}
}

//void recursiveRegionGrowRGB(Mat &image, Mat &outR, Mat &outG, Mat &outB, Point p) {
//
//	if (p.x >= 0 && p.x < (image.cols - 1) && p.y >= 0 && p.y < (image.rows - 1)) {
//
//		if (image.at<uchar>(p.y, p.x) > 150) {
//			outImage.at<uchar>(p.y, p.x) = 255;
//			image.at<uchar>(p.y, p.x) = 0;
//			recursiveRegionGrow(image, outImage, Point(p.x - 1, p.y));
//			recursiveRegionGrow(image, outImage, Point(p.x, p.y + 1));
//			recursiveRegionGrow(image, outImage, Point(p.x + 1, p.y));
//			recursiveRegionGrow(image, outImage, Point(p.x, p.y - 1));
//		}
//	}
//}

//used to iterate an image
//MatIterator_<Vec3b> it_in = image.begin<Vec3b>(), it_in_end = image.end<Vec3b>();
//for (; it_in != it_in_end; ++it_in) {
//	(*it_in)[0] = 0;
//	(*it_in)[1] = 0;
//	(*it_in)[2] = 0;
//}

void IsoData(Mat &segmentedImage, Mat &labeledImage, double umbral)
{
	labeledImage = Mat(segmentedImage.rows, segmentedImage.cols, segmentedImage.type(), Scalar(0));

	static Mat intLablesImage = Mat(segmentedImage.rows, segmentedImage.cols, CV_16UC1);
	static unsigned short labels_lut[65536];

	MatIterator_<Vec3b> it_in = segmentedImage.begin<Vec3b>(), it_in_end = segmentedImage.end<Vec3b>();
	MatIterator_<unsigned short> it_out = intLablesImage.begin<unsigned short>();

	for (; it_out != (intLablesImage.begin<unsigned short>() + intLablesImage.cols + 1); ++it_out, ++it_in)
	{
		(*it_out) = 0;
		(*it_in)[0] = 255;
	}

	unsigned short labels = 1;
	for (unsigned int i = 0; i < 65536; i++) labels_lut[i] = i;
	unsigned short i, clabel, labelizq, labelback, minlabel;

	it_in = segmentedImage.begin<Vec3b>() + 1 + segmentedImage.cols;
	it_in_end = segmentedImage.end<Vec3b>();
	it_out = intLablesImage.begin<unsigned short>() + 1 + segmentedImage.cols;

	MatConstIterator_<Vec3b> it_in_izq = it_in - 1;
	MatConstIterator_<Vec3b> it_in_back = it_in - segmentedImage.cols;
	MatIterator_<unsigned short> it_label_izq = it_out - 1;
	MatIterator_<unsigned short> it_label_back = it_out - segmentedImage.cols;

	double diff, max_up_diff, max_left_diff;
	for (; it_in != it_in_end; ++it_in, ++it_out, ++it_in_izq, ++it_in_back, ++it_label_izq, ++it_label_back)
	{
		minlabel = 0xffff;
		labelizq = labelback = -1;
		max_up_diff = fabs((*it_in)[0] - (*it_in_back)[0]);
		max_left_diff = fabs((*it_in)[0] - (*it_in_izq)[0]);
		for (i = 1; i < 3; i++)
		{
			diff = fabs((*it_in)[i] - (*it_in_back)[i]);
			if (diff > max_up_diff) max_up_diff = diff;
			diff = fabs((*it_in)[i] - (*it_in_izq)[i]);
			if (diff > max_left_diff) max_left_diff = diff;
		}
		if (max_up_diff < umbral)
		{
			labelback = labels_lut[*it_label_back];
			minlabel = labelback;
		}
		if (max_left_diff < umbral)
		{
			labelizq = labels_lut[*it_label_izq];
			if (minlabel > labelizq) minlabel = labelizq;
		}
		if ((max_up_diff < umbral) && (max_left_diff < umbral) && (labelback != labelizq))
		{
			labels_lut[labelback] = minlabel;
			labels_lut[labelizq] = minlabel;
		}
		if ((max_left_diff >= umbral) && (max_up_diff >= umbral))
		{
			++labels;
			minlabel = labels;
		}
		(*it_out) = minlabel;
	}
	int changes = 0;
	do
	{
		changes = 0;
		for (i = labels; i > 0; i--)
		{
			clabel = labels_lut[i];
			while (labels_lut[clabel] != clabel)
			{
				clabel = labels_lut[clabel];
				++changes;
			}
			labels_lut[i] = clabel;
		}
	} while (changes > 0);
	MatIterator_<unsigned short> it_labelIn = intLablesImage.begin<unsigned short>() + segmentedImage.cols + 1;
	MatIterator_<unsigned short> it_labelEnd = intLablesImage.end<unsigned short>() - +segmentedImage.cols - 1;
	MatIterator_<Vec3b> it_segOut = labeledImage.begin<Vec3b>();
	MatConstIterator_<Vec3b> it_raw = segmentedImage.begin<Vec3b>();
	MatIterator_<unsigned short> it_label_der = it_labelIn + 1;
	MatIterator_<unsigned short> it_label_front = it_labelIn + segmentedImage.cols;
	it_label_izq = it_labelIn - 1;
	it_label_back = it_labelIn - segmentedImage.cols;
	for (; it_labelIn != it_labelEnd; ++it_labelIn, ++it_segOut, ++it_label_der, ++it_label_front, ++it_label_izq, ++it_label_back, ++it_raw)
	{
		labels = labels_lut[(*it_labelIn)];
		(*it_segOut) = (*it_raw);
		if ((labels != labels_lut[(*it_label_der)])
			|| (labels != labels_lut[(*it_label_izq)])
			|| (labels != labels_lut[(*it_label_back)])
			|| (labels != labels_lut[(*it_label_front)]))
		{
			(*it_segOut)[0] = 0;
			(*it_segOut)[1] = 255;
			(*it_segOut)[2] = 255;
		}
		else
		{
			(*it_segOut)[2] = 16 * (labels % 16);
		}
	}
}
