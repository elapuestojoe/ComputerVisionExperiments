// SpyExerciseExamAnswer.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;
void DistanceTransform(Mat &sourceImage, Mat &destinationImage);
void MedialAxisTransform(Mat &sourceImage, Mat &distanceTransformedImage, Mat &destinationImage);
void SkeletonizeImage(Mat &sourceImage, Mat &skeleton, Mat tmpImage);
void Dilation(Mat &sourceImage, Mat &destinationImage, Mat &structuringElement);
void Erotion(Mat &sourceImage, Mat &destinationImage, Mat &structuringElement);
void ConnectedComponents(Mat &binaryImage, Mat &outImage);
int main()
{
	Mat originalImage, image, skeletonImage, distanceTransform, MAT, coloredImage;
	image = imread("C:\\Users\\Joe\\Pictures\\IMG-0990.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	image.copyTo(originalImage);

	cv::medianBlur(image, image, 3);
	cv::imwrite("C:\\Users\\Joe\\Pictures\\SpyExamMean.jpg", image);

	cv::adaptiveThreshold(image, image, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 7, 2);
	cv::imwrite("C:\\Users\\Joe\\Pictures\\SpyExamThresh.jpg", image);

	//SkeletonizeImage(originalImage, skeletonImage, image);
	//cv::imwrite("C:\\Users\\Joe\\Pictures\\SpyExamMAT.jpg", skeletonImage);

	ConnectedComponents(image, coloredImage);
	cv::imwrite("C:\\Users\\Joe\\Pictures\\SpyExamColor.jpg", coloredImage);
	return 1;
}

void SkeletonizeImage(Mat &sourceImage, Mat &skeleton, Mat tmpImage) {
	MatSize imgSize = sourceImage.size;

	skeleton = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type(), Scalar(0));

	int mask_size = 1;
	Mat structuringElement = getStructuringElement(
		MORPH_CROSS,
		Size(2 * mask_size + 1, 2 * mask_size + 1),
		Point(mask_size, mask_size));
	bool done = false;
	int i = 0;
	while (!done) {
		Mat eroded;
		Erotion(tmpImage, eroded, structuringElement);

		Mat temp;
		Dilation(eroded, temp, structuringElement);
		subtract(tmpImage, temp, temp);

		bitwise_or(skeleton, temp, skeleton);

		eroded.copyTo(tmpImage);

		done = true;
		MatConstIterator_<Vec3b> it_in = eroded.begin<Vec3b>(), it_in_end = eroded.end<Vec3b>();
		for (; it_in != it_in_end; ++it_in) {
			if ((*it_in)[0] != 0) {
				done = false;
				break;
			}
		}
	}
}

void DistanceTransform(Mat &threshImage, Mat &destinationImage) {

	distanceTransform(threshImage, destinationImage, DIST_L2, 3);
	normalize(destinationImage, destinationImage, 0, 1., NORM_MINMAX);
}

void MedialAxisTransform(Mat &sourceImage, Mat &distanceTransformedImage, Mat &destinationImage) {

	destinationImage = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type(), Scalar(0));

	MatConstIterator_<Vec3b> it_in = sourceImage.begin<Vec3b>(), it_in_end = sourceImage.end<Vec3b>();
	MatConstIterator_<Vec3b> dImg_begin = distanceTransformedImage.begin<Vec3b>();
	MatIterator_<Vec3b> out_in = destinationImage.begin<Vec3b>();

	for (; it_in != it_in_end; ++it_in, ++dImg_begin, ++out_in) {
		if ((*it_in)[0] != 0) {
			(*out_in)[0] = (*dImg_begin)[0];
		}
		else {
			(*out_in)[0] = 0;
		}
	}
}

void Dilation(Mat &sourceImage, Mat &destinationImage, Mat &structuringElement) {

	if (destinationImage.empty()) {
		destinationImage = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type());
	}

	int dilation_size = 1;

	dilate(sourceImage, destinationImage, structuringElement);
}

void Erotion(Mat &sourceImage, Mat &destinationImage, Mat &structuringElement) {

	if (destinationImage.empty()) {
		destinationImage = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type());
	}

	int erotion_size = 1;

	erode(sourceImage, destinationImage, structuringElement);
}

void ConnectedComponents(Mat &binaryImage, Mat &outImage) {
	Mat labels, stats, centroids;
	int i, nccomps = cv::connectedComponentsWithStats(
		binaryImage,
		labels,
		stats,
		centroids
	);
	vector<Vec3b> colors(nccomps + 1);
	colors[0] = Vec3b(0, 0, 0); // Set background pixels black
	for (int i = 1; i < stats.rows; i++) {
		int x = stats.at<int>(Point(0, i));
		int y = stats.at<int>(Point(1, i));
		int w = stats.at<int>(Point(2, i));
		int h = stats.at<int>(Point(3, i));

		if (stats.at<int>(i, CC_STAT_AREA) < 50) {
			colors[i] = Vec3b(0, 0, 0); // small regions are painted with black too
		}
		else {
			colors[i] = cv::Vec3b(rand() % 256, rand() % 256, rand() % 256);
		}
		Rect rect(x, y, w, h);
		//cv::rectangle(image, rect, colors[i]);
	}

	outImage = cv::Mat::zeros(binaryImage.size(), CV_8UC3);
	for (int y = 0; y < outImage.rows; y++) {
		for (int x = 0; x < outImage.cols; x++)
		{
			int label = labels.at<int>(y, x);
			CV_Assert(0 <= label && label <= nccomps);
			outImage.at<cv::Vec3b>(y, x) = colors[label];
		}
	}
}