// FeatureExtraction.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp> 
#include <algorithm>
#include <cmath>

using namespace cv;
using namespace std;

void Erotion(Mat &sourceImage, Mat &destinationImage, Mat &structuringElement);
void Dilation(Mat &sourceImage, Mat &destinationImage, Mat &structuringElement);
void ReduceImageOverlap(Mat &sourceImage, Mat &destinationImage);
void ProcessComponent(Mat &sourceimage, int x, int y, int w, int h);

RNG rng(12345);
int main()
{
	Mat image, otsuImage, labels, stats, centroids;

	vector<Vec4i> hierarchy;

	//Read image
	image = imread("C:\\Users\\Joe\\Pictures\\CircleSquare.png", CV_LOAD_IMAGE_GRAYSCALE);

	//Threshold
	cv::threshold(image, otsuImage, 128, 255, CV_THRESH_BINARY);
	bitwise_not(otsuImage, otsuImage);

	//Segment image
	int i, nccomps = cv::connectedComponentsWithStats(
		otsuImage,
		labels,
		stats,
		centroids
	);
	cout << "Image size: " << image.size() << endl;
	cout << "Total connected Components Detected: " << nccomps << endl;
	cout << "1 Background - " << nccomps - 1 << " Foreground" << endl << endl;

	vector<Vec3b> colors(nccomps + 1);
	colors[0] = Vec3b(0, 0, 0); // Background pixels black

	vector<int> areas(nccomps - 1);
	vector<double> perimeters(nccomps - 1);
	for (int i = 1; i < stats.rows; i++) {

		cout << "Element" << i << endl;
		int x = stats.at<int>(Point(0, i));
		int y = stats.at<int>(Point(1, i));
		int w = stats.at<int>(Point(2, i));
		int h = stats.at<int>(Point(3, i));
		std::cout << "x=" << x << " y=" << y << " w=" << w << " h=" << h << std::endl;

		if (stats.at<int>(i, CC_STAT_AREA) < 100) {
			colors[i] = Vec3b(0, 0, 0); // small regions are painted with black too
		}

		//Get area
		areas[i-1] = stats.at<int>(i, CC_STAT_AREA);
		std::cout << "Area " << areas[i-1] << endl;

		//Get perimeter

		//Get mask for i-th contour - We only care about the first contour which is the outermost 
		Mat1b mask_i = labels == i ;
		vector<vector<Point>> contours;
		findContours(mask_i.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
		perimeters[i - 1] = arcLength(contours[0], true);
		std::cout << "Perimeter " << perimeters[i-1] << endl;
		
		//Get Moments
		vector<vector<Point>> momentContours;
		vector<Vec4i> hierarchy;
		findContours(mask_i.clone(), momentContours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		
		vector<Moments> mu(momentContours.size());
		for (int j = 0; j < contours.size(); j++) {
			mu[j] = moments(contours[j], true);
			cout << "Moments: " << endl;
			cout << "m00 - " << mu[j].m00 << endl;
			cout << "m10 - " << mu[j].m10 << endl;
			cout << "m01 - " << mu[j].m01 << endl;
		}
		
		//Histogram of moments?

		// Get the mass centers:
		vector<Point2f> mc(contours.size());

		for (int j = 0; j < contours.size(); j++)
		{
			mc[j] = Point2f(0, 0);
			//mu[i].m00 could be 0 if object has self intersections
			if (mu[j].m00 != 0) {
				mc[j] = Point2f(mu[j].m10 / mu[j].m00, mu[j].m01 / mu[j].m00);
			}
			cout << "Centroid" << mc[j] << endl;
		}

		//Distance between edges and centroid
		cout << "Points" << endl;
		for (int j = 0; j < contours[0].size(); j++) {
			double distance = pow(mc[0].x - contours[0][j].x, 2) + pow(mc[0].y - contours[0][j].y, 2);
			distance = pow(distance, 0.5);
			//cout << contours[0][i] << " - " << mc[0] << " = " << distance << endl;
			
		}

		//Get Hu-Moments
		cv::Moments moment = cv::moments(contours[0]);
		double hu[7];
		cv::HuMoments(moment, hu);
		cout << "Hu moments: " << endl;
		for (int j = 0; j < 7; j++) {
			cout << hu[j] << ",";
		}
		std:cout << endl;

		std::cout << endl;
	}

	while (true) {
		if (waitKey(3) == 'x') {
			break;
		}
	}
	return 0;
}

void Erotion(Mat &sourceImage, Mat &destinationImage, Mat &structuringElement) {

	if (destinationImage.empty()) {
		destinationImage = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type());
	}

	int erotion_size = 1;

	erode(sourceImage, destinationImage, structuringElement);
}

void Dilation(Mat &sourceImage, Mat &destinationImage, Mat &structuringElement) {

	if (destinationImage.empty()) {
		destinationImage = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type());
	}

	int dilation_size = 1;

	dilate(sourceImage, destinationImage, structuringElement);
}

void ReduceImageOverlap(Mat &sourceImage, Mat &destinationImage) {

	destinationImage = sourceImage.clone();

	int morph_elem = 0;
	int morph_size = 9;
	Mat element = getStructuringElement(
		morph_elem, 
		Size(2 * morph_size + 1, 2 * morph_size + 1), 
		Point(morph_size, morph_size));

	int dilation_size = 3;
	Mat dilationElement = getStructuringElement(
		morph_elem,
		Size(2 * dilation_size + 1, 2 * dilation_size + 1),
		Point(dilation_size, dilation_size));

	Erotion(destinationImage, destinationImage, element);
	Dilation(destinationImage, destinationImage, dilationElement);
}

void ProcessComponent(Mat &sourceImage, int x, int y, int w, int h) {
	cv::Rect component(x, y, x + w, y + h);
	Mat croppedImage = sourceImage(component);

	//cv::imshow(to_string(x), croppedImage);
}