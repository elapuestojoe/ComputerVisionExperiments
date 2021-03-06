// SpyExercise.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

/*Implement your approach to the problem of segmenting characters under a non uniform condition.
Use your smart phone camera with the flash on on a dark room to capture 
a regular graduate - level textbook page.
The expected result, is that the segmentation algorithm should be able to 
separate every single character in the picture.
If your algorithm does not work, implement one that is able to do the segmentation task. */

#include "pch.h"
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

void Transform(Mat &sourceImage, Mat &destinationImage, int mask_size, int morph_type);
RNG rng(12345);
int main()
{
	Mat image, processedImage, foregroundImage, binaryImage, outImage;
	image = imread("C:\\Users\\Joe\\Pictures\\IMG-0990.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	//cv::resize(image, image, Size(image.cols*0.25, image.rows*0.25), 0, 0, CV_INTER_LINEAR);

	//Apply opening
	Transform(image, processedImage, 3, MORPH_OPEN);
	//Apply erotion
	Transform(processedImage, processedImage, 3, MORPH_ERODE);
	//Apply dilation 4 times
	Transform(processedImage, processedImage, 3, MORPH_DILATE);
	Transform(processedImage, processedImage, 3, MORPH_DILATE);
	Transform(processedImage, processedImage, 3, MORPH_DILATE);
	Transform(processedImage, processedImage, 3, MORPH_DILATE);

	cv::subtract(processedImage, image, foregroundImage);

	//Apply Otsu thresholding
	cv::threshold(foregroundImage, binaryImage, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

	//Clean (Open->Close)
	//Transform(binaryImage, binaryImage, 3, MORPH_CLOSE);
	/*Transform(binaryImage, binaryImage, 3, MORPH_OPEN);*/
	//Transform(binaryImage, binaryImage, 3, MORPH_ERODE);

	//Apply connected components
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

		if (stats.at<int>(i, CC_STAT_AREA) < 5) {
			colors[i] = Vec3b(0, 0, 0); // small regions are painted with black too
		}
		else {
			colors[i] = cv::Vec3b(rand() % 256, rand() % 256, rand() % 256);
		}
		Rect rect(x, y, w, h);
		//cv::rectangle(image, rect, colors[i]);
	}

	outImage = cv::Mat::zeros(image.size(), CV_8UC3);
	for (int y = 0; y < outImage.rows; y++) {
		for (int x = 0; x < outImage.cols; x++)
		{
			int label = labels.at<int>(y, x);
			CV_Assert(0 <= label && label <= nccomps);
			outImage.at<cv::Vec3b>(y, x) = colors[label];
		}
	}
	cv::imwrite("C:\\Users\\Joe\\Pictures\\SpyOutImage.jpg", outImage);
	cv::imwrite("C:\\Users\\Joe\\Pictures\\SpyProcessed.jpg", processedImage);
	cv::imwrite("C:\\Users\\Joe\\Pictures\\SpyForeground.jpg", foregroundImage);
	cv::imwrite("C:\\Users\\Joe\\Pictures\\SpyOtsu.jpg", binaryImage);
	while (true) {
		cv::imshow("Original", image);
		cv::imshow("Processed", processedImage);
		cv::imshow("Foreground", foregroundImage);
		cv::imshow("Otsu Image", binaryImage);
		cv::imshow("Out Image", outImage);
		if (waitKey(3) == 'x') {
			break;
		}
	}
	return 1;
}

void Transform(Mat &sourceImage, Mat &destinationImage, int mask_size, int morph_type) {
	Mat structuringElement = getStructuringElement(
		MORPH_CROSS,
		Size(2 * mask_size + 1, 2 * mask_size + 1),
		Point(mask_size, mask_size));
	if (destinationImage.empty()) {
		destinationImage = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type());
	}

	morphologyEx(sourceImage, destinationImage, morph_type, structuringElement);
}
