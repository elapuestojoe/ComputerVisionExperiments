// Hw3.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <iostream>
#include "opencv2\highgui.hpp"
#include "opencv2\imgproc.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/opencv.hpp"
using namespace cv;

int main()
{
	VideoCapture camera(0);
	if (!camera.isOpened()) {
		return -1;
	}
	Mat currentImage;
	Mat halfScaledImage;
	Mat quarterScaledImage;

	while (true) {
		camera.read(currentImage);
		cv::resize(currentImage, currentImage, Size(currentImage.cols * 2, currentImage.rows * 2), 0, 0, CV_INTER_LINEAR);
		imshow("Original", currentImage);

		//Resize 
		cv::resize(currentImage, halfScaledImage, Size(currentImage.cols *0.5, currentImage.rows*0.5), 0, 0, CV_INTER_LINEAR);
		imshow("Scaled", halfScaledImage);

		cv::resize(currentImage, quarterScaledImage, Size(currentImage.cols *0.25, currentImage.rows*0.25), 0, 0, CV_INTER_LINEAR);
		imshow("Quarter", quarterScaledImage);
		if (waitKey(3) == 'x') {
			break;
		}
	}

	return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
