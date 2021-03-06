// Hw5.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <iostream>
#include "opencv2\highgui.hpp"
#include "opencv2\imgproc.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/opencv.hpp"
#include <vector>
#include<math.h>
#include <time.h>   
using namespace cv;

void convolucion(const Mat &sourceImage, Mat &destinationImage, const Mat &kernel);
void cannyEdge(const Mat &sourceImage);
void splitImageRGB(const Mat &sourceImage);
void gradientProcess(const Mat &sourceImage);
void convolucion(const Mat &sourceImage, Mat &destinationImage, const Mat &kernel);
void cannyEdgeDetection(const Mat &sourceImage);
void nativeCannyEdgeDetection(Mat &sourceImg);
const double PI = 3.141592653589793238463;

int main()
{

	VideoCapture camera(0);
	if (!camera.isOpened()) {
		return -1;
	}
	Mat currentImage, currentImageHSV, gradientImage, currentImageGray;
	while (true) {

		camera.read(currentImage);
		cv::imshow("Original", currentImage);

		cvtColor(currentImage, currentImageHSV, CV_BGR2HSV);

		//cv::imshow("HSV", currentImageHSV);
		//gradientProcess(currentImageHSV);
		

		//cannyEdge(currentImageHSV);

		//cannyEdgeDetection(currentImageHSV);

		//cvtColor(currentImage, currentImageGray, CV_BGR2GRAY);
		nativeCannyEdgeDetection(currentImage);

		if (waitKey(3) == 'x') {
			break;
		}
	}

	return 0;
}

void convolucion(const Mat &sourceImage, Mat &destinationImage, const Mat &kernel)
{
	if (destinationImage.empty())
		destinationImage = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type());
	for (int y = 1; y < sourceImage.rows - 1; ++y)
		for (int x = 1; x < sourceImage.cols - 1; ++x)
			for (int i = 0; i < sourceImage.channels(); ++i)
			{
				float conv = 0;
				destinationImage.at<Vec3b>(y, x)[i] = sourceImage.at<Vec3b>(y, x)[i];
				MatConstIterator_<float> kernel_it = kernel.begin<float>();
				for (int n = -kernel.rows / 2; n <= kernel.rows / 2; ++n)
				{
					for (int m = -kernel.cols / 2; m <= kernel.cols / 2; ++m, ++kernel_it)
					{
						conv += (*kernel_it)*sourceImage.at<Vec3b>(y - m, x - n)[i];
					}
				}
				destinationImage.at<Vec3b>(y, x)[i] = (char)(conv + 0.5);
			}
}

void cannyEdge(const Mat &sourceImage) {
	Mat contours;
	Mat grayImage;
	std::vector<cv::Mat> channels;
	cv::split(sourceImage, channels);
	grayImage = channels[0];

	Canny(sourceImage, contours, 50, 350);

	cv::imshow("Canny", contours);

	cv::imshow("Gray", grayImage);
}

void splitImageRGB(const Mat &sourceImage) {
	std::vector<cv::Mat> channels;
	Mat zero = Mat::zeros(sourceImage.size(), CV_8UC1);
	
	cv::split(sourceImage, channels);
	
	std::vector<Mat> B = { channels[0], zero, zero };
	std::vector<Mat> G = { zero, channels[1], zero };
	std::vector<Mat> R = { zero, zero, channels[2] };
	
	Mat rdst, bdst, gdst;

	merge(B, bdst);
	merge(G, gdst);
	merge(R, rdst);


	cv::imshow("Channel B", bdst);
	cv::imshow("Channel G", gdst);
	cv::imshow("Channel R", rdst);
}

void gradientProcess(const Mat &sourceImage) {
	srand(time(NULL));
	//get V channel

	std::vector<cv::Mat> channels;
	cv::split(sourceImage, channels);
	Mat vChannel = channels[2];

	Mat destinationImage = Mat(vChannel.rows, vChannel.cols, CV_8UC3);
	Mat imagenLoca = Mat(vChannel.rows, vChannel.cols, CV_8UC3);

	Mat xImg = Mat(vChannel.rows, vChannel.cols, CV_8UC1);
	Mat xKernel = (Mat_<float>(1, 3) << 1, 0, -1);
	Mat yImg = Mat(vChannel.rows, vChannel.cols, CV_8UC1);
	Mat yKernel = (Mat_<float>(3, 1) << 1, 0, -1);

	filter2D(vChannel, xImg, -1, xKernel, Point(-1, -1), 0, BORDER_DEFAULT);
	filter2D(vChannel, yImg, -1, yKernel, Point(-1, -1), 0, BORDER_DEFAULT);

	Mat magnitudeImage = Mat(vChannel.rows, vChannel.cols, CV_8UC1);

	////Aply kernel


	for (int y = 0; y < destinationImage.rows; y++) {
		for (int x = 0; x < destinationImage.cols; x++) {

			float gx = xImg.at<uchar>(y, x);
			float gy = yImg.at<uchar>(y, x);

			destinationImage.at<Vec3b>(y, x)[2] = gx;
			destinationImage.at<Vec3b>(y, x)[1] = gy;

			float degrees = atan(gy / gx) * 180 / PI;
			float degreesToPix = degrees * 255 / 360;
			destinationImage.at<Vec3b>(y, x)[0] = degreesToPix;

			float magnitude = sqrt((gx * gx) + (gy * gy));

			magnitudeImage.at<uchar>(y, x) = magnitude;

			float rloca = 0;
			float gloca = 0;
			float bloca = 0;

			if (magnitude < 32) {
				magnitude = 0;
			}
			else {
				magnitude = 255;
				float baseX = 50;
				float baseY = 50;
				rloca = 2 * ((x+y) / 2);
				gloca = 7 * y ;
				bloca = 5*x;
			}
			imagenLoca.at<Vec3b>(y, x)[0] = rloca;
			imagenLoca.at<Vec3b>(y, x)[1] = gloca;
			imagenLoca.at<Vec3b>(y, x)[2] = bloca;
		}
	}

	Mat intensityImage = Mat(vChannel.rows, vChannel.cols, CV_8UC1);
	int size = 3;
	//Now that we have the magnitude, compute best threshold
	for (int y = size; y <= magnitudeImage.rows - size; y++) {
		for (int x = size; x <= magnitudeImage.cols - size; x++) {

			int total = 0;
			int s = 0;
			//Local threshold
			for (int i = y - size; i <= y + size; i++) {
				for (int j = x - size; j <= x + size; j++) {

					total += magnitudeImage.at<uchar>(i, j);
					s++;
				}
			}
			float t = total / s;
			int intensity = magnitudeImage.at<uchar>(y, x);

			if (intensity <  (t + 10)) {
				intensityImage.at<uchar>(y, x) = 0;
			}
			else {
				intensityImage.at<uchar>(y, x) = 255;
			}

		}
	}

	cv::imshow("x", xImg);
	cv::imshow("y", yImg);
	cv::imshow("atan", destinationImage);
	//imshow("imagenloca", imagenLoca);
	cv::imshow("mag", magnitudeImage);
	cv::imshow("intensity", intensityImage);
}

void cannyEdgeDetection(const Mat &sourceImage) {

	Mat smoothImage = Mat(sourceImage.rows, sourceImage.cols, CV_8UC1);

	std::vector<cv::Mat> channels;
	cv::split(sourceImage, channels);
	Mat vChannel = channels[2];
	//Filter noise
	Mat smoothKernel = (Mat_<float>(5, 5) << 1, 4, 7, 4, 1, 4, 16, 26, 16, 4, 7, 26, 41, 26, 7, 4, 16, 26, 16, 4, 1, 4, 7, 4, 1);
	smoothKernel /= 273;

	filter2D(vChannel, smoothImage, -1, smoothKernel, Point(-1, -1), 0, BORDER_DEFAULT);

	Mat xKernel = (Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
	Mat xImg = Mat(vChannel.rows, vChannel.cols, CV_8UC1);
	filter2D(smoothImage, xImg, -1, xKernel, Point(-1, -1), 0, BORDER_DEFAULT);

	Mat yKernel = (Mat_<float>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
	Mat yImg = Mat(vChannel.rows, vChannel.cols, CV_8UC1);
	filter2D(smoothImage, yImg, -1, yKernel, Point(-1, -1), 0, BORDER_DEFAULT);

	//Compute gradients
	std::vector<std::vector<float>> gradientVector(vChannel.rows);
	for (int i = 0; i < vChannel.rows; i++) {
		gradientVector[i] = std::vector<float>(vChannel.cols);
	}

	std::vector<std::vector<float>> angleVector(vChannel.rows);
	for (int i = 0; i < vChannel.rows; i++) {
		angleVector[i] = std::vector<float>(vChannel.cols);
	}

	std::vector<std::vector<float>> supressVector(vChannel.rows);
	for (int i = 0; i < vChannel.rows; i++) {
		supressVector[i] = std::vector<float>(vChannel.cols);
	}

	//No max supression
	for (int y = 0; y < vChannel.rows; y++) {
		for (int x = 0; x < vChannel.cols; x++) {
			float xInt = xImg.at<uchar>(y,x);
			float yInt = yImg.at<uchar>(y, x);

			angleVector[y][x] = atan(yInt / xInt);
			gradientVector[y][x] = sqrt( (xInt*xInt) + (yInt*yInt) );
		}
	}

	Mat suppressedImg = Mat(vChannel.rows, vChannel.cols, CV_8UC1);
	// Now that we have computed gradient vector magnitudes and angles, supress
	for (int y = 0; y < vChannel.rows; y++) {
		for (int x = 0; +x < vChannel.cols; x++) {
			float gradient = 0;

			if ((x > 0 && y > 0 && x < vChannel.cols - 1 && y < vChannel.rows - 1)) {

				float angle = angleVector[y][x];
				if (angle < 0) {
					angle += 6.28319; // add 360 degrees
				}
				float intensity = gradientVector[y][x];

				//0 Degrees
				if((angle >= 5.8904862 && angle < 0.3926991) || (angle >= 2.7488936 && angle < 3.5342917) ) {
					
					if (intensity > gradientVector[y][x + 1] && intensity > gradientVector[y][x - 1]) {
						gradient = gradientVector[y][x];
					}
				}
				//45 Degrees
				else if ((angle >= 0.3926991 && angle < 1.178097) || (angle >= 3.5  && angle < 4.3196899)){
					if (intensity > gradientVector[y - 1][x + 1] && intensity > gradientVector[y + 1][x - 1]) {
						gradient = gradientVector[y][x];
					}
				}
				//90 Degrees
				else if ( (angle >= 1.17 && angle < 1.9634954) || (angle >= 4.3196899 && angle < 5.10508806)) {
					if (intensity > gradientVector[y - 1][x] && intensity > gradientVector[y + 1][x]) {
						gradient = gradientVector[y][x];
					}
				}
				//135 Degrees
				else if ((angle >= 1.9634954 && angle < 2.7488936) || (angle >= 5.10508806 && angle < 5.8904862)) {
					if (intensity > gradientVector[y - 1][x - 1] && intensity > gradientVector[y + 1][x + 1]) {
						gradient = gradientVector[y][x];
					}
				}
			}
			suppressedImg.at<uchar>(y, x) = gradient;
		}
	}

	cv::imshow("Smooth", smoothImage);
	cv::imshow("sup", suppressedImg);
}

void nativeCannyEdgeDetection(Mat &sourceImg) {
	Mat gray, detectedEdges, destinationImage;

	destinationImage.create(sourceImg.size(), sourceImg.type());
	cvtColor(sourceImg, gray, CV_BGR2GRAY);

	blur(gray, detectedEdges, Size(3, 3));

	//Compute global lowThreshold
	Mat xKernel = (Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
	Mat xImg = Mat(gray.rows, gray.cols, CV_8UC1);
	filter2D(gray, xImg, -1, xKernel, Point(-1, -1), 0, BORDER_DEFAULT);

	Mat yKernel = (Mat_<float>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
	Mat yImg = Mat(gray.rows, gray.cols, CV_8UC1);
	filter2D(gray, yImg, -1, yKernel, Point(-1, -1), 0, BORDER_DEFAULT);

	float totalMag = 0;
	float d = 0;
	for (int y = 0; y < gray.rows; y++) {
		for (int x = 0; x < gray.cols; x++) {
			float gx = xImg.at<uchar>(y, x);
			float gy = yImg.at<uchar>(y, x);
			float magnitude = sqrt((gx * gx) + (gy * gy));
			totalMag += magnitude;
			d++;
		}
	}
	int lowThreshold = totalMag / (d);

	int kernel_size = 3;
	Canny(detectedEdges, detectedEdges, lowThreshold + 32, lowThreshold+100, kernel_size);
	destinationImage = Scalar::all(0);
	gray.copyTo(destinationImage, detectedEdges);
	cv::imshow("Native Canny", destinationImage);
}

void histogram(const Mat &inputImage, std::vector<Vec3d> &histo)
{
	int i;
	double isize = 0;
	MatConstIterator_<Vec3b> it_in = inputImage.begin<Vec3b>(), it_in_end = inputImage.end<Vec3b>();
	for (i = 0; i < 256; ++i) histo[i][0] = histo[i][1] = histo[i][2] = 0;
	for (; it_in != it_in_end; ++it_in)
	{
		histo[(*it_in)[0]][0] = histo[(*it_in)[0]][0] + 1;
		histo[(*it_in)[1]][1] = histo[(*it_in)[1]][1] + 1;
		histo[(*it_in)[2]][2] = histo[(*it_in)[2]][2] + 1;
		++isize;
	}
	std::cout << "size=" << isize << " Freq=" << histo[10][0] << " ";
	if (isize > 0)
	{
		isize = 1.0 / isize;
		for (i = 0; i < 256; ++i)
		{
			histo[i][0] *= isize;
			histo[i][1] *= isize;
			histo[i][2] *= isize;
		}
		std::cout << histo[10][0] << "\n";
	}
}