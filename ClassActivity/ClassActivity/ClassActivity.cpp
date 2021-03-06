// ClassActivity.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <iostream>
#include "opencv2\highgui.hpp"
#include "opencv2\imgproc.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/opencv.hpp"
using namespace cv;

void swapColors(Mat &sourceImage);
void histogramEqualization(Mat &sourceImage);
void histogram(const Mat &inputImage, std::vector<Vec3d> &histo);
void equaliza(const Mat &inputImage, Mat &eqImage, std::vector<Vec3d> &histo);
void histogramEqualization(Mat &sourceImage);
void hsvHistogramEqualization(Mat &sourceImage);
void equalizaHSV(const Mat &inputImage, Mat &destinationImage, std::vector<Vec3d> &histo);
void FIR(const Mat &inputImage, Mat &AvgImage, int n);
void mediana(const Mat &inputImage, Mat &AvgImage, int n);
void IIR(const Mat &inputImage, Mat &AvgImage, double alpha);
void convolucion(const Mat &sourceImage, Mat &destinationImage, const Mat &kernel);
double ImageSubstraction(Mat &sourceImage, const Mat &RefImage);
int main()
{
	VideoCapture camera(0);
	if (!camera.isOpened()) {
		return -1;
	}
	Mat currentImage;
	Mat currentImageHSV;
	Mat lastImage;
	Mat avgImage;
	int xPixels = -1;
	int yPixels;
	camera.read(currentImage);
	lastImage = currentImage;
	while (true) {
		camera.read(currentImage);
		imshow("Original", currentImage);

		
		//Get image pixels
		/*if (xPixels == -1) {
			xPixels = currentImage.cols;
			yPixels = currentImage.rows;
			std::cout << "X pixels" << xPixels << std::endl;
			std::cout << "Y pixels" << yPixels << std::endl;
		}*/

		//Image enhancement using live capture:
		//swapColors(currentImage);
		
		//histogram
		//histogramEqualization(currentImage);

		//HSV
		//cvtColor(currentImage, currentImageHSV, CV_BGR2HSV);
		//hsvHistogramEqualization(currentImageHSV);

		//FIR (TODO: probar 3 configuraciones)
		//Mat firImage = Mat(currentImage.rows, currentImage.cols, currentImage.type());
		//FIR(currentImage, firImage, 25);
		//imshow("FIR", firImage);

		//IIR
		//Mat iirImage = Mat(currentImage.rows, currentImage.cols, currentImage.type());
		//IIR(currentImage, iirImage, 0.05);
		//imshow("IIR", iirImage);

		// Median pass Convolution

		//Sharpening
		Mat sharpeningKernel = (Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
		Mat sharpenedImage = Mat(currentImage.rows, currentImage.cols, currentImage.type());
		convolucion(currentImage, sharpenedImage, sharpeningKernel);
		imshow("Sharpened", sharpenedImage);

		//Substraction
		ImageSubstraction(currentImage, sharpenedImage);
		imshow("Sub", currentImage);
		if (waitKey(3) == 'x') {
			break;
		}
	}
}

void swapColors(Mat &sourceImage) {
	Mat destinationImage = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type());
	
	for (int y = 0; y < sourceImage.rows; y++) {
		for (int x = 0; x < sourceImage.cols; x++) {
			for (int c = 0; c < sourceImage.channels(); c++) {
				int destC = c;
				if (c == 0) {
					destC = 2;
				}
				if (c == 2) {
					destC = 0;
				}
				destinationImage.at<Vec3b>(y, x)[destC] = sourceImage.at<Vec3b>(y, x)[c];
			}
		}
	}
	imshow("swapped", destinationImage);
}

void hsvHistogramEqualization(Mat &sourceImage) {
	imshow("HSV image", sourceImage);
	Mat destinationImage = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type());
	//HSV H- 0-180, S: 0-255, V:0-255
	std::vector<Vec3d> hist(256);
	histogram(sourceImage, hist);
	equalizaHSV(sourceImage, destinationImage, hist);
	imshow("HSV equalized", destinationImage);
}

void histogramEqualization(Mat &sourceImage) {
	Mat destinationImage = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type());
	std::vector<Vec3d> hist(256);
	histogram(sourceImage, hist);
	equaliza(sourceImage, destinationImage, hist);
	imshow("equalized", destinationImage);
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

void equaliza(const Mat &inputImage, Mat &eqImage, std::vector<Vec3d> &histo)
{
	int i;
	double lut[256][3];
	if (eqImage.empty())
		eqImage = Mat(inputImage.rows, inputImage.cols, inputImage.type());
	MatConstIterator_<Vec3b> it_in = inputImage.begin<Vec3b>(), it_in_end = inputImage.end<Vec3b>();
	MatIterator_<Vec3b> it_out = eqImage.begin<Vec3b>();
	lut[0][0] = histo[0][0];
	lut[0][1] = histo[0][1];
	lut[0][2] = histo[0][2];
	for (i = 1; i < 256; ++i)
	{
		lut[i][0] = lut[i - 1][0] + histo[i][0];
		lut[i][1] = lut[i - 1][1] + histo[i][1];
		lut[i][2] = lut[i - 1][2] + histo[i][2];
	}
	std::cout << "lut=" << 255 * lut[10][0] << "\n";
	for (; it_in != it_in_end; ++it_in, ++it_out)
	{
		(*it_out)[0] = 255 * lut[(*it_in)[0]][0];
		(*it_out)[1] = 255 * lut[(*it_in)[1]][1];
		(*it_out)[2] = 255 * lut[(*it_in)[2]][2];
	}
}

void equalizaHSV(const Mat &inputImage, Mat &eqImage, std::vector<Vec3d> &histo) {
	int i;
	double lut[256];
	if (eqImage.empty())
		eqImage = Mat(inputImage.rows, inputImage.cols, inputImage.type());

	MatConstIterator_<Vec3b> it_in = inputImage.begin<Vec3b>(), it_in_end = inputImage.end<Vec3b>();
	MatIterator_<Vec3b> it_out = eqImage.begin<Vec3b>();
	lut[0] = histo[0][2];
	for (i = 1; i < 256; ++i)
	{
		lut[i] = lut[i - 1] + histo[i][2];
	}
	
	for (; it_in != it_in_end; ++it_in, ++it_out)
	{
		(*it_out)[2] = 255 * lut[(*it_in)[2]];
	}
}

void FIR(const Mat &inputImage, Mat &AvgImage, int n)
{
	int i, m;
	double avg[3];
	static std::vector<Mat> lastframes(n);
	static int current_frame = -1;
	static int n_in = n;
	MatConstIterator_<Vec3b> it_in = inputImage.begin<Vec3b>(), it_in_end = inputImage.end<Vec3b>();
	std::vector< MatIterator_<Vec3b> > frame_out(n_in);
	if (current_frame < 0)
	{
		current_frame = 0;
		AvgImage = Mat(inputImage.rows, inputImage.cols, inputImage.type());
		for (i = 0; i < n_in; i++)
		{
			lastframes[i] = Mat(inputImage.rows, inputImage.cols, inputImage.type());
			frame_out[i] = lastframes[i].begin<Vec3b>();
			for (; it_in != it_in_end; ++it_in, ++frame_out[i])
			{
				(*frame_out[i]) = (*it_in);
			}
		}
	}
	MatIterator_<Vec3b> av_out = AvgImage.begin<Vec3b>(), it_out_end = AvgImage.end<Vec3b>();
	frame_out[current_frame] = lastframes[current_frame].begin<Vec3b>();
	for (it_in = inputImage.begin<Vec3b>(); it_in != it_in_end; ++it_in, ++frame_out[current_frame])
	{
		(*frame_out[current_frame]) = (*it_in);
	}
	av_out = AvgImage.begin<Vec3b>();
	for (i = 0; i < n_in; i++)
	{
		frame_out[i] = lastframes[i].begin<Vec3b>();
	}
	for (av_out = AvgImage.begin<Vec3b>(); av_out != it_out_end; ++av_out)
	{
		for (i = 0; i < 3; i++) avg[i] = 0;
		for (m = 0; m < n_in; m++)
		{
			for (i = 0; i < 3; i++) avg[i] += (*frame_out[m])[i];
			++frame_out[m];
		}
		for (i = 0; i < 3; i++) (*av_out)[i] = (int)(avg[i] / n_in);
	}
	current_frame = (current_frame + 1) % n_in;
}

void mediana(const Mat &inputImage, Mat &AvgImage, int n)
{
	int i, m, stop, ctr[3];
	int median[3];
	static std::vector<Mat> lastframes(n);
	static int current_frame = -1;
	static int n_in = n;
	stop = n_in / 2;
	MatConstIterator_<Vec3b> it_in = inputImage.begin<Vec3b>(), it_in_end = inputImage.end<Vec3b>();
	std::vector< MatIterator_<Vec3b> > frame_out(n_in);
	if (current_frame < 0)
	{
		current_frame = 0;
		AvgImage = Mat(inputImage.rows, inputImage.cols, inputImage.type());
		for (i = 0; i < n_in; i++)
		{
			lastframes[i] = Mat(inputImage.rows, inputImage.cols, inputImage.type());
			frame_out[i] = lastframes[i].begin<Vec3b>();
			for (; it_in != it_in_end; ++it_in, ++frame_out[i])
			{
				(*frame_out[i]) = (*it_in);
			}
		}
	}
	MatIterator_<Vec3b> av_out = AvgImage.begin<Vec3b>(), it_out_end = AvgImage.end<Vec3b>();
	frame_out[current_frame] = lastframes[current_frame].begin<Vec3b>();
	for (it_in = inputImage.begin<Vec3b>(); it_in != it_in_end; ++it_in, ++frame_out[current_frame])
	{
		(*frame_out[current_frame]) = (*it_in);
	}
	av_out = AvgImage.begin<Vec3b>();
	for (i = 0; i < n_in; i++)
	{
		frame_out[i] = lastframes[i].begin<Vec3b>();
	}
	for (av_out = AvgImage.begin<Vec3b>(); av_out != it_out_end; ++av_out)
	{
		for (i = 0; i < 3; i++)
		{
			ctr[i] = 0;
			median[i] = (*frame_out[0])[i];
		}
		++frame_out[0];
		for (m = 1; m < n_in; m++)
		{
			for (i = 0; i < 3; i++)
			{
				if ((ctr[i] < stop) && (median[i] < (*frame_out[m])[i]))
				{
					median[i] = (*frame_out[m])[i];
					++ctr[i];
				}
			}
			++frame_out[m];
		}
		for (i = 0; i < 3; i++) (*av_out)[i] = median[i];
	}
	current_frame = (current_frame + 1) % n_in;
}

void IIR(const Mat &inputImage, Mat &AvgImage, double alpha)
{
	int i;
	MatConstIterator_<Vec3b> it_in = inputImage.begin<Vec3b>(), it_in_end = inputImage.end<Vec3b>();
	MatIterator_<Vec3b>  frame_out;
	if (AvgImage.empty())
	{
		AvgImage = Mat(inputImage.rows, inputImage.cols, inputImage.type());
		frame_out = AvgImage.begin<Vec3b>();
		for (; it_in != it_in_end; ++it_in, ++frame_out)
		{
			(*frame_out) = (*it_in);
		}
	}
	for (it_in = inputImage.begin<Vec3b>(), frame_out = AvgImage.begin<Vec3b>(); it_in != it_in_end; ++it_in, ++frame_out)
	{
		for (i = 0; i < 3; i++)
			(*frame_out)[i] = (int)(alpha*(*it_in)[i] + (1 - alpha)*(*frame_out)[i]);
	}
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

double ImageSubstraction(Mat &sourceImage, const Mat &RefImage)
{
	if (RefImage.empty()) return -1;
	MatConstIterator_<Vec3b> it_in = RefImage.begin<Vec3b>(), it_in_end = RefImage.end<Vec3b>();
	MatIterator_<Vec3b> it_out = sourceImage.begin<Vec3b>();
	double var = 0;
	double rb, gb, bb;
	for (; it_in != it_in_end; ++it_in, ++it_out)
	{
		rb = (double)(*it_out)[0] - (double)(*it_in)[0];
		gb = (double)(*it_out)[1] - (double)(*it_in)[1];
		bb = (double)(*it_out)[2] - (double)(*it_in)[2];
		var += (rb*rb + gb * gb + bb * bb);
		(*it_out)[0] = abs(rb);
		(*it_out)[1] = abs(gb);
		(*it_out)[2] = abs(bb);
	}
	return sqrt(var / (6 * sourceImage.rows*sourceImage.cols));
}