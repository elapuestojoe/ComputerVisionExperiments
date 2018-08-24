#include <iostream>
#include <dos.h>
#include <conio.h>
#include <windows.h>
#include <time.h>

using namespace std;

/*
* Title: Flipped Image Sample (Cascaron)
* Class: Vision para Robot
* Instructor: Dr. Jose Luis Gordillo (http://robvis.mty.itesm.mx/~gordillo/)
* Code: Manlio Barajas (manlito@gmail.com)
* Institution: Tec de Monterrey, Campus Monterrey
* Date: January 10, 2012
*
* Description: This program takes input from a camera (recognizable by
* OpenCV) and it flips it horizontally. Two versions of a flipping method are
* provided. The "Efficient" version uses pointers to speed the operation, while
* "Basic" uses frequently the "Cv::Mat::At" method which slows down
* performance. This program has illustrative purposes, provided the existence
* of cv::flip method.
*
* TODO: Validate when source and destination image are the same
*
* This programs uses OpenCV http://opencv.willowgarage.com/wiki/
*/

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <stdio.h>
using namespace std;
using namespace cv;

void flipImageBasic(const Mat &sourceImage, Mat &destinationImage);
void swapColorsImageBasic(const Mat &sourceImage, Mat &destinationImage);
void convolucion(const Mat &sourceImage, Mat &destinationImage, const Mat &kernel);
double ImageSubstraction(Mat &sourceImage, const Mat &RefImage);
void ImageThreshold(const Mat &sourceImage, Mat &outImage, double umbral);
double otsu(const Mat &sourceImage);
double kmean(const Mat &sourceImage, Mat &labelImage, int k, std::vector<Vec3d> &start_colors);
double fuzzyCmean(const Mat &sourceImage, Mat &labelImage, int k, std::vector<Vec3d> &start_colors);
void regionLabel(const Mat &segmentedImage, Mat &labeledImage);


int main(int argc, char *argv[])
{

	/* First, open camera device */
	VideoCapture camera = VideoCapture(0);

	/* Create images where captured and transformed frames are going to be stored */
	Mat currentImage;
	Mat flipImage;
	Mat swapImage;


	//	int i=0;
	camera.read(currentImage);
	while (true)
	{
		/* Obtain a new frame from camera */
		camera.read(currentImage);

		// flip the images Left-right
		//        flipImageBasic(currentImage,flipImage);

		// Swap red-blue channels
		swapColorsImageBasic(currentImage, swapImage);

		/* Show images */
		imshow("Original", currentImage);
		//		imshow("flipped", flipImage);
		imshow("swapped", swapImage);

		if (waitKey(3) == 'x')
			break;
	}
}

/*
* This method flips horizontally the sourceImage into destinationImage. Because it uses
* "Mat::at" method, its performance is low (redundant memory access searching for pixels).
*/
void flipImageBasic(const Mat &sourceImage, Mat &destinationImage)
{
	if (destinationImage.empty())
		destinationImage = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type());

	for (int y = 0; y < sourceImage.rows; ++y)
		for (int x = 0; x < sourceImage.cols / 2; ++x)
			for (int i = 0; i < sourceImage.channels(); ++i)
			{
				destinationImage.at<Vec3b>(y, x)[i] = sourceImage.at<Vec3b>(y, sourceImage.cols - 1 - x)[i];
				destinationImage.at<Vec3b>(y, sourceImage.cols - 1 - x)[i] = sourceImage.at<Vec3b>(y, x)[i];
			}
}

void swapColorsImageBasic(const Mat &sourceImage, Mat &destinationImage)
{
	if (destinationImage.empty())
		destinationImage = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type());

	MatConstIterator_<Vec3b> it_in = sourceImage.begin<Vec3b>(), it_in_end = sourceImage.end<Vec3b>();
	MatIterator_<Vec3b> it_out = destinationImage.begin<Vec3b>();

	for (it_in = sourceImage.begin<Vec3b>(), it_out = destinationImage.begin<Vec3b>(); it_in != it_in_end; ++it_in, ++it_out)
	{
		(*it_out)[0] = (*it_in)[2];
		(*it_out)[1] = (*it_in)[1];
		(*it_out)[2] = (*it_in)[0];
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

double otsu(const Mat &sourceImage)
{
	if (sourceImage.empty()) return -1;

	MatConstIterator_<Vec3b> it_in = sourceImage.begin<Vec3b>(), it_in_end = sourceImage.end<Vec3b>();

	int i;
	double t, tm, umbral = 0;
	double mean_1, mean_2, sample_1, sample_2;
	double hist[256];
	unsigned char sig = 0;
	for (i = 0; i<255; i++)  hist[i] = 0;
	mean_2 = 0;
	for (; it_in != it_in_end; ++it_in)
	{
		sig = (unsigned char)(((float)(*it_in)[0] + (float)(*it_in)[1] + (float)(*it_in)[2]) / 3.0);
		mean_2 += sig;
		++hist[sig];
	}
	t = 0;
	tm = 0;
	mean_1 = sample_1 = 0;
	sample_2 = sourceImage.rows*sourceImage.cols;
	for (i = 0; i<255; i++)
	{
		mean_1 += i * hist[i];
		mean_2 -= i * hist[i];
		sample_1 += hist[i];
		sample_2 -= hist[i];
		t = (mean_1 - mean_2);
		t = sample_1 * sample_2*t*t;
		if (t>tm)
		{
			tm = t;
			umbral = i;
		}
	}
	return umbral;
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
		t = (*it_in)[0]>umbral;
		t |= (*it_in)[1]>umbral;
		t |= (*it_in)[2]>umbral;
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

	if (sourceImage.empty()) return -1;

	Mat sampImage;
	Mat labImage;
	int j, n, changes, currentClass, kclass;
	double distance, mindistance;


	resize(sourceImage, sampImage, Size(), 0.125, 0.125, INTER_NEAREST);
	labImage = sampImage.clone();

	//			imshow("Small", sampImage);


	MatConstIterator_<Vec3b> it_in = sampImage.begin<Vec3b>(), it_in_end = sampImage.end<Vec3b>();
	MatIterator_<Vec3b> it_label = labImage.begin<Vec3b>();


	Vec3d red(0.75, 0, 0);
	Vec3d green(0, 0.75, 0);
	Vec3d blue(0, 0, 0.75);
	Vec3d black(0, 0, 0);
	Vec3d white(0.75, 0.75, 0.75);
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
		for (n = 0; n<k; n++)
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
			for (n = 0; n<k; n++)
			{
				for (j = 0; j<3; j++) mean_colors[n][j] = (double)rand() / RAND_MAX;
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
		for (n = 0; n<k; n++)
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
			for (j = 0; j<3; j++)
			{
				dis[j] = (double)((*it_in)[j]) / 255.0 - mean_colors[0][j];
				mindistance += dis[j] * dis[j];
			}
			for (n = 1; n<k; n++)
			{
				distance = 0;
				for (j = 0; j<3; j++)
				{
					dis[j] = (double)((*it_in)[j]) / 255.0 - mean_colors[n][j];
					distance += dis[j] * dis[j];
				}
				if (distance<mindistance)
				{
					mindistance = distance;
					kclass = n;
				}
			}
			changes += (currentClass != kclass);
			(*it_label)[0] = kclass;
			++size_colors[kclass];
			for (j = 0; j<3; j++)
			{
				newmean_colors[kclass][j] += (double)((*it_in)[j]) / 255.0;
			}
		}
		for (n = 0; n<k; n++)
		{
			if (size_colors[n]>0)
			{
				for (j = 0; j<3; j++)
				{
					mean_colors[n][j] = newmean_colors[n][j] / size_colors[n];
				}
			}
		}
		//        imshow("SmallLabels", 20*labImage);
		printf("%d Changes=%d  (%d, %8.4f),(%d, %8.4f),(%d, %8.4f),(%d, %8.4f),(%d, %8.4f) \n", loops, changes, size_colors[0], mean_colors[0][0], size_colors[1], mean_colors[1][1], size_colors[2], mean_colors[2][2], size_colors[3], mean_colors[3][0], size_colors[4], mean_colors[4][0]);
		++loops;

	} while ((changes>100) && (loops<100));

	MatConstIterator_<Vec3b> it_inSource = sourceImage.begin<Vec3b>(), it_in_endSource = sourceImage.end<Vec3b>();
	MatIterator_<Vec3b> it_inLabel = labelImage.begin<Vec3b>();

	for (it_inSource = sourceImage.begin<Vec3b>(), it_inLabel = labelImage.begin<Vec3b>(); it_inSource != it_in_endSource; ++it_inSource, ++it_inLabel)
	{
		currentClass = 0;
		kclass = 0;
		mindistance = 0;
		for (j = 0; j<3; j++)
		{
			dis[j] = (double)((*it_inSource)[j]) / 255.0 - mean_colors[0][j];
			mindistance += dis[j] * dis[j];
		}
		for (n = 1; n<k; n++)
		{
			distance = 0;
			for (j = 0; j<3; j++)
			{
				dis[j] = (double)((*it_inSource)[j]) / 255.0 - mean_colors[n][j];
				distance += dis[j] * dis[j];
			}
			if (distance<mindistance)
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
	for (n = 0; n<k; n++)
	{
		start_colors[n] = mean_colors[n];
	}

	return 0;
}

// fuzzy c means segmentation of a color image
double fuzzyCmean(const Mat &sourceImage, Mat &labelImage, int k, std::vector<Vec3d> &start_colors)
{
	srand(time(NULL));

	if (sourceImage.empty()) return -1;

	Mat sampImage;
	Mat labImage;
	int j, n, m, changes, currentClass, kclass;
	double distance, mindistance, membership, maxMembership;
	std::vector<double> mdis;
	mdis.resize(k, 0);


	resize(sourceImage, sampImage, Size(), 0.125, 0.125, INTER_NEAREST);
	labImage = sampImage.clone();

	//			imshow("Small", sampImage);


	MatConstIterator_<Vec3b> it_in = sampImage.begin<Vec3b>(), it_in_end = sampImage.end<Vec3b>();
	MatIterator_<Vec3b> it_label = labImage.begin<Vec3b>();


	Vec3d red(0.75, 0, 0);
	Vec3d green(0, 0.75, 0);
	Vec3d blue(0, 0, 0.75);
	Vec3d black(0, 0, 0);
	Vec3d white(0.75, 0.75, 0.75);
	Vec3d dis;
	std::vector<Vec3d> mean_colors, newmean_colors, five_colors;
	std::vector<double> size_colors;
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
		for (n = 0; n<k; n++)
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
			for (n = 0; n<k; n++)
			{
				for (j = 0; j<3; j++) mean_colors[n][j] = (double)rand() / RAND_MAX;
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
		for (n = 0; n<k; n++)
		{
			size_colors[n] = 0;
			newmean_colors[n][0] = 0;
			newmean_colors[n][1] = 0;
			newmean_colors[n][2] = 0;
		}
		for (it_in = sampImage.begin<Vec3b>(), it_label = labImage.begin<Vec3b>(); it_in != it_in_end; ++it_in, ++it_label)
		{
			currentClass = (*it_label)[0] % k;
			maxMembership = 0;
			kclass = currentClass;
			for (n = 0; n<k; n++)
			{
				mdis[n] = 1.0e-10;
				for (j = 0; j<3; j++)
				{
					dis[j] = (double)((*it_in)[j]) / 255.0 - mean_colors[n][j];
					mdis[n] += dis[j] * dis[j];
				}
			}
			for (n = 0; n<k; n++)
			{
				membership = 0;
				for (m = 0; m<k; m++)
				{
					membership += pow(mdis[n] / mdis[m], 2);
				}
				membership = 1.0 / membership;
				if (maxMembership<membership)
				{
					kclass = n;
					maxMembership = membership;
				}
				for (j = 0; j<3; j++)
				{
					newmean_colors[n][j] += membership * ((double)((*it_in)[j]) / 255.0);
					size_colors[n] += membership;
				}
			}
			changes += (currentClass != kclass);
			(*it_label)[0] = kclass;
		}
		for (n = 0; n<k; n++)
		{
			if (size_colors[n]>0)
			{
				for (j = 0; j<3; j++)
				{
					mean_colors[n][j] = newmean_colors[n][j] / size_colors[n];
				}
			}
		}
		//        imshow("SmallLabels", 20*labImage);
		printf("%d Changes=%d  (%6.0f,%8.4f),(%6.0f,%8.4f),(%6.0f,%8.4f),(%6.4f,%8.4f),(%6.0f,%8.4f)\n", loops, changes, size_colors[0], mean_colors[0][0], size_colors[1], mean_colors[1][1], size_colors[2], mean_colors[2][2], size_colors[3], mean_colors[3][0], size_colors[4], mean_colors[4][0]);
		++loops;

	} while ((changes>100) && (loops<100));

	MatConstIterator_<Vec3b> it_inSource = sourceImage.begin<Vec3b>(), it_in_endSource = sourceImage.end<Vec3b>();
	MatIterator_<Vec3b> it_inLabel = labelImage.begin<Vec3b>();

	for (it_inSource = sourceImage.begin<Vec3b>(), it_inLabel = labelImage.begin<Vec3b>(); it_inSource != it_in_endSource; ++it_inSource, ++it_inLabel)
	{
		currentClass = 0;
		kclass = 0;
		mindistance = 0;
		for (j = 0; j<3; j++)
		{
			dis[j] = (double)((*it_inSource)[j]) / 255.0 - mean_colors[0][j];
			mindistance += dis[j] * dis[j];
		}
		for (n = 1; n<k; n++)
		{
			distance = 0;
			for (j = 0; j<3; j++)
			{
				dis[j] = (double)((*it_inSource)[j]) / 255.0 - mean_colors[n][j];
				distance += dis[j] * dis[j];
			}
			if (distance<mindistance)
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
		else
		{
			(*it_inLabel)[0] = five_colors[kclass][0] * 255;
			(*it_inLabel)[1] = five_colors[kclass][1] * 255;
			(*it_inLabel)[2] = five_colors[kclass][2] * 255;
		}
	}
	for (n = 0; n<k; n++)
	{
		start_colors[n] = mean_colors[n];
	}

	return 0;
}

void regionLabel(const Mat &segmentedImage, Mat &labeledImage)
{
	if (labeledImage.empty())
		labeledImage = Mat(segmentedImage.rows, segmentedImage.cols, segmentedImage.type());

	Mat intLablesImage = Mat(segmentedImage.rows, segmentedImage.cols, CV_16U);


	MatConstIterator_<Vec3b> it_in = segmentedImage.begin<Vec3b>(), it_in_end = segmentedImage.end<Vec3b>();
	MatIterator_<Vec3b> it_out = labeledImage.begin<Vec3b>();


	for (; it_out != (labeledImage.begin<Vec3b>() + segmentedImage.cols); ++it_in, ++it_out)
	{
		(*it_out)[0] = 0;
		(*it_out)[1] = 0;
		(*it_out)[2] = 0;
	}

	MatConstIterator_<Vec3b> it_in_izq = it_in - 1;
	MatConstIterator_<Vec3b> it_in_back = it_in - segmentedImage.cols;

	MatIterator_<Vec3b> it_label_izq = it_out - 1;
	MatIterator_<Vec3b> it_label_back = it_out - segmentedImage.cols;

	unsigned short labels = 0;
	unsigned short labels_lut[65536];
	for (unsigned int i = 0; i<65536; i++) labels_lut[i] = i;
	for (; it_in != it_in_end; ++it_in, ++it_out, ++it_in_izq, ++it_in_back, ++it_label_izq, ++it_label_back)
	{
		if (*it_in == *it_in_back)
		{
			*it_out = *it_label_back;
		}
		else
			if (*it_in == *it_in_izq)
			{
				*it_out = *it_label_izq;
			}
			else
			{
				++labels;
				(*it_out)[1] = (unsigned char)(16 * (labels % 16));
				(*it_out)[1] = (unsigned char)(16 * ((labels / 16) % 16));
				(*it_out)[2] = (unsigned char)(16 * ((labels / 256) % 16));
			}
	}
}
