#include "pch.h"
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp> 
#include <time.h>
#include <vector>

using namespace cv;
using namespace std;
void Dilation(Mat &sourceImage, Mat &destinationImage, Mat &structuringElement);
void Erotion(Mat &sourceImage, Mat &destinationImage, Mat &structuringElement);
void SkeletonizeImage(Mat &sourceImage, Mat &destinationImage);
void MedialAxisTransform(Mat &sourceImage, Mat &distanceTransformImage, Mat &destinationImage);
void DistanceTransform(Mat &sourceImage, Mat &destinationImage);
void IsoData(Mat &segmentedImage, Mat &labeledImage, double umbral);
double fuzzyCmean(const Mat &sourceImage, Mat &labelImage, int k, std::vector<Vec3d> &start_colors);
double kmean(const Mat &sourceImage, Mat &labelImage, int k, std::vector<Vec3d> &start_colors);

int threshStart = 25;

int main()
{
	Mat image, skeletonImage, MAT, distanceTransform, isoDataImage, isoSourceImage, fuzzyCMeanImage, kMeansImage, kMeansCenters;

	image = imread("C:\\Users\\Joe\\Pictures\\cube.png", CV_LOAD_IMAGE_GRAYSCALE);
	//cv::resize(image, image, Size(), 0.25, 0.25, INTER_NEAREST);
	
	isoSourceImage = imread("C:\\Users\\Joe\\Pictures\\fruits.jpg", CV_LOAD_IMAGE_COLOR);
	cv::resize(isoSourceImage, isoSourceImage, Size(), 0.25, 0.25, INTER_NEAREST);

	Mat testImage = imread("C:\\Users\\Joe\\Pictures\\lena.jpg", CV_LOAD_IMAGE_COLOR);
	cv::resize(testImage, testImage, Size(), 0.25, 0.25, INTER_NEAREST);

	SkeletonizeImage(image, skeletonImage);
	DistanceTransform(image, distanceTransform);
	MedialAxisTransform(skeletonImage , distanceTransform, MAT);



	for (int i = 0; i < 50; i++) {
		Mat bgr[3];
		split(isoSourceImage, bgr);
		Mat out[3];
		IsoData(bgr[0], out[0], i);
		IsoData(bgr[1], out[1], i);
		IsoData(bgr[2], out[2], i);
		Mat outImage;
		cv::merge(out, 3, outImage);

		String name = "C:\\Users\\Joe\\Pictures\\ISO" + std::to_string(i) + ".png";
		cv::imwrite(name, outImage);
	}

	vector<Vec3d> v;
	fuzzyCmean(testImage, fuzzyCMeanImage, 5, v);
	cv::imwrite("C:\\Users\\Joe\\Pictures\\fuzzyC.jpg", fuzzyCMeanImage);
	//
	vector<Vec3d> v2;
	kmean(testImage, kMeansImage, 5, v2);
	cv::imwrite("C:\\Users\\Joe\\Pictures\\kmeans.jpg", kMeansImage);

	cv::imwrite("C:\\Users\\Joe\\Pictures\\skel1.jpg", image);
	cv::imwrite("C:\\Users\\Joe\\Pictures\\skel2.jpg", skeletonImage);
	cv::imwrite("C:\\Users\\Joe\\Pictures\\skelDist.jpg", distanceTransform);
	cv::imwrite("C:\\Users\\Joe\\Pictures\\skelMAT.jpg", MAT);

	while (true) {
		cv::imshow("Original", image);
		cv::imshow("Skeleton", skeletonImage);
		cv::imshow("Distance", distanceTransform);
		
		cv::imshow("MAT", MAT);
		
		cv::imshow("ISO-Orig", isoSourceImage);
		

		//cv::imshow("Orig", testImage);
		cv::imshow("Fuzzy", fuzzyCMeanImage);
		
		cv::imshow("kmeans", kMeansImage);
		if (waitKey(3) == 'x') {
			break;
		}
	}
	return 1;
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

void SkeletonizeImage(Mat &sourceImage, Mat &skeleton) {

	Mat tmpImage;

	MatSize imgSize = sourceImage.size;
	
	skeleton = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type(), Scalar(0));

	threshold(sourceImage, tmpImage, threshStart, 255, THRESH_BINARY);

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

void DistanceTransform(Mat &sourceImage, Mat &destinationImage) {
	Mat threshImage;
	threshold(sourceImage, threshImage, threshStart, 255, THRESH_BINARY);
	distanceTransform(threshImage, threshImage, DIST_L2, 3);
	normalize(threshImage, threshImage, 0, 1., NORM_MINMAX);
	destinationImage = threshImage;
}

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

double fuzzyCmean(const Mat &sourceImage, Mat &labelImage, int k, std::vector<Vec3d> &start_colors)
{
	srand(time(NULL));
	labelImage = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type(), Scalar(0));

	if (sourceImage.empty()) return -1;
	Mat sampImage;
	Mat labImage;
	int j, n, m, changes, currentClass, kclass;
	double distance, mindistance, membership, maxMembership;
	std::vector<double> mdis;
	mdis.resize(k, 0);
	resize(sourceImage, sampImage, Size(), 0.125, 0.125, INTER_NEAREST);
	labImage = sampImage.clone();
	//  imshow("Small", sampImage);
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
			maxMembership = 0;
			kclass = currentClass;
			for (n = 0; n < k; n++)
			{
				mdis[n] = 1.0e-10;
				for (j = 0; j < 3; j++)
				{
					dis[j] = (double)((*it_in)[j]) / 255.0 - mean_colors[n][j];
					mdis[n] += dis[j] * dis[j];
				}
			}
			for (n = 0; n < k; n++)
			{
				membership = 0;
				for (m = 0; m < k; m++)
				{
					membership += pow(mdis[n] / mdis[m], 2);
				}
				membership = 1.0 / membership;
				if (maxMembership < membership)
				{
					kclass = n;
					maxMembership = membership;
				}
				for (j = 0; j < 3; j++)
				{
					newmean_colors[n][j] += membership * ((double)((*it_in)[j]) / 255.0);
					size_colors[n] += membership;
				}
			}
			changes += (currentClass != kclass);
			(*it_label)[0] = kclass;
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
		printf("%d Changes=%d  (%6.0f,%8.4f),(%6.0f,%8.4f),(%6.0f,%8.4f),(%6.4f,%8.4f),(%6.0f,%8.4f)\n", loops, changes, size_colors[0], mean_colors[0][0], size_colors[1], mean_colors[1][1], size_colors[2], mean_colors[2][2], size_colors[3], mean_colors[3][0], size_colors[4], mean_colors[4][0]);
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

double kmean(const Mat &sourceImage, Mat &labelImage, int k, std::vector<Vec3d> &start_colors)
{
	srand(time(NULL));

	labelImage = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type(), Scalar(0));

	if (sourceImage.empty()) return -1;
	Mat sampImage;
	Mat labImage;
	int j, n, changes, currentClass, kclass;
	double distance, mindistance;
	resize(sourceImage, sampImage, Size(), 0.125, 0.125, INTER_NEAREST);
	labImage = sampImage.clone();

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
		printf("%d Changes=%d  (%d, %8.4f),(%d, %8.4f),(%d, %8.4f),(%d, %8.4f),(%d, %8.4f) \n", loops, changes, size_colors[0], mean_colors[0][0], size_colors[1], mean_colors[1][1], size_colors[2], mean_colors[2][2], size_colors[3], mean_colors[3][0], size_colors[4], mean_colors[4][0]);
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