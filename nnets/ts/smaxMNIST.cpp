#include "../include/im2col.h"
#include "../include/imbase.h"
#include "../include/dataset.h"
#include "../include/smaxreg.h"

#include <cstring>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

void vec2Mat(vector<Mat> &vec, Mat &vecMat)
{
	int numImages = vec.size();
	int rows = vec[0].rows;
	int cols = vec[0].cols;
	Mat temp;
	vecMat = Mat::zeros(numImages, rows * cols, vec[0].type());
	for (int i = 0; i < numImages; ++i) {
		temp = vec[i].reshape(1, rows * cols).t();
		temp.copyTo(vecMat.row(i));
	}
}

void binarizeLabels(const Mat &labels, const int numClasses, Mat &biLabels)
{
	biLabels = Mat::zeros(labels.cols, numClasses, CV_32FC1);
	float *blptr = (float *)biLabels.data;
	uchar *lptr = (uchar *)labels.data;
	for (int i = 0; i < labels.cols; i++) {
		blptr[i*biLabels.cols + (*lptr++)] = 1;
	}

	blptr = NULL;
	lptr = NULL;
}

const string trainImageFile = "data/MNIST/train-images.idx3-ubyte";
const string trainLabelFile = "data/MNIST/train-labels.idx1-ubyte";
const string testImageFile = "data/MNIST/t10k-images.idx3-ubyte";
const string testLabelFile = "data/MNIST/t10k-labels.idx1-ubyte";

int main()
{
	vector<Mat> digits;
	Mat labels, biLabels, digitMat;

	// load MNIST database
	nnet::dataset::MNIST imdb;
	imdb.loadDigits(trainImageFile, digits);
	imdb.loadLabels(trainLabelFile, labels);
	
	// convert into classifier structure
	vec2Mat(digits, digitMat);
	nnet::imop::im2single(digitMat, digitMat);
	digits.clear();
	
	// params for Softmax classifier
	int numClasses = 10;
	int epochs = 10;
	int batchSize = 500;
	double learningRate = 0.1;
	double moment = 0.9;
	double regularizer = 0.001;
	double eplison = 1e-5;

	double initTime = (double)cv::getTickCount();
	nnet::SMaxReg classifier(epochs, numClasses, learningRate, moment, regularizer, eplison);
	classifier.train(digitMat, labels, batchSize);
	
	Mat predictions, probs;
	double accuracy;
	classifier.predict(digitMat, labels, predictions, probs, accuracy);
 	printf("training accuracy: %f%% cost time: %fs\n", accuracy * 100, ((double)cv::getTickCount()-initTime)/cv::getTickFrequency());
	
	digits.clear();
	labels.release();
	digitMat.release();

	imdb.loadDigits(testImageFile, digits);
	imdb.loadLabels(testLabelFile, labels);
	vec2Mat(digits, digitMat);
	nnet::imop::im2single(digitMat, digitMat);
	
	digits.clear();
	predictions.release();
	probs.release();
	accuracy = 0;
	
	initTime = (double)cv::getTickCount();
	classifier.predict(digitMat, labels, predictions, probs, accuracy);
	printf("testing accuracy: %f%% cost time: %fs\n", accuracy * 100, ((double)cv::getTickCount() - initTime) / cv::getTickFrequency());

}