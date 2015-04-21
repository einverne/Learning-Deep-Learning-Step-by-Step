// This is test for loading MNIST database
//

#include "../include/dataset.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;


const string trainImageFile = "data/MNIST/train-images.idx3-ubyte";
const string trainLabelFile = "data/MNIST/train-labels.idx1-ubyte";
const string testImageFile = "data/MNIST/t10k-images.idx3-ubyte";
const string testLabelFile = "data/MNIST/t10k-labels.idx1-ubyte";

int main()
{
	vector<Mat> trainDigits;
	Mat trainLabels;
	
	nnet::Dataset::MNIST imdb;
	imdb.loadDigits(trainImageFile, trainDigits);
	imdb.loadLabels(trainLabelFile, trainLabels);
	imdb.showDigits(trainDigits, 100);

	trainDigits.clear();
	trainLabels.release();
	return 0;
}


