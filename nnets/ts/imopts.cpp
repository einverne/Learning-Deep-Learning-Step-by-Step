#include "../include/im2col.h"
#include "../include/imbase.h"
#include "../include/dataset.h"

#include <iostream> // cout
#include <opencv2/core/core.hpp> // mat 
#include <opencv2/imgcodecs.hpp> // im opts
#include <opencv2/highgui/highgui.hpp> // GUI

using namespace std;
using namespace cv;


int main()
{
	Mat image = imread("data/im1.jpg", 1);
	Mat resizedImage(256, 256, CV_8UC3);
	resize(image, resizedImage, Size(256, 256));
	
	Mat colImage;
	nnet::imop::im2single(resizedImage, resizedImage);
	nnet::imop::im2col<float>(resizedImage, Size(6, 6), Size(9, 9), colImage, 0);

	Mat block = resizedImage(Range(0, 6), Range(0, 6));
	cout << block << endl << endl;
	cout << colImage.row(0) << endl << endl;
	return 0;
}