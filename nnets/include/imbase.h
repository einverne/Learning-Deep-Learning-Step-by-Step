#ifndef _NNETS_IMBASE_H_
#define _NNETS_IMBASE_H_
#pragma once

#include <stdio.h>  // printf
#include <stdlib.h> // abort

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

namespace nnet
{
	namespace imop
	{
		inline void im2double(const Mat &image, Mat &imDouble)
		{
			// maybe we need lookup manner to speed up
			if (image.channels() == 1)
				image.convertTo(imDouble, CV_64FC1, 1.0 / 255);
			else
				image.convertTo(imDouble, CV_64FC3, 1.0 / 255);
		}

		inline void im2single(const Mat &image, Mat &imFloat)
		{
			if (image.channels() == 1)
				image.convertTo(imFloat, CV_32FC1, 1.0 / 255);
			else
				image.convertTo(imFloat, CV_32FC3, 1.0 / 255);
		}
	}
}



#endif