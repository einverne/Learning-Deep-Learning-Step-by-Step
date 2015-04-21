#ifndef _NNETS_IM2COL_H_
#define _NNETS_IM2COL_H_
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
		template <typename DType>
		void im2col(const Mat &image, const Size block, const Size stride, Mat &imcols, int padding = 0)
		{
			if (block.area() == 0 || block.height < 1 || block.width < 1) {
				printf("blockSize should be well initialized (width > 1 and height > 1)\n");
				abort();
			}

			Mat padImage;
			if (padding)
				copyMakeBorder(image, padImage, padding, padding, padding, padding, BORDER_CONSTANT, Scalar(0));
			else
				padImage = image;

			// initialize output col image
			int chns = padImage.channels();
			int rows = (padImage.rows + 2*padding - block.height) / stride.height + 1;
			int cols = (padImage.cols + 2*padding - block.width) / stride.width + 1;

			int numBlock = rows * cols;
			int dimBlock = block.height * block.width * chns;
			imcols = Mat_<DType>::zeros(numBlock, dimBlock);
			DType *ptr = (DType *)imcols.data;

#ifndef AT_MAT 
#define AT_MAT(r, c, ch) ((DType *)(padImage.data))[(r)*(padImage.cols)*(chns) + (c)*(chns) + (ch)]
#endif
			for (int r = 0; r < padImage.rows - block.height; r += stride.height) {
				for (int c = 0; c < padImage.cols - block.width; c += stride.width) {
					for (int br = 0; br < block.height; ++br) {
						for (int bc = 0; bc < block.width; ++bc) {
							for (int ch = 0; ch < chns; ++ch) {
								*ptr++ = AT_MAT(br + r, bc + c, ch);
							}
						}
					}
				}
			}

			ptr = NULL;
			padImage.release();
		}

		// explicit im2col template
		template void im2col<uchar>(const Mat &image, const Size block, const Size stride, Mat &imcols, int padding);
		template void im2col<float>(const Mat &image, const Size block, const Size stride, Mat &imcols, int padding);
		template void im2col<double>(const Mat &image, const Size block, const Size stride, Mat &imcols, int padding);



		template <typename DType>
		void col2im()
		{
		}
	}
}


#endif