#include "../include/imbase.h"
#include "../include/smaxreg.h"

#include <vector>
#include <algorithm>
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <cassert>

using namespace std;
using namespace cv;

namespace nnet
{
	SMaxReg::SMaxReg(const int epochs, const int numClasses, const double learningRate,
					 const double moment, const double regularizer, const double epsilon)
		: epochs(epochs)
		, numClasses(numClasses)
		, learningRate(learningRate)
		, moment(moment)
		, regularizer(regularizer)
		, epsilon(epsilon)
	{}

	SMaxReg::~SMaxReg()
	{
		this->weight.release();
	}

	void SMaxReg::train(const Mat &data, const Mat &label, int batchSize)
	{
		assert(data.type() == CV_32FC1 && label.type() == CV_32FC1);
		assert(batchSize >= 1);

		srand((uchar)time(NULL));

		Mat labelMatrix;
		realLabel2Matrix(label, labelMatrix);

		// initialize weight and bias
		int numData = data.rows;
		int dimData = data.cols;
		if (weight.empty() == true) {
			weight = Mat::zeros(dimData, numClasses, data.type());
			randu(weight, 0, 1); weight *= 0.001;
		}
		else {
			if (weight.rows != dimData || weight.cols != numClasses) {
				printf("Initial weight dimension wrong: weight.rows == dimData && weight.cols == numClasses!\n");
				abort();
			}
		}
		Mat velocity = Mat::zeros(weight.size(), weight.type());

		vector<int> index(numData, 0);
		for (int i = 0; i < numData; ++i) {
			index[i] = i;
		}
		

		Mat batchData, batchLabel; // batch data and label
		Mat rsp, maxRsp; // feed-forward response
		Mat prob, sumProb, logProb; // softmax prediction probability
		Mat gradient, ww; // update 
		
		bool isConverge = false; 
		double prevCost = 0;
		double currCost = 0; // cost
		double mom = 0.5; // moment
		int t = 0; // iteration
		int numBatches = floor(numData / batchSize);
		for (int ei = 0; ei < epochs; ++ei) {
			randperm(index);
			batchData = Mat::zeros(batchSize, dimData, data.type());
			batchLabel = Mat::zeros(batchSize, numClasses, labelMatrix.type());
			for (int batchIdx = 0; batchIdx < numBatches; ++batchIdx) {
				// batch SGD
				t++;
				if (t == 20)
					mom = moment;

				getBatchData(data, labelMatrix, index, batchSize, batchIdx, batchData, batchLabel);
				
// 				if (batchIdx == (numBatches - 1)) {
// 					int batchRange = data.rows - batchIdx*batchSize;
// 					batchData = batchData.rowRange(0, batchRange);
// 					batchLabel = batchLabel.rowRange(0, batchRange);
// 				}

				rsp = batchData * weight;
				reduce(rsp, maxRsp, 1, REDUCE_MAX, rsp.type());
				rsp -= repeat(maxRsp, 1, numClasses);
				exp(rsp, prob);
				reduce(prob, sumProb, 1, REDUCE_SUM, prob.type());
				prob = prob / repeat(sumProb, 1, numClasses);

				// compute gradient
				gradient = batchLabel - prob;
				gradient = batchData.t() * gradient;
				gradient = -gradient / batchData.rows;
				gradient += regularizer * weight;

				// update weight and bias
				velocity = mom * velocity + learningRate * gradient;
				weight -= velocity;

				// compute objective cost
				log(prob, logProb);
				logProb = batchLabel.mul(logProb);
				currCost = -(sum(logProb)[0] / batchData.rows);
				pow(weight, 2, ww);
				currCost += sum(ww)[0] * 0.5 * regularizer;
				printf("epoch %d: processing batch %d / %d cost %f\n", ei + 1, batchIdx + 1, numBatches, currCost);

				if (abs(currCost - prevCost) < epsilon && ei != 0) {
					printf("objective cost variation less than pre-defined %f\n", epsilon);
					isConverge = true;
					break;
				}
				prevCost = currCost;
			}

			batchData.release();
			batchLabel.release();
			if (isConverge == true) break;
		}

		if (isConverge == false) {
			printf("stopped by reaching maximum number of iterations\n");
		}

		// free and destroy space
		index.clear();
		labelMatrix.release();
		batchData.release(); 
		batchLabel.release(); 
		rsp.release();
		maxRsp.release(); 
		prob.release(); 
		sumProb.release();
		logProb.release();
		gradient.release();
		ww.release();
		velocity.release();
	}

	void SMaxReg::predict(const Mat &data, const Mat &label, Mat &predictLabel, 
					      Mat &prob, double &accuracy)
	{
		prob = data * weight;
		predictLabel = Mat::zeros(1, data.rows, CV_32FC1);
		int *maxLoc = (int *)calloc(2, sizeof(int));
		float *pptr = (float *)predictLabel.data;
		float *lptr = (float *)label.data;
		accuracy = 0;
		for (int i = 0; i < data.rows; ++i) {
			minMaxIdx(prob.row(i), NULL, NULL, NULL, maxLoc);
			*pptr++ = maxLoc[1];
			if ((*lptr++) == maxLoc[1])
				accuracy++;
		}
		accuracy /= data.rows;

		if (maxLoc != NULL) free(maxLoc); 
		maxLoc = NULL;
		lptr = NULL;
	}

	void SMaxReg::realLabel2Matrix(const Mat &label, Mat &labelMatrix)
	{
		assert(label.type() == CV_32FC1);

		labelMatrix = Mat::zeros(label.cols, numClasses, label.type());
		float *lmPtr = (float *)labelMatrix.data;
		float *lPtr = (float *)label.data;
		for (int i = 0; i < label.cols; i++) {
			lmPtr[i*labelMatrix.cols + (uchar)(*lPtr++)] = 1;
		}

		lmPtr = NULL;
		lPtr = NULL;
	}

	void SMaxReg::randperm(vector<int> &index)
	{
		srand((uchar)time(NULL));
		std::random_shuffle(index.begin(), index.end());
	}

	void SMaxReg::getBatchData(const Mat &data, const Mat &label, const vector<int> &index,
					           const int batchSize, const int batchIdx, 
							   Mat &batchData, Mat &batchLabel)
	{
		int batchStart = batchIdx * batchSize;
		int batchEnd = min((batchIdx + 1)*batchSize, data.rows);
		for (int i = batchStart; i < batchEnd; ++i) {
			data.row(index[i]).copyTo(batchData.row(i - batchStart));
			label.row(index[i]).copyTo(batchLabel.row(i - batchStart));
		}
	}
}