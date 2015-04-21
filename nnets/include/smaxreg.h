#ifndef _NNETS_SMAXREG_H_
#define _NNETS_SMAXREG_H_
#pragma once

#include <vector>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

namespace nnet
{
	class SMaxReg
	{
	public:
		SMaxReg() {}
		SMaxReg(const int epochs, const int numClasses, const double learningRate,
				const double moment, const double regularizer, const double epsilon);
		~SMaxReg();

		// inline function of setter and getter
		inline void setInitWeight(const Mat &weight) 
		{
			weight.copyTo(this->weight); 
		}
		
		inline Mat getWeight() const 
		{
			return this->weight; 
		}

		// train
		void train(const Mat &data, const Mat &label, int batchSize);

		// prediction
		void predict(const Mat &data, const Mat &label, Mat &predictLabel, 
					 Mat &prob, double &accuracy);

	protected:
		void realLabel2Matrix(const Mat &label, Mat &biLabel);
		void randperm(vector<int> &index);
		void getBatchData(const Mat &data, const Mat &label, const vector<int> &index,
						  const int batchSize, const int batchIdx, 
						  Mat &batchData, Mat &batchLabel);

	private:
		Mat weight;
		int epochs;
		int numClasses;
		double epsilon;
		double learningRate;
		double moment;
		double regularizer;

	private:
		SMaxReg(const SMaxReg &rhs); // do not allow copy constructor
		const SMaxReg &operator = (const SMaxReg &); // nor assignement operator
	};
}



#endif // end of softmax regression