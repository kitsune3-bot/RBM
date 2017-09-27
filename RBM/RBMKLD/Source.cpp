#include <iostream>
#include "rbmutil.h"
#include "GeneralizedRBM.h"
#include "GeneralizedRBMTrainer.h"

//
// 生成モデルと学習モデルとのカルバックライブラー情報量を比較
//
int main(void) {
	int vsize = 5;
	int hsize = 10;
	auto rbm_gen = GeneralizedRBM(vsize, hsize);
	auto rbm_t = GeneralizedRBM(vsize, hsize);

	auto rbm_train = GeneralizedRBMTrainer(rbm_t);
	rbm_train.epoch = 1;
	rbm_train.cdk = 1;
	rbm_train.batchSize = 10;
	rbm_train.learningRate = 0.01;

	std::vector<std::vector<double>> dataset;
	for (int i = 0; i < 1000; i++) {
		dataset.push_back(rbmutil::data_gen<GeneralizedRBM, std::vector<double> >(rbm_gen, 10));
	}


	std::cout << rbmutil::kld(rbm_gen, rbm_t, std::vector<int>{0, 1}) << std::endl;
	rbm_train.trainExact(rbm_t, dataset);
	std::cout << rbmutil::kld(rbm_gen, rbm_t, std::vector<int>{0, 1}) << std::endl;


}