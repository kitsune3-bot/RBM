#include <iostream>
#include "rbmutil.h"
#include "GeneralizedRBM.h"
#include "GeneralizedRBMTrainer.h"

//
// 生成モデルと学習モデルとのカルバックライブラー情報量を比較
//
int main(void) {
	int vsize = 6;
	int hsize = 10;
	int datasize = 1000;
	int epoch = 100;
	int batchsize = datasize/10;
	double learning_rate = 0.01;

	auto rbm_gen = GeneralizedRBM(vsize, hsize);
	auto rbm_exact = GeneralizedRBM(vsize, hsize);
	auto rbm_cd = GeneralizedRBM(vsize, hsize);

	auto rbm_trainer_exact = GeneralizedRBMTrainer(rbm_exact);
	rbm_trainer_exact.epoch = epoch;
	rbm_trainer_exact.cdk = 1;
	rbm_trainer_exact.batchSize = batchsize;
	rbm_trainer_exact.learningRate = learning_rate;

	auto rbm_trainer_cd = GeneralizedRBMTrainer(rbm_cd);
	rbm_trainer_cd.epoch = epoch;
	rbm_trainer_cd.cdk = 1;
	rbm_trainer_cd.batchSize = batchsize;
	rbm_trainer_cd.learningRate = learning_rate;

	std::vector<std::vector<double>> dataset;
	for (int i = 0; i < 1000; i++) {
		dataset.push_back(rbmutil::data_gen<GeneralizedRBM, std::vector<double> >(rbm_gen, 10));
	}


	std::cout << "[Exact]" << std::endl;
	std::cout << rbmutil::kld(rbm_gen, rbm_exact, std::vector<int>{0, 1}) << std::endl;
	rbm_trainer_exact.trainExact(rbm_exact, dataset);
	std::cout << rbmutil::kld(rbm_gen, rbm_exact, std::vector<int>{0, 1}) << std::endl;

	std::cout << std::endl;

	std::cout << "[Contrastive Divergence]" << std::endl;
	std::cout << rbmutil::kld(rbm_gen, rbm_cd, std::vector<int>{0, 1}) << std::endl;
	rbm_trainer_cd.trainCD(rbm_cd, dataset);
	std::cout << rbmutil::kld(rbm_gen, rbm_cd, std::vector<int>{0, 1}) << std::endl;

	return 0;
}