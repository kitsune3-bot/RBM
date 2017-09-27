#include <iostream>
#include "rbmutil.h"
#include "GeneralizedRBM.h"
#include "GeneralizedRBMTrainer.h"

//
// 生成モデルと学習モデルとのカルバックライブラー情報量を比較
//
int main(void) {
	int vsize = 6;
	int hsize = 4;
	int datasize = 100;
	int epoch = 100;
	int cdk = 1;
	int batchsize = datasize;
	double learning_rate = 0.3;

	auto rbm_gen = GeneralizedRBM(vsize, hsize);
	auto rbm_exact = GeneralizedRBM(vsize, hsize+5);
	rbm_exact.setHiddenDiveSize(3);
	auto rbm_cd = GeneralizedRBM(vsize, hsize+5);
	rbm_cd.setHiddenDiveSize(3);

	auto rbm_trainer_exact = GeneralizedRBMTrainer(rbm_exact);
	rbm_trainer_exact.epoch = epoch;
	rbm_trainer_exact.cdk = cdk;
	rbm_trainer_exact.batchSize = batchsize;
	rbm_trainer_exact.learningRate = learning_rate;

	auto rbm_trainer_cd = GeneralizedRBMTrainer(rbm_cd);
	rbm_trainer_cd.epoch = epoch;
	rbm_trainer_cd.cdk = cdk;
	rbm_trainer_cd.batchSize = batchsize;
	rbm_trainer_cd.learningRate = learning_rate;

	std::vector<std::vector<double>> dataset;
	for (int i = 0; i < datasize; i++) {
		dataset.push_back(rbmutil::data_gen<GeneralizedRBM, std::vector<double> >(rbm_gen, vsize));
	}


	//std::cout << "[Exact]" << std::endl;
	//rbmutil::print_params(rbm_exact);
	//std::cout << "KLD: " << rbmutil::kld(rbm_gen, rbm_exact, std::vector<int>{0, 1}) << std::endl;
	//std::cout << "Logliklihood: " << rbm_trainer_exact.logLikeliHood(rbm_exact, dataset) << std::endl;
	//rbm_trainer_exact.trainExact(rbm_exact, dataset);
	//rbmutil::print_params(rbm_exact);
	//std::cout << "KLD: " << rbmutil::kld(rbm_gen, rbm_exact, std::vector<int>{0, 1}) << std::endl;
	//std::cout << "Logliklihood: " << rbm_trainer_exact.logLikeliHood(rbm_exact, dataset) << std::endl;

	std::cout << std::endl;

	std::cout << "[Contrastive Divergence]" << std::endl;
	rbmutil::print_params(rbm_cd);
	std::cout << "KLD: " << rbmutil::kld(rbm_gen, rbm_cd, std::vector<int>{0, 1}) << std::endl;
	std::cout << "Logliklihood: " << rbm_trainer_cd.logLikeliHood(rbm_cd, dataset) << std::endl;
	rbm_trainer_cd.trainCD(rbm_cd, dataset);
	rbmutil::print_params(rbm_cd);
	std::cout << "KLD: " << rbmutil::kld(rbm_gen, rbm_cd, std::vector<int>{0, 1}) << std::endl;
	std::cout << "Logliklihood: " << rbm_trainer_cd.logLikeliHood(rbm_cd, dataset) << std::endl;

	return 0;
}