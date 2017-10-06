#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include "rbmutil.h"
#include "GeneralizedRBM.h"
#include "GeneralizedRBMTrainer.h"

// パラメータファイル書き込み
template <class RBM>
void write_params(RBM & rbm, std::string fname) {
	std::ofstream of(fname, std::ios_base::out | std::ios_base::trunc);
	of << rbm.params.serialize();
}

// 学習情報書き込み
template <class RBM, class Trainer>
void write_train_info(RBM & rbm, Trainer & trainer, std::string fname) {
	std::ofstream of(fname, std::ios_base::out | std::ios_base::trunc);
	auto js = trainer.trainInfoJson(rbm);
	of << js;
}

// (汎化|訓練)誤差情報書き込み
template <class RBMGEN, class RBMTRAIN, class Trainer, class Dataset>
void write_error_info(RBMGEN & rbm_gen, RBMTRAIN & rbm_train, Trainer & trainer, Dataset & dataset,  std::string fname) {
	std::ofstream of(fname, std::ios_base::out | std::ios_base::trunc);

	auto js = nlohmann::json();
	js["kld"] = rbmutil::kld(rbm_gen, rbm_train, std::vector<int>{0, 1});
	js["logLikelihood"] = trainer.logLikeliHood(rbm_train, dataset);
	
	of << js;
}


//
// 生成モデルと学習モデルとのカルバックライブラー情報量を比較
//
int main(void) {

	for (int i = 0; i < 1000; i++) {
		int vsize = 5;
		int hsize = 10;
		int datasize = 25;
		int epoch = 1000;
		int cdk = 1;
		int batchsize = datasize;
		double learning_rate = 0.1;
		double momentum_rate = 0.9;
		int div_size = 20;


		auto rbm_gen = GeneralizedRBM(vsize, hsize);
		rbm_gen.params.initParamsRandom(-2, 2);

		auto rbm_exact = GeneralizedRBM(vsize, hsize + 5);
		rbm_exact.params.initParamsRandom(-0.01, 0.01);
		rbm_exact.setHiddenDiveSize(div_size);
		rbm_exact.setHiddenMin(-1.0);
		rbm_exact.setHiddenMax(1.0);
		rbm_exact.setRealHiddenValue(true);

		auto rbm_trainer_exact = GeneralizedRBMTrainer(rbm_exact);
		rbm_trainer_exact.epoch = epoch;
		rbm_trainer_exact.cdk = cdk;
		rbm_trainer_exact.batchSize = batchsize;
		rbm_trainer_exact.learningRate = learning_rate;
		rbm_trainer_exact.momentumRate = momentum_rate;

		auto rbm_cd = GeneralizedRBM(vsize, hsize + 5);
		rbm_cd.params.initParamsRandom(-0.01, 0.01);
		rbm_cd.setHiddenDiveSize(div_size);
		rbm_cd.setHiddenMin(-1.0);
		rbm_cd.setHiddenMax(1.0);
		rbm_cd.setRealHiddenValue(true);

		auto rbm_trainer_cd = GeneralizedRBMTrainer(rbm_cd);
		rbm_trainer_cd.epoch = epoch;
		rbm_trainer_cd.cdk = cdk;
		rbm_trainer_cd.batchSize = batchsize;
		rbm_trainer_cd.learningRate = learning_rate;
		rbm_trainer_cd.momentumRate = momentum_rate;

		std::vector<std::vector<double>> dataset;
		for (int i = 0; i < datasize; i++) {
			dataset.push_back(rbmutil::data_gen<GeneralizedRBM, std::vector<double> >(rbm_gen, vsize));
		}

		//std::cout << "[Generative Model]" << std::endl;
		//rbmutil::print_params(rbm_gen);
		std::stringstream ss_gen_fname;
		ss_gen_fname << "./output/" << i << "_gen.rbm.json";
		write_params(rbm_gen, ss_gen_fname.str());

		//std::cout << "[Exact]" << std::endl;
		//rbmutil::print_params(rbm_exact);
		//std::cout << "KLD: " << rbmutil::kld(rbm_gen, rbm_exact, std::vector<int>{0, 1}) << std::endl;
		//std::cout << "Logliklihood: " << rbm_trainer_exact.logLikeliHood(rbm_exact, dataset) << std::endl;
		//std::cout << std::endl;

		//std::cout << "[Contrastive Divergence]" << std::endl;
		//rbmutil::print_params(rbm_cd);
		//std::cout << "KLD: " << rbmutil::kld(rbm_gen, rbm_cd, std::vector<int>{0, 1}) << std::endl;
		//std::cout << "Logliklihood: " << rbm_trainer_cd.logLikeliHood(rbm_cd, dataset) << std::endl;

		for (int epoch_count = 0; epoch_count < epoch; epoch_count++) {
			// Exact

			rbm_trainer_exact.trainOnceExact(rbm_exact, dataset);
			std::stringstream ss_exact_fname;
			ss_exact_fname << "./output/" << i << "_exact" << "_epoch" << epoch_count << ".train.json";
			write_train_info(rbm_exact, rbm_trainer_exact, ss_exact_fname.str());

			std::stringstream ss_exact_error_fname;
			ss_exact_error_fname << "./output/" << i << "_error_exact" << "_epoch" << epoch_count << ".error.json";
			write_error_info(rbm_gen, rbm_exact, rbm_trainer_exact, dataset, ss_exact_error_fname.str());




			// Contrastive Divergence
			rbm_trainer_cd.trainOnceCD(rbm_cd, dataset);
			std::stringstream ss_cd_fname;
			ss_cd_fname << "./output/" << i << "_cd" << "_epoch" << epoch_count << ".train.json";
			write_train_info(rbm_cd, rbm_trainer_cd, ss_cd_fname.str());

			std::stringstream ss_cd_error_fname;
			ss_cd_error_fname << "./output/" << i << "_error_cd" << "_epoch" << epoch_count << ".error.json";
			write_error_info(rbm_gen, rbm_cd, rbm_trainer_cd, dataset, ss_cd_error_fname.str());
		}
	}
	return 0;
}