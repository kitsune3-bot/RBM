#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include "rbmutil.h"
#include "GeneralizedRBM.h"
#include "GeneralizedRBMTrainer.h"

typedef struct {
	int vSize = 5;
	int hSize = 10;
	int appendH = 0;
	int datasize = 25;
	int epoch = 1000;
	int cdk = 1;
	int batchsize = datasize;
	double learningRate = 0.1;
	double momentumRate = 0.9;
	int divSize = 1;
	bool realFlag = false;
} OPTION;


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

// 実行ルーチン
template<class RBM, class DATASET>
void run(OPTION & option, int try_count, RBM & rbm_gen, DATASET & dataset) {
	auto rbm_exact = GeneralizedRBM(option.vSize, option.hSize + option.appendH);
	rbm_exact.params.initParamsRandom(-0.01, 0.01);
	rbm_exact.setHiddenDiveSize(option.divSize);
	rbm_exact.setHiddenMin(-1.0);
	rbm_exact.setHiddenMax(1.0);
	rbm_exact.setRealHiddenValue(option.realFlag);

	auto rbm_trainer_exact = GeneralizedRBMTrainer(rbm_exact);
	rbm_trainer_exact.epoch = option.epoch;
	rbm_trainer_exact.cdk = option.cdk;
	rbm_trainer_exact.batchSize = option.batchsize;
	rbm_trainer_exact.learningRate = option.learningRate;
	rbm_trainer_exact.momentumRate = option.momentumRate;

	auto rbm_cd = GeneralizedRBM(option.vSize, option.hSize + option.appendH);
	rbm_cd.params.initParamsRandom(-0.01, 0.01);
	rbm_cd.setHiddenDiveSize(option.divSize);
	rbm_cd.setHiddenMin(-1.0);
	rbm_cd.setHiddenMax(1.0);
	rbm_cd.setRealHiddenValue(option.realFlag);

	auto rbm_trainer_cd = GeneralizedRBMTrainer(rbm_cd);
	rbm_trainer_cd.epoch = option.epoch;
	rbm_trainer_cd.cdk = option.cdk;
	rbm_trainer_cd.batchSize = option.batchsize;
	rbm_trainer_cd.learningRate = option.learningRate;
	rbm_trainer_cd.momentumRate = option.momentumRate;

	//std::cout << "[Exact]" << std::endl;
	//rbmutil::print_params(rbm_exact);
	//std::cout << "KLD: " << rbmutil::kld(rbm_gen, rbm_exact, std::vector<int>{0, 1}) << std::endl;
	//std::cout << "Logliklihood: " << rbm_trainer_exact.logLikeliHood(rbm_exact, dataset) << std::endl;
	//std::cout << std::endl;

	//std::cout << "[Contrastive Divergence]" << std::endl;
	//rbmutil::print_params(rbm_cd);
	//std::cout << "KLD: " << rbmutil::kld(rbm_gen, rbm_cd, std::vector<int>{0, 1}) << std::endl;
	//std::cout << "Logliklihood: " << rbm_trainer_cd.logLikeliHood(rbm_cd, dataset) << std::endl;

	for (int epoch_count = 0; epoch_count < option.epoch; epoch_count++) {
		// Exact
		std::string rbm_div = option.realFlag ? "c" : std::to_string(option.divSize);

		rbm_trainer_exact.trainOnceExact(rbm_exact, dataset);
		std::stringstream ss_exact_fname;
		ss_exact_fname << "./output/" << try_count << "_exact" << "_epoch" << epoch_count << "_div" << rbm_div << ".train.json";
		write_train_info(rbm_exact, rbm_trainer_exact, ss_exact_fname.str());

		std::stringstream ss_exact_error_fname;
		ss_exact_error_fname << "./output/" << try_count << "_error_exact" << "_epoch" << epoch_count << "_div" << rbm_div << ".error.json";
		write_error_info(rbm_gen, rbm_exact, rbm_trainer_exact, dataset, ss_exact_error_fname.str());




		// Contrastive Divergence
		rbm_trainer_cd.trainOnceCD(rbm_cd, dataset);
		std::stringstream ss_cd_fname;
		ss_cd_fname << "./output/" << try_count << "_cd" << "_epoch" << epoch_count << "_div" << rbm_div << ".train.json";
		write_train_info(rbm_cd, rbm_trainer_cd, ss_cd_fname.str());

		std::stringstream ss_cd_error_fname;
		ss_cd_error_fname << "./output/" << try_count << "_error_cd" << "_epoch" << epoch_count << "_div" << rbm_div << ".error.json";
		write_error_info(rbm_gen, rbm_cd, rbm_trainer_cd, dataset, ss_cd_error_fname.str());
	}
}


//
// 生成モデルと学習モデルとのカルバックライブラー情報量を比較
//
int main(void) {

	std::cout << "append h:";
	int append_h;
	std::cin >> append_h;

	OPTION option;
	option.vSize = 5;
	option.hSize = 10;
	option.appendH = append_h;
	option.datasize = 25;
	option.epoch = 1000;
	option.cdk = 1;
	option.batchsize = option.datasize;
	option.learningRate = 0.1;
	option.momentumRate = 0.9;
	option.divSize = 1;
	option.realFlag = false;

	for (int try_count = 0; try_count < 1000; try_count++) {

		auto rbm_gen = GeneralizedRBM(option.vSize, option.hSize);
		rbm_gen.params.initParamsRandom(-2, 2);

		std::vector<std::vector<double>> dataset;
		for (int i = 0; i < option.datasize; i++) {
			dataset.push_back(rbmutil::data_gen<GeneralizedRBM, std::vector<double> >(rbm_gen, option.vSize));
		}

		//std::cout << "[Generative Model]" << std::endl;
		//rbmutil::print_params(rbm_gen);
		std::stringstream ss_gen_fname;
		ss_gen_fname << "./output/" << try_count << "_gen.rbm.json";
		write_params(rbm_gen, ss_gen_fname.str());

		// try rbm 2, 3, 4, 5, cont
		option.realFlag = false;
		option.divSize = 1;
		run(option, try_count, rbm_gen, dataset);

		option.realFlag = false;
		option.divSize = 2;
		run(option, try_count, rbm_gen, dataset);

		option.realFlag = false;
		option.divSize = 3;
		run(option, try_count, rbm_gen, dataset);

		option.realFlag = false;
		option.divSize = 4;
		run(option, try_count, rbm_gen, dataset);

		option.realFlag = true;
		run(option, try_count, rbm_gen, dataset);
	}
	return 0;
}