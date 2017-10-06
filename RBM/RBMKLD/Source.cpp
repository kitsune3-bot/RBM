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

		for (int epoch_count = 0; epoch_count < epoch; epoch_count++) {
			// Exact

			//std::cout << "[Exact]" << std::endl;
			//rbmutil::print_params(rbm_exact);
			//std::cout << "KLD: " << rbmutil::kld(rbm_gen, rbm_exact, std::vector<int>{0, 1}) << std::endl;
			//std::cout << "Logliklihood: " << rbm_trainer_exact.logLikeliHood(rbm_exact, dataset) << std::endl;
			rbm_trainer_exact.trainOnceExact(rbm_exact, dataset);
			//rbmutil::print_params(rbm_exact);
			//std::cout << "KLD: " << rbmutil::kld(rbm_gen, rbm_exact, std::vector<int>{0, 1}) << std::endl;
			//std::cout << "Logliklihood: " << rbm_trainer_exact.logLikeliHood(rbm_exact, dataset) << std::endl;
			std::stringstream ss_exact_fname;

			// filename format
			// ./output/{RBMTYPE}_v{VSIZE}_h{HSIZE}_{EPOCH}
			// FIXME: Trainerクラスでパラメータと学習情報のJSON作るように
			ss_exact_fname << "./output/" << i << "_exact" << "_v" << rbm_exact.getVisibleSize() << "_h" << rbm_exact.getHiddenSize() << "_epoch" << epoch_count << ".rbm.json";
			write_params(rbm_exact, ss_exact_fname.str());



			//std::cout << std::endl;

			// Contrastive Divergence

			//std::cout << "[Contrastive Divergence]" << std::endl;
			//rbmutil::print_params(rbm_cd);
			//std::cout << "KLD: " << rbmutil::kld(rbm_gen, rbm_cd, std::vector<int>{0, 1}) << std::endl;
			//std::cout << "Logliklihood: " << rbm_trainer_cd.logLikeliHood(rbm_cd, dataset) << std::endl;
			rbm_trainer_cd.trainOnceCD(rbm_cd, dataset);
			//rbmutil::print_params(rbm_cd);
			//std::cout << "KLD: " << rbmutil::kld(rbm_gen, rbm_cd, std::vector<int>{0, 1}) << std::endl;
			//std::cout << "Logliklihood: " << rbm_trainer_cd.logLikeliHood(rbm_cd, dataset) << std::endl;
			std::stringstream ss_cd_fname;

			// filename format
			// ./output/{RBMTYPE}_v{VSIZE}_h{HSIZE}_{EPOCH}
			// FIXME: Trainerクラスでパラメータと学習情報のJSON作るように
			ss_cd_fname << "./output/" << i << "_cd" << "_v" << rbm_cd.getVisibleSize() << "_h" << rbm_cd.getHiddenSize() << "_epoch" << epoch_count << ".rbm.json";
			write_params(rbm_cd, ss_cd_fname.str());
		}
	}
	return 0;
}