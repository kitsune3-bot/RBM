#include <iostream>
#include "RBMCore.h"
#include "rbmutil.h"

int main(int argc, char **argv) {
	auto dataset = std::vector<std::vector<double>>();
	dataset.push_back(std::vector<double>({ 1,1,1,0,0,0,0,0,0,1 }));
	dataset.push_back(std::vector<double>({ 0,0,0,1,1,1,0,0,0,1 }));
	dataset.push_back(std::vector<double>({ 0,0,0,0,0,0,1,1,1,1 }));

	auto test_dataset = std::vector<std::vector<double>>();
	test_dataset.push_back(std::vector<double>({ 1,1,0,0,0,0,0,0,0,1 }));
	test_dataset.push_back(std::vector<double>({ 0,0,1,1,1,1,0,0,0,1 }));
	test_dataset.push_back(std::vector<double>({ 0,1,0,0,1,0,1,1,1,1 }));


	size_t vsize = 10;
	size_t hsize = 10;
	auto learning_rate = 0.1;
	auto cdk = 3;
	auto batch_size = 1;
	auto epoch = 10;


	auto rbm = GeneralizedRBM(vsize, hsize);
	rbm.setHiddenMax(1.0);
	rbm.setHiddenMin(-1.0);
	rbm.setHiddenDiveSize(3);
	rbm.setRealHiddenValue(true);

	auto rbm_cd = GeneralizedRBM(vsize, hsize);
	rbm.setHiddenMax(1.0);
	rbm.setHiddenMin(-1.0);
	rbm.setHiddenDiveSize(3);
	rbm.setRealHiddenValue(true);


	auto trainer = GeneralizedRBMTrainer(rbm);
	trainer.learningRate = learning_rate;
	trainer.cdk = cdk;
	trainer.batchSize = batch_size;
	trainer.epoch = epoch;
	trainer.trainExact(rbm, dataset);

	auto trainer_cd = GeneralizedRBMTrainer(rbm);
	trainer_cd.learningRate = learning_rate;
	trainer_cd.cdk = cdk;
	trainer_cd.batchSize = batch_size;
	trainer_cd.epoch = epoch;
	trainer_cd.trainExact(rbm, dataset);
	trainer_cd.trainCD(rbm_cd, dataset);

	auto set_data = [](auto & rbm, auto & data) {
		for (int i = 0; i < rbm.getVisibleSize(); i++) {
			rbm.nodes.v(i) = data[i];
		}
	};

	auto sampler = GeneralizedRBMSampler();
	for (int i = 0; i < dataset.size(); i++) {
		set_data(rbm, test_dataset[i]);
		set_data(rbm_cd, test_dataset[i]);

		sampler.updateByBlockedGibbsSamplingHidden(rbm);
		sampler.updateByBlockedGibbsSamplingVisible(rbm);

		sampler.updateByBlockedGibbsSamplingHidden(rbm_cd);
		sampler.updateByBlockedGibbsSamplingVisible(rbm_cd);


		std::cout << "-------------------" << std::endl;
		std::cout << "[test_data]" << std::endl;
		rbmutil::print_stl(test_dataset[i]);
		std::cout << "[train_data]" << std::endl;
		rbmutil::print_stl(dataset[i]);
		std::cout << "[inference(Exact)]" << std::endl;
		std::cout << rbm.nodes.v.transpose() << std::endl;
		std::cout << "[inference(CD)]" << std::endl;
		std::cout << rbm_cd.nodes.v.transpose() << std::endl;
	}

	return 0;
}