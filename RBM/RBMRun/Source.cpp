#include <iostream>
#include "GeneralizedRBM.h"
#include "GeneralizedRBMTrainer.h"
#include "GeneralizedRBMSampler.h"
#include "rbmutil.h"

int main(int argc, char **argv) {
	auto dataset = std::vector<std::vector<double>>();
	dataset.push_back(std::vector<double>({ 1,1,1,0,0,0,0,0,0,1 }));
	dataset.push_back(std::vector<double>({ 0,0,0,1,1,1,0,0,0,1 }));
	dataset.push_back(std::vector<double>({ 0,0,0,0,0,0,1,1,1,1 }));
	auto cond_dataset = dataset;

	size_t vsize = 10;
	size_t hsize = 20;

	auto rbm = GeneralizedRBM(vsize, hsize);
	rbm.setHiddenMax(1.0);
	rbm.setHiddenMin(-1.0);
	rbm.setHiddenDiveSize(1);
	rbm.setRealHiddenValue(false);

	auto trainer = GeneralizedRBMTrainer(rbm);
	trainer.learningRate = 0.1;
	trainer.cdk = 1;
	trainer.batchSize = 1;
	trainer.epoch = 50;

	trainer.trainExact(rbm, dataset);

	auto set_data = [&](auto data) {
		for (int i = 0; i < rbm.getVisibleSize(); i++) {
			rbm.nodes.v(i) = data[i];
		}
	};

	auto sampler = GeneralizedRBMSampler();
	for (int i = 0; i < dataset.size(); i++) {
		set_data(dataset[i]);

		sampler.updateByBlockedGibbsSamplingHidden(rbm);
		sampler.updateByBlockedGibbsSamplingVisible(rbm);

		rbmutil::print_stl(dataset[i]);
		std::cout << rbm.nodes.v.transpose() << std::endl;
	}

	return 0;
}