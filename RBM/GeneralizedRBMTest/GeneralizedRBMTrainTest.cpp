#include "stdafx.h"

TEST(GeneralizeRBMTrainTest, TrainCDTest) {
	int vsize = 10;
	int hsize = 100;
	auto rbm = GeneralizedRBM(vsize, hsize);
	auto rbm_train = Trainer<GeneralizedRBM, OptimizerType::AdaMax>(rbm);

	auto dataset = std::vector< std::vector<double>>();
	dataset.push_back(std::vector<double>{ 0, 1, 0, 1, 0, 1, 0, 1, 0, 1 });
	dataset.push_back(std::vector<double>{ 1, 1, 1, 1, 0, 1, 0, 1, 0, 1 });
	dataset.push_back(std::vector<double>{ 0, 1, 0, 1, 1, 1, 1, 1, 0, 1 });
	dataset.push_back(std::vector<double>{ 0, 1, 0, 1, 0, 1, 1, 1, 1, 1 });
	dataset.push_back(std::vector<double>{ 0, 0, 0, 1, 0, 0, 0, 1, 0, 0 });

	rbm_train.epoch = 10;
	rbm_train.batchSize = 5;
	rbm_train.cdk = 1;

	rbm_train.trainCD(rbm, dataset);

	// chk params
	auto b = rbm.params.b.sum();
	ASSERT_FALSE(isnan(b));
	ASSERT_FALSE(isinf(b));

	auto c = rbm.params.c.sum();
	ASSERT_FALSE(isnan(c));
	ASSERT_FALSE(isinf(c));

	auto w = rbm.params.w.sum();
	ASSERT_FALSE(isnan(w));
	ASSERT_FALSE(isinf(w));

}


TEST(GeneralizeRBMTrainTest, TrainExactTest) {
	int vsize = 10;
	int hsize = 10;
	auto rbm = GeneralizedRBM(vsize, hsize);
	auto rbm_train = Trainer<GeneralizedRBM, OptimizerType::AdaMax>(rbm);

	auto dataset = std::vector< std::vector<double>>();
	dataset.push_back(std::vector<double>{ 0, 1, 0, 1, 0, 1, 0, 1, 0, 1 });
	dataset.push_back(std::vector<double>{ 1, 1, 1, 1, 0, 1, 0, 1, 0, 1 });
	dataset.push_back(std::vector<double>{ 0, 1, 0, 1, 1, 1, 1, 1, 0, 1 });
	dataset.push_back(std::vector<double>{ 0, 1, 0, 1, 0, 1, 1, 1, 1, 1 });
	dataset.push_back(std::vector<double>{ 0, 0, 0, 1, 0, 0, 0, 1, 0, 0 });

	rbm_train.epoch = 10;
	rbm_train.batchSize = 5;
	rbm_train.cdk = 1;

	rbm_train.trainExact(rbm, dataset);

	// chk params
	auto b = rbm.params.b.sum();
	ASSERT_FALSE(isnan(b));
	ASSERT_FALSE(isinf(b));

	auto c = rbm.params.c.sum();
	ASSERT_FALSE(isnan(c));
	ASSERT_FALSE(isinf(c));

	auto w = rbm.params.w.sum();
	ASSERT_FALSE(isnan(w));
	ASSERT_FALSE(isinf(w));

}