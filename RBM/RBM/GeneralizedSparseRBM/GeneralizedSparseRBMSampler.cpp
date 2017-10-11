#include "GeneralizedSparseRBMSampler.h"
#include "GeneralizedSparseRBM.h"
#include <random>


GeneralizedSparseRBMSampler::GeneralizedSparseRBMSampler()
{
}


GeneralizedSparseRBMSampler::~GeneralizedSparseRBMSampler()
{
}



double GeneralizedSparseRBMSampler::gibbsSamplingVisible(GeneralizedSparseRBM & rbm, int vindex) {
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<double> dist(0.0, 1.0);

	double value = dist(mt) < rbm.condProbVis(vindex, 0.0) ? 0.0 : 1.0;

	return value;
}

double GeneralizedSparseRBMSampler::gibbsSamplingHidden(GeneralizedSparseRBM & rbm, int hindex) {
	// 離散型
	auto sample_discrete = [&] {
		std::vector<double> probs(rbm.getHiddenValueSetSize());
		auto hidset = rbm.splitHiddenSet();
		for (int i = 0; i < hidset.size(); i++) {
			// FIXME: 分配関数の計算を使いまわせば高速化可能
			probs[i] = rbm.condProbHid(hindex, hidset[i]);
		}

		std::random_device rd;
		std::mt19937 mt(rd());
		std::discrete_distribution<> dist(probs.begin(), probs.end());

		double value = hidset[dist(mt)];

		return value;
	};

	// 連続型
	auto sample_real = [&] {
		// TODO: まだ導出していない
		throw;
		// 連続値は逆関数法で
		std::random_device rd;
		std::mt19937 mt(rd());
		std::uniform_real_distribution<double> dist(0.0, 1.0);
		auto u = dist(mt);
		auto h_max = rbm.getHiddenMax();
		auto h_min = rbm.getHiddenMin();

		auto mu_j = rbm.mu(hindex);
		auto z_j = (exp(h_max * mu_j) - exp(h_min * mu_j)) / mu_j;

		double value = log(z_j * u * mu_j + exp(h_min * mu_j)) / mu_j;

		return value;
	};

	auto value = rbm.isRealHiddenValue() ? sample_real() : sample_discrete();

	return value;
}

Eigen::VectorXd & GeneralizedSparseRBMSampler::blockedGibbsSamplingVisible(GeneralizedSparseRBM & rbm) {
	auto vect = rbm.nodes.v;

	for (int i = 0; i < rbm.getVisibleSize(); i++) {
		vect(i) = gibbsSamplingVisible(rbm, i);
	}

	return vect;
}

Eigen::VectorXd & GeneralizedSparseRBMSampler::blockedGibbsSamplingHidden(GeneralizedSparseRBM & rbm) {
	auto vect = rbm.nodes.h;

	for (int j = 0; j < rbm.getHiddenSize(); j++) {
		vect(j) = gibbsSamplingHidden(rbm, j);
	}

	return vect;
}

double GeneralizedSparseRBMSampler::updateByGibbsSamplingVisible(GeneralizedSparseRBM & rbm, int vindex) {
	auto value = gibbsSamplingVisible(rbm, vindex);
	rbm.nodes.v(vindex) = value;
	return value;
}

double GeneralizedSparseRBMSampler::updateByGibbsSamplingHidden(GeneralizedSparseRBM & rbm, int hindex) {
	auto value = gibbsSamplingHidden(rbm, hindex);
	rbm.nodes.h(hindex) = value;
	return value;
}

Eigen::VectorXd & GeneralizedSparseRBMSampler::updateByBlockedGibbsSamplingVisible(GeneralizedSparseRBM & rbm) {

	for (int i = 0; i < rbm.getVisibleSize(); i++) {
		updateByGibbsSamplingVisible(rbm, i);
	}

	return rbm.nodes.v;
}

Eigen::VectorXd & GeneralizedSparseRBMSampler::updateByBlockedGibbsSamplingHidden(GeneralizedSparseRBM & rbm) {
	for (int j = 0; j < rbm.getHiddenSize(); j++) {
		updateByGibbsSamplingHidden(rbm, j);
	}

	return rbm.nodes.h;
}
