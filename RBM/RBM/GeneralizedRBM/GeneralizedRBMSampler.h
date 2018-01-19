#pragma once
#include "../Sampler.h"
#include "GeneralizedRBM.h"
#include "Eigen/Core"
#include <vector>
#include <numeric>
#include <random>

template<>
class Sampler<GeneralizedRBM> {
public:
	std::mt19937 randEngine = std::mt19937();
public:
	Sampler();
	~Sampler() = default;

	// 可視変数一つをギブスサンプリング
	double gibbsSamplingVisible(GeneralizedRBM & rbm, int vindex);

	// 隠れ変数一つをギブスサンプリング
	double gibbsSamplingHidden(GeneralizedRBM & rbm, int hindex);

	// 可視層すべてをギブスサンプリング
	Eigen::VectorXd & blockedGibbsSamplingVisible(GeneralizedRBM & rbm);

	// 隠れ層すべてをギブスサンプリング
	Eigen::VectorXd & blockedGibbsSamplingHidden(GeneralizedRBM & rbm);

	// 可視変数一つをギブスサンプリングで更新
	double updateByGibbsSamplingVisible(GeneralizedRBM & rbm, int vindex);

	// 隠れ変数一つをギブスサンプリングで更新
	double updateByGibbsSamplingHidden(GeneralizedRBM & rbm, int hindex);

	// 可視層すべてをギブスサンプリングで更新
	Eigen::VectorXd & updateByBlockedGibbsSamplingVisible(GeneralizedRBM & rbm);

	// 隠れ層すべてをギブスサンプリングで更新
	Eigen::VectorXd & updateByBlockedGibbsSamplingHidden(GeneralizedRBM & rbm);
};


inline Sampler<GeneralizedRBM>::Sampler() {
	std::random_device rd;
	this->randEngine = std::mt19937(rd());
}


inline double Sampler<GeneralizedRBM>::gibbsSamplingVisible(GeneralizedRBM & rbm, int vindex) {
	std::uniform_real_distribution<double> dist(0.0, 1.0);

	double value = dist(this->randEngine) < rbm.condProbVis(vindex, -1.0) ? -1.0 : 1.0;

	return value;
}

inline double Sampler<GeneralizedRBM>::gibbsSamplingHidden(GeneralizedRBM & rbm, int hindex) {
	// 離散型
	auto sample_discrete = [&] {
		std::vector<double> probs(rbm.getHiddenValueSetSize());
		auto hidset = rbm.splitHiddenSet();
		for (int i = 0; i < hidset.size(); i++) {
			// FIXME: 分配関数の計算を使いまわせば高速化可能
			probs[i] = rbm.condProbHid(hindex, hidset[i]);
		}

		std::discrete_distribution<> dist(probs.begin(), probs.end());

		double value = hidset[dist(this->randEngine)];

		return value;
	};

	// 連続型
	auto sample_real = [&] {
		// 連続値は逆関数法で
		std::uniform_real_distribution<double> dist(0.0, 1.0);
		auto u = dist(this->randEngine);
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

inline Eigen::VectorXd & Sampler<GeneralizedRBM>::blockedGibbsSamplingVisible(GeneralizedRBM & rbm) {
	auto vect = rbm.nodes.v;

	for (int i = 0; i < rbm.getVisibleSize(); i++) {
		vect(i) = gibbsSamplingVisible(rbm, i);
	}

	return vect;
}

inline Eigen::VectorXd & Sampler<GeneralizedRBM>::blockedGibbsSamplingHidden(GeneralizedRBM & rbm) {
	auto vect = rbm.nodes.h;

	for (int j = 0; j < rbm.getHiddenSize(); j++) {
		vect(j) = gibbsSamplingHidden(rbm, j);
	}

	return vect;
}

inline double Sampler<GeneralizedRBM>::updateByGibbsSamplingVisible(GeneralizedRBM & rbm, int vindex) {
	auto value = gibbsSamplingVisible(rbm, vindex);
	rbm.nodes.v(vindex) = value;
	return value;
}

inline double Sampler<GeneralizedRBM>::updateByGibbsSamplingHidden(GeneralizedRBM & rbm, int hindex) {
	auto value = gibbsSamplingHidden(rbm, hindex);
	rbm.nodes.h(hindex) = value;
	return value;
}

inline Eigen::VectorXd & Sampler<GeneralizedRBM>::updateByBlockedGibbsSamplingVisible(GeneralizedRBM & rbm) {

	for (int i = 0; i < rbm.getVisibleSize(); i++) {
		updateByGibbsSamplingVisible(rbm, i);
	}

	return rbm.nodes.v;
}

inline Eigen::VectorXd & Sampler<GeneralizedRBM>::updateByBlockedGibbsSamplingHidden(GeneralizedRBM & rbm) {
	for (int j = 0; j < rbm.getHiddenSize(); j++) {
		updateByGibbsSamplingHidden(rbm, j);
	}

	return rbm.nodes.h;
}
