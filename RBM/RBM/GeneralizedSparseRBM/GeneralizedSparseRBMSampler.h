#pragma once
#include "../Sampler.h"
#include "GeneralizedSparseRBM.h"
#include "Eigen/Core"
#include <vector>
#include <numeric>
#include <random>

class GeneralizedSparseRBM;

template<>
class Sampler<GeneralizedSparseRBM> {
public:
	Sampler() = default;
	~Sampler() = default;

	// 可視変数一つをギブスサンプリング
	double gibbsSamplingVisible(GeneralizedSparseRBM & rbm, int vindex);

	// 隠れ変数一つをギブスサンプリング
	double gibbsSamplingHidden(GeneralizedSparseRBM & rbm, int hindex);

	// 可視層すべてをギブスサンプリング
	Eigen::VectorXd & blockedGibbsSamplingVisible(GeneralizedSparseRBM & rbm);

	// 隠れ層すべてをギブスサンプリング
	Eigen::VectorXd & blockedGibbsSamplingHidden(GeneralizedSparseRBM & rbm);

	// 可視変数一つをギブスサンプリングで更新
	double updateByGibbsSamplingVisible(GeneralizedSparseRBM & rbm, int vindex);

	// 隠れ変数一つをギブスサンプリングで更新
	double updateByGibbsSamplingHidden(GeneralizedSparseRBM & rbm, int hindex);

	// 可視層すべてをギブスサンプリングで更新
	Eigen::VectorXd & updateByBlockedGibbsSamplingVisible(GeneralizedSparseRBM & rbm);

	// 隠れ層すべてをギブスサンプリングで更新
	Eigen::VectorXd & updateByBlockedGibbsSamplingHidden(GeneralizedSparseRBM & rbm);
};



inline double Sampler<GeneralizedSparseRBM>::gibbsSamplingVisible(GeneralizedSparseRBM & rbm, int vindex) {
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<double> dist(0.0, 1.0);

	double value = dist(mt) < rbm.condProbVis(vindex, 0.0) ? 0.0 : 1.0;

	return value;
}

inline double Sampler<GeneralizedSparseRBM>::gibbsSamplingHidden(GeneralizedSparseRBM & rbm, int hindex) {
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

inline Eigen::VectorXd & Sampler<GeneralizedSparseRBM>::blockedGibbsSamplingVisible(GeneralizedSparseRBM & rbm) {
	auto vect = rbm.nodes.v;

	for (int i = 0; i < rbm.getVisibleSize(); i++) {
		vect(i) = gibbsSamplingVisible(rbm, i);
	}

	return vect;
}

inline Eigen::VectorXd & Sampler<GeneralizedSparseRBM>::blockedGibbsSamplingHidden(GeneralizedSparseRBM & rbm) {
	auto vect = rbm.nodes.h;

	for (int j = 0; j < rbm.getHiddenSize(); j++) {
		vect(j) = gibbsSamplingHidden(rbm, j);
	}

	return vect;
}

inline double Sampler<GeneralizedSparseRBM>::updateByGibbsSamplingVisible(GeneralizedSparseRBM & rbm, int vindex) {
	auto value = gibbsSamplingVisible(rbm, vindex);
	rbm.nodes.v(vindex) = value;
	return value;
}

inline double Sampler<GeneralizedSparseRBM>::updateByGibbsSamplingHidden(GeneralizedSparseRBM & rbm, int hindex) {
	auto value = gibbsSamplingHidden(rbm, hindex);
	rbm.nodes.h(hindex) = value;
	return value;
}

inline Eigen::VectorXd & Sampler<GeneralizedSparseRBM>::updateByBlockedGibbsSamplingVisible(GeneralizedSparseRBM & rbm) {

	for (int i = 0; i < rbm.getVisibleSize(); i++) {
		updateByGibbsSamplingVisible(rbm, i);
	}

	return rbm.nodes.v;
}

inline Eigen::VectorXd & Sampler<GeneralizedSparseRBM>::updateByBlockedGibbsSamplingHidden(GeneralizedSparseRBM & rbm) {
	for (int j = 0; j < rbm.getHiddenSize(); j++) {
		updateByGibbsSamplingHidden(rbm, j);
	}

	return rbm.nodes.h;
}
