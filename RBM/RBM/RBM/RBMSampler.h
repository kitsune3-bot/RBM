#pragma once
#include "../Sampler.h"
#include "RBM.h"
#include <random>

class RBM;

template<>
class Sampler<RBM> {
public:
	Sampler() = default;
	~Sampler() = default;

	// 可視変数一つをギブスサンプリング
	double gibbsSamplingVisible(RBM & rbm, int vindex);

	// 隠れ変数一つをギブスサンプリング
	double gibbsSamplingHidden(RBM & rbm, int hindex);

	// 可視層すべてをギブスサンプリング
	Eigen::VectorXd & blockedGibbsSamplingVisible(RBM & rbm);

	// 隠れ層すべてをギブスサンプリング
	Eigen::VectorXd & blockedGibbsSamplingHidden(RBM & rbm);

	// 可視変数一つをギブスサンプリングで更新
	double updateByGibbsSamplingVisible(RBM & rbm, int vindex);

	// 隠れ変数一つをギブスサンプリングで更新
	double updateByGibbsSamplingHidden(RBM & rbm, int hindex);

	// 可視層すべてをギブスサンプリングで更新
	Eigen::VectorXd & updateByBlockedGibbsSamplingVisible(RBM & rbm);

	// 隠れ層すべてをギブスサンプリングで更新
	Eigen::VectorXd & updateByBlockedGibbsSamplingHidden(RBM & rbm);
};



inline double Sampler<RBM>::gibbsSamplingVisible(RBM &rbm, int vindex) {
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<double> dist(0.0, 1.0);

	double value = rbm.condProbVis(vindex, 0.0) < dist(mt) ? 0.0 : 1.0;
	return value;
}

inline double Sampler<RBM>::gibbsSamplingHidden(RBM &rbm, int hindex) {
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<double> dist(0.0, 1.0);

	double value = rbm.condProbHid(hindex, 0.0) < dist(mt) ? 0.0 : 1.0;
	return value;
}

inline Eigen::VectorXd & Sampler<RBM>::blockedGibbsSamplingVisible(RBM &rbm) {
	auto vect = rbm.nodes.v;

	for (int i = 0; i < rbm.getVisibleSize(); i++) {
		vect(i) = gibbsSamplingVisible(rbm, i);
	}

	return vect;
}

inline Eigen::VectorXd & Sampler<RBM>::blockedGibbsSamplingHidden(RBM &rbm) {
	auto vect = rbm.nodes.h;

	for (int j = 0; j < rbm.getHiddenSize(); j++) {
		vect(j) = gibbsSamplingHidden(rbm, j);
	}

	return vect;
}

inline double Sampler<RBM>::updateByGibbsSamplingVisible(RBM &rbm, int vindex) {
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<double> dist(0.0, 1.0);

	double value = rbm.condProbVis(vindex, 0.0) < dist(mt) ? 0.0 : 1.0;
	rbm.nodes.v(vindex) = value;
	return value;
}

inline double Sampler<RBM>::updateByGibbsSamplingHidden(RBM &rbm, int hindex) {
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<double> dist(0.0, 1.0);

	double value = rbm.condProbHid(hindex, 0.0) < dist(mt) ? 0.0 : 1.0;
	rbm.nodes.h(hindex) = value;
	return value;
}

inline Eigen::VectorXd & Sampler<RBM>::updateByBlockedGibbsSamplingVisible(RBM &rbm) {

	for (int i = 0; i < rbm.getVisibleSize(); i++) {
		updateByGibbsSamplingVisible(rbm, i);
	}

	return rbm.nodes.v;
}

inline Eigen::VectorXd & Sampler<RBM>::updateByBlockedGibbsSamplingHidden(RBM &rbm) {
	for (int j = 0; j < rbm.getHiddenSize(); j++) {
		updateByGibbsSamplingHidden(rbm, j);
	}

	return rbm.nodes.h;
}
