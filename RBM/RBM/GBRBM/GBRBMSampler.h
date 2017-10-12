#pragma once
#include "../Sampler.h"
#include "GBRBM.h"

template<>
class Sampler<GBRBM> {
public:
    Sampler() = default;
    ~Sampler() = default;

    // 可視変数一つをギブスサンプリング
    double gibbsSamplingVisible(GBRBM & rbm, int vindex);

    // 隠れ変数一つをギブスサンプリング
    double gibbsSamplingHidden(GBRBM & rbm, int hindex);

    // 可視層すべてをギブスサンプリング
    Eigen::VectorXd & blockedGibbsSamplingVisible(GBRBM & rbm);

    // 隠れ層すべてをギブスサンプリング
    Eigen::VectorXd & blockedGibbsSamplingHidden(GBRBM & rbm);

    // 可視変数一つをギブスサンプリングで更新
    double updateByGibbsSamplingVisible(GBRBM & rbm, int vindex);

    // 隠れ変数一つをギブスサンプリングで更新
    double updateByGibbsSamplingHidden(GBRBM & rbm, int hindex);

    // 可視層すべてをギブスサンプリングで更新
    Eigen::VectorXd & updateByBlockedGibbsSamplingVisible(GBRBM & rbm);

    // 隠れ層すべてをギブスサンプリングで更新
    Eigen::VectorXd & updateByBlockedGibbsSamplingHidden(GBRBM & rbm);
};


inline double Sampler<GBRBM>::gibbsSamplingVisible(GBRBM &rbm, int vindex) {
	std::random_device rd;
	std::mt19937 mt(rd());
	std::normal_distribution<double> dist(rbm.meanVisible(vindex), sqrt(1 / rbm.params.lambda(vindex)));

	double value = dist(mt);
	return value;
}

inline double Sampler<GBRBM>::gibbsSamplingHidden(GBRBM &rbm, int hindex) {
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<double> dist(0.0, 1.0);

	double value = rbm.condProbHid(hindex, 0.0) < dist(mt) ? 0.0 : 1.0;
	return value;
}

inline Eigen::VectorXd & Sampler<GBRBM>::blockedGibbsSamplingVisible(GBRBM &rbm) {
	auto vect = rbm.nodes.v;

	for (int i = 0; i < rbm.getVisibleSize(); i++) {
		vect(i) = gibbsSamplingVisible(rbm, i);
	}

	return vect;
}

inline Eigen::VectorXd & Sampler<GBRBM>::blockedGibbsSamplingHidden(GBRBM &rbm) {
	auto vect = rbm.nodes.h;

	for (int j = 0; j < rbm.getHiddenSize(); j++) {
		vect(j) = gibbsSamplingHidden(rbm, j);
	}

	return vect;
}

inline double Sampler<GBRBM>::updateByGibbsSamplingVisible(GBRBM &rbm, int vindex) {
	std::random_device rd;
	std::mt19937 mt(rd());
	std::normal_distribution<double> dist(rbm.meanVisible(vindex), sqrt(1 / rbm.params.lambda(vindex)));

	double value = dist(mt);
	rbm.nodes.v(vindex) = value;
	return value;
}

inline double Sampler<GBRBM>::updateByGibbsSamplingHidden(GBRBM &rbm, int hindex) {
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<double> dist(0.0, 1.0);

	double value = rbm.condProbHid(hindex, 0.0) < dist(mt) ? 0.0 : 1.0;
	rbm.nodes.h(hindex) = value;
	return value;
}

inline Eigen::VectorXd & Sampler<GBRBM>::updateByBlockedGibbsSamplingVisible(GBRBM &rbm) {

	for (int i = 0; i < rbm.getVisibleSize(); i++) {
		updateByGibbsSamplingVisible(rbm, i);
	}

	return rbm.nodes.v;
}

inline Eigen::VectorXd & Sampler<GBRBM>::updateByBlockedGibbsSamplingHidden(GBRBM &rbm) {
	for (int j = 0; j < rbm.getHiddenSize(); j++) {
		updateByGibbsSamplingHidden(rbm, j);
	}

	return rbm.nodes.h;
}
