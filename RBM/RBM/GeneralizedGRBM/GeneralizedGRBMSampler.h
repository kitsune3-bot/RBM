#pragma once
#include "../Sampler.h"
#include "GeneralizedGRBM.h"
#include <random>


template<>
class Sampler<GeneralizedGRBM> {
public:
    Sampler() = default;
    ~Sampler() = default;

    // 可視変数一つをギブスサンプリング
    double gibbsSamplingVisible(GeneralizedGRBM & rbm, int vindex);

    // 隠れ変数一つをギブスサンプリング
    double gibbsSamplingHidden(GeneralizedGRBM & rbm, int hindex);

    // 可視層すべてをギブスサンプリング
    Eigen::VectorXd & blockedGibbsSamplingVisible(GeneralizedGRBM & rbm);

    // 隠れ層すべてをギブスサンプリング
    Eigen::VectorXd & blockedGibbsSamplingHidden(GeneralizedGRBM & rbm);

    // 可視変数一つをギブスサンプリングで更新
    double updateByGibbsSamplingVisible(GeneralizedGRBM & rbm, int vindex);

    // 隠れ変数一つをギブスサンプリングで更新
    double updateByGibbsSamplingHidden(GeneralizedGRBM & rbm, int hindex);

    // 可視層すべてをギブスサンプリングで更新
    Eigen::VectorXd & updateByBlockedGibbsSamplingVisible(GeneralizedGRBM & rbm);

    // 隠れ層すべてをギブスサンプリングで更新
    Eigen::VectorXd & updateByBlockedGibbsSamplingHidden(GeneralizedGRBM & rbm);
};


inline double Sampler<GeneralizedGRBM>::gibbsSamplingVisible(GeneralizedGRBM &rbm, int vindex) {
	std::random_device rd;
	std::mt19937 mt(rd());
	std::normal_distribution<double> dist(rbm.meanVisible(vindex), sqrt(1 / rbm.params.lambda(vindex)));

	double value = dist(mt);
	return value;
}

inline double Sampler<GeneralizedGRBM>::gibbsSamplingHidden(GeneralizedGRBM &rbm, int hindex) {
	std::vector<double> probs(rbm.getHiddenValueSetSize());
	auto hidset = rbm.splitHiddenSet();
	for (int i = 0; i < hidset.size(); i++) {
		probs[i] = rbm.condProbHid(hindex, hidset[i]);
	}

	std::random_device rd;
	std::mt19937 mt(rd());
	std::discrete_distribution<> dist(probs.begin(), probs.end());

	double value = hidset[dist(mt)];
	return value;
}

inline Eigen::VectorXd & Sampler<GeneralizedGRBM>::blockedGibbsSamplingVisible(GeneralizedGRBM &rbm) {
	auto vect = rbm.nodes.v;

	for (int i = 0; i < rbm.getVisibleSize(); i++) {
		vect(i) = gibbsSamplingVisible(rbm, i);
	}

	return vect;
}

inline Eigen::VectorXd & Sampler<GeneralizedGRBM>::blockedGibbsSamplingHidden(GeneralizedGRBM &rbm) {
	auto vect = rbm.nodes.h;

	for (int j = 0; j < rbm.getHiddenSize(); j++) {
		vect(j) = gibbsSamplingHidden(rbm, j);
	}

	return vect;
}

inline double Sampler<GeneralizedGRBM>::updateByGibbsSamplingVisible(GeneralizedGRBM &rbm, int vindex) {
	std::random_device rd;
	std::mt19937 mt(rd());
	std::normal_distribution<double> dist(rbm.meanVisible(vindex), sqrt(1 / rbm.params.lambda(vindex)));

	double value = dist(mt);
	rbm.nodes.v(vindex) = value;
	return value;
}

inline double Sampler<GeneralizedGRBM>::updateByGibbsSamplingHidden(GeneralizedGRBM &rbm, int hindex) {
	std::vector<double> probs(rbm.getHiddenValueSetSize());
	auto hidset = rbm.splitHiddenSet();
	for (int i = 0; i < hidset.size(); i++) {
		probs[i] = rbm.condProbHid(hindex, hidset[i]);
	}

	std::random_device rd;
	std::mt19937 mt(rd());
	std::discrete_distribution<> dist(probs.begin(), probs.end());

	int index = dist(mt);
	double value = hidset[index];
	rbm.nodes.h(hindex) = value;
	return value;
}

inline Eigen::VectorXd & Sampler<GeneralizedGRBM>::updateByBlockedGibbsSamplingVisible(GeneralizedGRBM &rbm) {

	for (int i = 0; i < rbm.getVisibleSize(); i++) {
		updateByGibbsSamplingVisible(rbm, i);
	}

	return rbm.nodes.v;
}

inline Eigen::VectorXd & Sampler<GeneralizedGRBM>::updateByBlockedGibbsSamplingHidden(GeneralizedGRBM &rbm) {
	for (int j = 0; j < rbm.getHiddenSize(); j++) {
		updateByGibbsSamplingHidden(rbm, j);
	}

	return rbm.nodes.h;
}
