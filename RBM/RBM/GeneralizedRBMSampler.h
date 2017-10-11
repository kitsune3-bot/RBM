#pragma once
#include "RBMSamplerBase.h"
#include "Eigen/Core"

class GeneralizedRBM;

class GeneralizedRBMSampler : RBMSamplerBase {
public:
	GeneralizedRBMSampler();
	~GeneralizedRBMSampler();

	// 可視変数一つをギブスサンプリング
	double gibbsSamplingVisible(RBMBase & rbm, int vindex) { return gibbsSamplingVisible(reinterpret_cast<GeneralizedRBM &>(rbm), vindex); };
	double gibbsSamplingVisible(GeneralizedRBM & rbm, int vindex);

	// 隠れ変数一つをギブスサンプリング
	double gibbsSamplingHidden(RBMBase & rbm, int hindex) { return gibbsSamplingHidden(reinterpret_cast<GeneralizedRBM &>(rbm), hindex); }
	double gibbsSamplingHidden(GeneralizedRBM & rbm, int hindex);

	// 可視層すべてをギブスサンプリング
	Eigen::VectorXd & blockedGibbsSamplingVisible(RBMBase & rbm) { return blockedGibbsSamplingVisible(reinterpret_cast<GeneralizedRBM &>(rbm)); };
	Eigen::VectorXd & blockedGibbsSamplingVisible(GeneralizedRBM & rbm);

	// 隠れ層すべてをギブスサンプリング
	Eigen::VectorXd & blockedGibbsSamplingHidden(RBMBase & rbm) { return blockedGibbsSamplingHidden(reinterpret_cast<GeneralizedRBM &>(rbm)); };
	Eigen::VectorXd & blockedGibbsSamplingHidden(GeneralizedRBM & rbm);

	// 可視変数一つをギブスサンプリングで更新
	double updateByGibbsSamplingVisible(RBMBase & rbm, int vindex) { return updateByGibbsSamplingVisible(reinterpret_cast<GeneralizedRBM &>(rbm), vindex); };
	double updateByGibbsSamplingVisible(GeneralizedRBM & rbm, int vindex);

	// 隠れ変数一つをギブスサンプリングで更新
	double updateByGibbsSamplingHidden(RBMBase & rbm, int hindex) { return updateByGibbsSamplingHidden(reinterpret_cast<GeneralizedRBM &>(rbm), hindex); };
	double updateByGibbsSamplingHidden(GeneralizedRBM & rbm, int hindex);

	// 可視層すべてをギブスサンプリングで更新
	Eigen::VectorXd & updateByBlockedGibbsSamplingVisible(RBMBase & rbm) { return updateByBlockedGibbsSamplingVisible(reinterpret_cast<GeneralizedRBM &>(rbm)); };
	Eigen::VectorXd & updateByBlockedGibbsSamplingVisible(GeneralizedRBM & rbm);

	// 隠れ層すべてをギブスサンプリングで更新
	Eigen::VectorXd & updateByBlockedGibbsSamplingHidden(RBMBase & rbm) { return updateByBlockedGibbsSamplingHidden(reinterpret_cast<GeneralizedRBM &>(rbm)); };
	Eigen::VectorXd & updateByBlockedGibbsSamplingHidden(GeneralizedRBM & rbm);
};

