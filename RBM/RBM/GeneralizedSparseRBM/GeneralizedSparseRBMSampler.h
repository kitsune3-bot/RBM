#pragma once
#include "../RBMSamplerBase.h"
#include "Eigen/Core"

class GeneralizedSparseRBM;

class GeneralizedSparseRBMSampler : RBMSamplerBase {
public:
	GeneralizedSparseRBMSampler();
	~GeneralizedSparseRBMSampler();

	// 可視変数一つをギブスサンプリング
	double gibbsSamplingVisible(RBMBase & rbm, int vindex) { return gibbsSamplingVisible(reinterpret_cast<GeneralizedSparseRBM &>(rbm), vindex); };
	double gibbsSamplingVisible(GeneralizedSparseRBM & rbm, int vindex);

	// 隠れ変数一つをギブスサンプリング
	double gibbsSamplingHidden(RBMBase & rbm, int hindex) { return gibbsSamplingHidden(reinterpret_cast<GeneralizedSparseRBM &>(rbm), hindex); }
	double gibbsSamplingHidden(GeneralizedSparseRBM & rbm, int hindex);

	// 可視層すべてをギブスサンプリング
	Eigen::VectorXd & blockedGibbsSamplingVisible(RBMBase & rbm) { return blockedGibbsSamplingVisible(reinterpret_cast<GeneralizedSparseRBM &>(rbm)); };
	Eigen::VectorXd & blockedGibbsSamplingVisible(GeneralizedSparseRBM & rbm);

	// 隠れ層すべてをギブスサンプリング
	Eigen::VectorXd & blockedGibbsSamplingHidden(RBMBase & rbm) { return blockedGibbsSamplingHidden(reinterpret_cast<GeneralizedSparseRBM &>(rbm)); };
	Eigen::VectorXd & blockedGibbsSamplingHidden(GeneralizedSparseRBM & rbm);

	// 可視変数一つをギブスサンプリングで更新
	double updateByGibbsSamplingVisible(RBMBase & rbm, int vindex) { return updateByGibbsSamplingVisible(reinterpret_cast<GeneralizedSparseRBM &>(rbm), vindex); };
	double updateByGibbsSamplingVisible(GeneralizedSparseRBM & rbm, int vindex);

	// 隠れ変数一つをギブスサンプリングで更新
	double updateByGibbsSamplingHidden(RBMBase & rbm, int hindex) { return updateByGibbsSamplingHidden(reinterpret_cast<GeneralizedSparseRBM &>(rbm), hindex); };
	double updateByGibbsSamplingHidden(GeneralizedSparseRBM & rbm, int hindex);

	// 可視層すべてをギブスサンプリングで更新
	Eigen::VectorXd & updateByBlockedGibbsSamplingVisible(RBMBase & rbm) { return updateByBlockedGibbsSamplingVisible(reinterpret_cast<GeneralizedSparseRBM &>(rbm)); };
	Eigen::VectorXd & updateByBlockedGibbsSamplingVisible(GeneralizedSparseRBM & rbm);

	// 隠れ層すべてをギブスサンプリングで更新
	Eigen::VectorXd & updateByBlockedGibbsSamplingHidden(RBMBase & rbm) { return updateByBlockedGibbsSamplingHidden(reinterpret_cast<GeneralizedSparseRBM &>(rbm)); };
	Eigen::VectorXd & updateByBlockedGibbsSamplingHidden(GeneralizedSparseRBM & rbm);
};

