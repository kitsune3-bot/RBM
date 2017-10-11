#pragma once
#include "RBMSamplerBase.h"
#include "GeneralizedGRBM.h"


class GeneralizedGRBMSampler : RBMSamplerBase {
public:
    GeneralizedGRBMSampler();
    ~GeneralizedGRBMSampler();

    // 可視変数一つをギブスサンプリング
    double gibbsSamplingVisible(RBMBase & rbm, int vindex) { return gibbsSamplingVisible(reinterpret_cast<GeneralizedGRBM &>(rbm), vindex); };
    double gibbsSamplingVisible(GeneralizedGRBM & rbm, int vindex);

    // 隠れ変数一つをギブスサンプリング
    double gibbsSamplingHidden(RBMBase & rbm, int hindex) { return gibbsSamplingHidden(reinterpret_cast<GeneralizedGRBM &>(rbm), hindex); }
    double gibbsSamplingHidden(GeneralizedGRBM & rbm, int hindex);

    // 可視層すべてをギブスサンプリング
    Eigen::VectorXd & blockedGibbsSamplingVisible(RBMBase & rbm) { return blockedGibbsSamplingVisible(reinterpret_cast<GeneralizedGRBM &>(rbm)); };
    Eigen::VectorXd & blockedGibbsSamplingVisible(GeneralizedGRBM & rbm);

    // 隠れ層すべてをギブスサンプリング
    Eigen::VectorXd & blockedGibbsSamplingHidden(RBMBase & rbm) { return blockedGibbsSamplingHidden(reinterpret_cast<GeneralizedGRBM &>(rbm)); };
    Eigen::VectorXd & blockedGibbsSamplingHidden(GeneralizedGRBM & rbm);

    // 可視変数一つをギブスサンプリングで更新
    double updateByGibbsSamplingVisible(RBMBase & rbm, int vindex) { return updateByGibbsSamplingVisible(reinterpret_cast<GeneralizedGRBM &>(rbm), vindex); };
    double updateByGibbsSamplingVisible(GeneralizedGRBM & rbm, int vindex);

    // 隠れ変数一つをギブスサンプリングで更新
    double updateByGibbsSamplingHidden(RBMBase & rbm, int hindex) { return updateByGibbsSamplingHidden(reinterpret_cast<GeneralizedGRBM &>(rbm), hindex); };
    double updateByGibbsSamplingHidden(GeneralizedGRBM & rbm, int hindex);

    // 可視層すべてをギブスサンプリングで更新
    Eigen::VectorXd & updateByBlockedGibbsSamplingVisible(RBMBase & rbm) { return updateByBlockedGibbsSamplingVisible(reinterpret_cast<GeneralizedGRBM &>(rbm)); };
    Eigen::VectorXd & updateByBlockedGibbsSamplingVisible(GeneralizedGRBM & rbm);

    // 隠れ層すべてをギブスサンプリングで更新
    Eigen::VectorXd & updateByBlockedGibbsSamplingHidden(RBMBase & rbm) { return updateByBlockedGibbsSamplingHidden(reinterpret_cast<GeneralizedGRBM &>(rbm)); };
    Eigen::VectorXd & updateByBlockedGibbsSamplingHidden(GeneralizedGRBM & rbm);
};

