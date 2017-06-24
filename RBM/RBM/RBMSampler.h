#pragma once
#include "RBMSamplerBase.h"
#include "RBM.h"


class RBMSampler : RBMSamplerBase{
public:
    RBMSampler();
    ~RBMSampler();

    // 可視変数一つをギブスサンプリング
    double gibbsSamplingVisible(RBMBase & rbm, int vindex) { return gibbsSamplingVisible(reinterpret_cast<RBM &>(rbm), vindex); };
    double gibbsSamplingVisible(RBM & rbm, int vindex);

    // 隠れ変数一つをギブスサンプリング
    double gibbsSamplingHidden(RBMBase & rbm, int hindex) { return gibbsSamplingHidden(reinterpret_cast<RBM &>(rbm), hindex); }
    double gibbsSamplingHidden(RBM & rbm, int hindex);

    // 可視層すべてをギブスサンプリング
    Eigen::VectorXd & blockedGibbsSamplingVisible(RBMBase & rbm) { return blockedGibbsSamplingVisible(reinterpret_cast<RBM &>(rbm)); };
    Eigen::VectorXd & blockedGibbsSamplingVisible(RBM & rbm);

    // 隠れ層すべてをギブスサンプリング
    Eigen::VectorXd & blockedGibbsSamplingHidden(RBMBase & rbm) { return blockedGibbsSamplingHidden(reinterpret_cast<RBM &>(rbm)); };
    Eigen::VectorXd & blockedGibbsSamplingHidden(RBM & rbm);

    // 可視変数一つをギブスサンプリングで更新
    double updateByGibbsSamplingVisible(RBMBase & rbm, int vindex) { return updateByGibbsSamplingVisible(reinterpret_cast<RBM &>(rbm), vindex); };
    double updateByGibbsSamplingVisible(RBM & rbm, int vindex);

    // 隠れ変数一つをギブスサンプリングで更新
    double updateByGibbsSamplingHidden(RBMBase & rbm, int hindex) { return updateByGibbsSamplingHidden(reinterpret_cast<RBM &>(rbm), hindex); };
    double updateByGibbsSamplingHidden(RBM & rbm, int hindex);

    // 可視層すべてをギブスサンプリングで更新
    Eigen::VectorXd & updateByBlockedGibbsSamplingVisible(RBMBase & rbm) { return updateByBlockedGibbsSamplingVisible(reinterpret_cast<RBM &>(rbm)); };
    Eigen::VectorXd & updateByBlockedGibbsSamplingVisible(RBM & rbm);

    // 隠れ層すべてをギブスサンプリングで更新
    Eigen::VectorXd & updateByBlockedGibbsSamplingHidden(RBMBase & rbm) { return updateByBlockedGibbsSamplingHidden(reinterpret_cast<RBM &>(rbm)); };
    Eigen::VectorXd & updateByBlockedGibbsSamplingHidden(RBM & rbm);
};

