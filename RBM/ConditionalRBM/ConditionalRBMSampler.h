#pragma once
#include "ConditionalRBMSamplerBase.h"
#include "ConditionalRBM.h"


class ConditionalRBMSampler : ConditionalRBMSamplerBase {
public:
    ConditionalRBMSampler();
    ~ConditionalRBMSampler();

    // 可視変数一つをギブスサンプリング
    double gibbsSamplingVisible(ConditionalRBMBase & rbm, int vindex) { return gibbsSamplingVisible(reinterpret_cast<ConditionalRBM &>(rbm), vindex); };
    double gibbsSamplingVisible(ConditionalRBM & rbm, int vindex);

    // 隠れ変数一つをギブスサンプリング
    double gibbsSamplingHidden(ConditionalRBMBase & rbm, int hindex) { return gibbsSamplingHidden(reinterpret_cast<ConditionalRBM &>(rbm), hindex); }
    double gibbsSamplingHidden(ConditionalRBM & rbm, int hindex);

    // 可視層すべてをギブスサンプリング
    Eigen::VectorXd & blockedGibbsSamplingVisible(ConditionalRBMBase & rbm) { return blockedGibbsSamplingVisible(reinterpret_cast<ConditionalRBM &>(rbm)); };
    Eigen::VectorXd & blockedGibbsSamplingVisible(ConditionalRBM & rbm);

    // 隠れ層すべてをギブスサンプリング
    Eigen::VectorXd & blockedGibbsSamplingHidden(ConditionalRBMBase & rbm) { return blockedGibbsSamplingHidden(reinterpret_cast<ConditionalRBM &>(rbm)); };
    Eigen::VectorXd & blockedGibbsSamplingHidden(ConditionalRBM & rbm);

    // 可視変数一つをギブスサンプリングで更新
    double updateByGibbsSamplingVisible(ConditionalRBMBase & rbm, int vindex) { return updateByGibbsSamplingVisible(reinterpret_cast<ConditionalRBM &>(rbm), vindex); };
    double updateByGibbsSamplingVisible(ConditionalRBM & rbm, int vindex);

    // 隠れ変数一つをギブスサンプリングで更新
    double updateByGibbsSamplingHidden(ConditionalRBMBase & rbm, int hindex) { return updateByGibbsSamplingHidden(reinterpret_cast<ConditionalRBM &>(rbm), hindex); };
    double updateByGibbsSamplingHidden(ConditionalRBM & rbm, int hindex);

    // 可視層すべてをギブスサンプリングで更新
    Eigen::VectorXd & updateByBlockedGibbsSamplingVisible(ConditionalRBMBase & rbm) { return updateByBlockedGibbsSamplingVisible(reinterpret_cast<ConditionalRBM &>(rbm)); };
    Eigen::VectorXd & updateByBlockedGibbsSamplingVisible(ConditionalRBM & rbm);

    // 隠れ層すべてをギブスサンプリングで更新
    Eigen::VectorXd & updateByBlockedGibbsSamplingHidden(ConditionalRBMBase & rbm) { return updateByBlockedGibbsSamplingHidden(reinterpret_cast<ConditionalRBM &>(rbm)); };
    Eigen::VectorXd & updateByBlockedGibbsSamplingHidden(ConditionalRBM & rbm);
};

