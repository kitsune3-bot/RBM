#pragma once
#include "ConditionalRBMSamplerBase.h"
#include "ConditionalGRBM.h"


class ConditionalGRBMSampler : ConditionalRBMSamplerBase {
public:
    ConditionalGRBMSampler();
    ~ConditionalGRBMSampler();

    // 可視変数一つをギブスサンプリング
    double gibbsSamplingVisible(ConditionalRBMBase & rbm, int vindex) { return gibbsSamplingVisible(reinterpret_cast<ConditionalGRBM &>(rbm), vindex); };
    double gibbsSamplingVisible(ConditionalGRBM & rbm, int vindex);

    // 隠れ変数一つをギブスサンプリング
    double gibbsSamplingHidden(ConditionalRBMBase & rbm, int hindex) { return gibbsSamplingHidden(reinterpret_cast<ConditionalGRBM &>(rbm), hindex); }
    double gibbsSamplingHidden(ConditionalGRBM & rbm, int hindex);

    // 可視層すべてをギブスサンプリング
    Eigen::VectorXd & blockedGibbsSamplingVisible(ConditionalRBMBase & rbm) { return blockedGibbsSamplingVisible(reinterpret_cast<ConditionalGRBM &>(rbm)); };
    Eigen::VectorXd & blockedGibbsSamplingVisible(ConditionalGRBM & rbm);

    // 隠れ層すべてをギブスサンプリング
    Eigen::VectorXd & blockedGibbsSamplingHidden(ConditionalRBMBase & rbm) { return blockedGibbsSamplingHidden(reinterpret_cast<ConditionalGRBM &>(rbm)); };
    Eigen::VectorXd & blockedGibbsSamplingHidden(ConditionalGRBM & rbm);

    // 可視変数一つをギブスサンプリングで更新
    double updateByGibbsSamplingVisible(ConditionalRBMBase & rbm, int vindex) { return updateByGibbsSamplingVisible(reinterpret_cast<ConditionalGRBM &>(rbm), vindex); };
    double updateByGibbsSamplingVisible(ConditionalGRBM & rbm, int vindex);

    // 隠れ変数一つをギブスサンプリングで更新
    double updateByGibbsSamplingHidden(ConditionalRBMBase & rbm, int hindex) { return updateByGibbsSamplingHidden(reinterpret_cast<ConditionalGRBM &>(rbm), hindex); };
    double updateByGibbsSamplingHidden(ConditionalGRBM & rbm, int hindex);

    // 可視層すべてをギブスサンプリングで更新
    Eigen::VectorXd & updateByBlockedGibbsSamplingVisible(ConditionalRBMBase & rbm) { return updateByBlockedGibbsSamplingVisible(reinterpret_cast<ConditionalGRBM &>(rbm)); };
    Eigen::VectorXd & updateByBlockedGibbsSamplingVisible(ConditionalGRBM & rbm);

    // 隠れ層すべてをギブスサンプリングで更新
    Eigen::VectorXd & updateByBlockedGibbsSamplingHidden(ConditionalRBMBase & rbm) { return updateByBlockedGibbsSamplingHidden(reinterpret_cast<ConditionalGRBM &>(rbm)); };
    Eigen::VectorXd & updateByBlockedGibbsSamplingHidden(ConditionalGRBM & rbm);
};

