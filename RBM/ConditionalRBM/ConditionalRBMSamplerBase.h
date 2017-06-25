#pragma once
#include <Eigen/Core>

class ConditionalRBMBase;

class ConditionalRBMSamplerBase {
    // 可視変数一つをギブスサンプリング
    virtual double gibbsSamplingVisible(ConditionalRBMBase & rbm, int vindex) = 0;

    // 隠れ変数一つをギブスサンプリング
    virtual double gibbsSamplingHidden(ConditionalRBMBase & rbm, int hindex) = 0;

    // 可視層すべてをギブスサンプリング
    virtual Eigen::VectorXd & blockedGibbsSamplingVisible(ConditionalRBMBase & rbm) = 0;

    // 隠れ層すべてをギブスサンプリング
    virtual Eigen::VectorXd & blockedGibbsSamplingHidden(ConditionalRBMBase & rbm) = 0;

    // 可視変数一つをギブスサンプリングで更新
    virtual double updateByGibbsSamplingVisible(ConditionalRBMBase & rbm, int vindex) = 0;

    // 隠れ変数一つをギブスサンプリングで更新
    virtual double updateByGibbsSamplingHidden(ConditionalRBMBase & rbm, int hindex) = 0;

    // 可視層すべてをギブスサンプリングで更新
    virtual Eigen::VectorXd & updateByBlockedGibbsSamplingVisible(ConditionalRBMBase & rbm) = 0;

    // 隠れ層すべてをギブスサンプリングで更新
    virtual Eigen::VectorXd & updateByBlockedGibbsSamplingHidden(ConditionalRBMBase & rbm) = 0;
};