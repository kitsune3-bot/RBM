#pragma once
#include "Eigen/Core"

class RBMBase;

class RBMSamplerBase {
    // 可視変数一つをギブスサンプリング
    virtual double gibbsSamplingVisible(RBMBase & rbm, int vindex) = 0;

    // 隠れ変数一つをギブスサンプリング
    virtual double gibbsSamplingHidden(RBMBase & rbm, int hindex) = 0;

    // 可視層すべてをギブスサンプリング
    virtual Eigen::VectorXd & blockedGibbsSamplingVisible(RBMBase & rbm) = 0;

    // 隠れ層すべてをギブスサンプリング
    virtual Eigen::VectorXd & blockedGibbsSamplingHidden(RBMBase & rbm) = 0;

    // 可視変数一つをギブスサンプリングで更新
    virtual double updateByGibbsSamplingVisible(RBMBase & rbm, int vindex) = 0;

    // 隠れ変数一つをギブスサンプリングで更新
    virtual double updateByGibbsSamplingHidden(RBMBase & rbm, int hindex) = 0;

    // 可視層すべてをギブスサンプリングで更新
    virtual Eigen::VectorXd & updateByBlockedGibbsSamplingVisible(RBMBase & rbm) = 0;

    // 隠れ層すべてをギブスサンプリングで更新
    virtual Eigen::VectorXd & updateByBlockedGibbsSamplingHidden(RBMBase & rbm) = 0;
};