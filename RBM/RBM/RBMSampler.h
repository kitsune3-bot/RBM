#pragma once
#include "RBM.h"

class RBMSampler {
public:
    RBMSampler();
    ~RBMSampler();

    // 可視変数一つをギブスサンプリング
    double gibbsSamplingVisible(RBM * rbm, int vindex);

    // 隠れ変数一つをギブスサンプリング
    double gibbsSamplingHidden(RBM * rbm, int hindex);

    // 可視層すべてをギブスサンプリング
    Eigen::VectorXd & blockedGibbsSamplingVisible(RBM * rbm);

    // 隠れ層すべてをギブスサンプリング
    Eigen::VectorXd & blockedGibbsSamplingHidden(RBM * rbm);

    // 可視変数一つをギブスサンプリングで更新
    double updateByGibbsSamplingVisible(RBM * rbm, int vindex);

    // 隠れ変数一つをギブスサンプリングで更新
    double updateByGibbsSamplingHidden(RBM * rbm, int hindex);

    // 可視層すべてをギブスサンプリングで更新
    Eigen::VectorXd & updateByBlockedGibbsSamplingVisible(RBM * rbm);

    // 隠れ層すべてをギブスサンプリングで更新
    Eigen::VectorXd & updateByBlockedGibbsSamplingHidden(RBM * rbm);

};

