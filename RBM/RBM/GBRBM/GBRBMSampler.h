#pragma once
#include "RBMSamplerBase.h"
#include "GBRBM.h"


class GBRBMSampler : RBMSamplerBase {
public:
    GBRBMSampler();
    ~GBRBMSampler();

    // 可視変数一つをギブスサンプリング
    double gibbsSamplingVisible(RBMBase & rbm, int vindex) { return gibbsSamplingVisible(reinterpret_cast<GBRBM &>(rbm), vindex); };
    double gibbsSamplingVisible(GBRBM & rbm, int vindex);

    // 隠れ変数一つをギブスサンプリング
    double gibbsSamplingHidden(RBMBase & rbm, int hindex) { return gibbsSamplingHidden(reinterpret_cast<GBRBM &>(rbm), hindex); }
    double gibbsSamplingHidden(GBRBM & rbm, int hindex);

    // 可視層すべてをギブスサンプリング
    Eigen::VectorXd & blockedGibbsSamplingVisible(RBMBase & rbm) { return blockedGibbsSamplingVisible(reinterpret_cast<GBRBM &>(rbm)); };
    Eigen::VectorXd & blockedGibbsSamplingVisible(GBRBM & rbm);

    // 隠れ層すべてをギブスサンプリング
    Eigen::VectorXd & blockedGibbsSamplingHidden(RBMBase & rbm) { return blockedGibbsSamplingHidden(reinterpret_cast<GBRBM &>(rbm)); };
    Eigen::VectorXd & blockedGibbsSamplingHidden(GBRBM & rbm);

    // 可視変数一つをギブスサンプリングで更新
    double updateByGibbsSamplingVisible(RBMBase & rbm, int vindex) { return updateByGibbsSamplingVisible(reinterpret_cast<GBRBM &>(rbm), vindex); };
    double updateByGibbsSamplingVisible(GBRBM & rbm, int vindex);

    // 隠れ変数一つをギブスサンプリングで更新
    double updateByGibbsSamplingHidden(RBMBase & rbm, int hindex) { return updateByGibbsSamplingHidden(reinterpret_cast<GBRBM &>(rbm), hindex); };
    double updateByGibbsSamplingHidden(GBRBM & rbm, int hindex);

    // 可視層すべてをギブスサンプリングで更新
    Eigen::VectorXd & updateByBlockedGibbsSamplingVisible(RBMBase & rbm) { return updateByBlockedGibbsSamplingVisible(reinterpret_cast<GBRBM &>(rbm)); };
    Eigen::VectorXd & updateByBlockedGibbsSamplingVisible(GBRBM & rbm);

    // 隠れ層すべてをギブスサンプリングで更新
    Eigen::VectorXd & updateByBlockedGibbsSamplingHidden(RBMBase & rbm) { return updateByBlockedGibbsSamplingHidden(reinterpret_cast<GBRBM &>(rbm)); };
    Eigen::VectorXd & updateByBlockedGibbsSamplingHidden(GBRBM & rbm);
};

