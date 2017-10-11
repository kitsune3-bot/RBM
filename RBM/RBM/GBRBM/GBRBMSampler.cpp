#include "GBRBMSampler.h"
#include <random>


GBRBMSampler::GBRBMSampler()
{
}


GBRBMSampler::~GBRBMSampler()
{
}

double GBRBMSampler::gibbsSamplingVisible(GBRBM &rbm, int vindex) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::normal_distribution<double> dist(rbm.meanVisible(vindex), sqrt(1/rbm.params.lambda(vindex)));

    double value = dist(mt);
    return value;
}

double GBRBMSampler::gibbsSamplingHidden(GBRBM &rbm, int hindex) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    double value = rbm.condProbHid(hindex, 0.0) < dist(mt) ? 0.0 : 1.0;
    return value;
}

Eigen::VectorXd & GBRBMSampler::blockedGibbsSamplingVisible(GBRBM &rbm) {
    auto vect = rbm.nodes.v;

    for (int i = 0; i < rbm.getVisibleSize(); i++) {
        vect(i) = gibbsSamplingVisible(rbm, i);
    }

    return vect;
}

Eigen::VectorXd & GBRBMSampler::blockedGibbsSamplingHidden(GBRBM &rbm) {
    auto vect = rbm.nodes.h;

    for (int j = 0; j < rbm.getHiddenSize(); j++) {
        vect(j) = gibbsSamplingHidden(rbm, j);
    }

    return vect;
}

double GBRBMSampler::updateByGibbsSamplingVisible(GBRBM &rbm, int vindex) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::normal_distribution<double> dist(rbm.meanVisible(vindex), sqrt(1 / rbm.params.lambda(vindex)));

    double value = dist(mt);
    rbm.nodes.v(vindex) = value;
    return value;
}

double GBRBMSampler::updateByGibbsSamplingHidden(GBRBM &rbm, int hindex) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    double value = rbm.condProbHid(hindex, 0.0) < dist(mt) ? 0.0 : 1.0;
    rbm.nodes.h(hindex) = value;
    return value;
}

Eigen::VectorXd & GBRBMSampler::updateByBlockedGibbsSamplingVisible(GBRBM &rbm) {

    for (int i = 0; i < rbm.getVisibleSize(); i++) {
        updateByGibbsSamplingVisible(rbm, i);
    }

    return rbm.nodes.v;
}

Eigen::VectorXd & GBRBMSampler::updateByBlockedGibbsSamplingHidden(GBRBM &rbm) {
    for (int j = 0; j < rbm.getHiddenSize(); j++) {
        updateByGibbsSamplingHidden(rbm, j);
    }

    return rbm.nodes.h;
}
