#include "RBMSampler.h"
#include <random>


RBMSampler::RBMSampler()
{
}


RBMSampler::~RBMSampler()
{
}

double RBMSampler::gibbsSamplingVisible(RBM * rbm, int vindex) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    double value = rbm->condProbVis(vindex, 0.0) < dist(mt) ? 0.0 : 1.0;
    return value;
}

double RBMSampler::gibbsSamplingHidden(RBM * rbm, int hindex) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    double value = rbm->condProbHid(hindex, 0.0) < dist(mt) ? 0.0 : 1.0;
    return value;
}

Eigen::VectorXd & RBMSampler::blockedGibbsSamplingVisible(RBM * rbm) {
    auto vect = rbm->nodes.v;

    for (int i = 0; i < rbm->getVisibleSize(); i++) {
        vect(i) = gibbsSamplingVisible(rbm, i);
    }

    return vect;
}

Eigen::VectorXd & RBMSampler::blockedGibbsSamplingHidden(RBM * rbm) {
    auto vect = rbm->nodes.h;

    for (int j = 0; j < rbm->getHiddenSize(); j++) {
        vect(j) = gibbsSamplingHidden(rbm, j);
    }

    return vect;
}

double RBMSampler::updateByGibbsSamplingVisible(RBM * rbm, int vindex) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    double value = rbm->condProbVis(vindex, 0.0) < dist(mt) ? 0.0 : 1.0;
    rbm->nodes.v(vindex) = value;
    return value;
}

double RBMSampler::updateByGibbsSamplingHidden(RBM * rbm, int hindex) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    double value = rbm->condProbHid(hindex, 0.0) < dist(mt) ? 0.0 : 1.0;
    rbm->nodes.h(hindex) = value;
    return value;
}

Eigen::VectorXd & RBMSampler::updateByBlockedGibbsSamplingVisible(RBM * rbm) {

    for (int i = 0; i < rbm->getVisibleSize(); i++) {
        updateByGibbsSamplingVisible(rbm, i);
    }

    return rbm->nodes.v;
}

Eigen::VectorXd & RBMSampler::updateByBlockedGibbsSamplingHidden(RBM * rbm) {
    for (int j = 0; j < rbm->getHiddenSize(); j++) {
        updateByGibbsSamplingHidden(rbm, j);
    }

    return rbm->nodes.h;
}
