#include "ConditionalRBMSampler.h"
#include <random>


ConditionalRBMSampler::ConditionalRBMSampler()
{
}


ConditionalRBMSampler::~ConditionalRBMSampler()
{
}

double ConditionalRBMSampler::gibbsSamplingVisible(ConditionalRBM &rbm, int vindex) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    double value = rbm.condProbVis(vindex, 0.0) < dist(mt) ? 0.0 : 1.0;
    return value;
}

double ConditionalRBMSampler::gibbsSamplingHidden(ConditionalRBM &rbm, int hindex) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    double value = rbm.condProbHid(hindex, 0.0) < dist(mt) ? 0.0 : 1.0;
    return value;
}

Eigen::VectorXd & ConditionalRBMSampler::blockedGibbsSamplingVisible(ConditionalRBM &rbm) {
    auto vect = rbm.nodes.v;

    for (int i = 0; i < rbm.getVisibleSize(); i++) {
        vect(i) = gibbsSamplingVisible(rbm, i);
    }

    return vect;
}

Eigen::VectorXd & ConditionalRBMSampler::blockedGibbsSamplingHidden(ConditionalRBM &rbm) {
    auto vect = rbm.nodes.h;

    for (int j = 0; j < rbm.getHiddenSize(); j++) {
        vect(j) = gibbsSamplingHidden(rbm, j);
    }

    return vect;
}

double ConditionalRBMSampler::updateByGibbsSamplingVisible(ConditionalRBM &rbm, int vindex) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    double value = rbm.condProbVis(vindex, 0.0) < dist(mt) ? 0.0 : 1.0;
    rbm.nodes.v(vindex) = value;
    return value;
}

double ConditionalRBMSampler::updateByGibbsSamplingHidden(ConditionalRBM &rbm, int hindex) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    double value = rbm.condProbHid(hindex, 0.0) < dist(mt) ? 0.0 : 1.0;
    rbm.nodes.h(hindex) = value;
    return value;
}

Eigen::VectorXd & ConditionalRBMSampler::updateByBlockedGibbsSamplingVisible(ConditionalRBM &rbm) {

    for (int i = 0; i < rbm.getVisibleSize(); i++) {
        updateByGibbsSamplingVisible(rbm, i);
    }

    return rbm.nodes.v;
}

Eigen::VectorXd & ConditionalRBMSampler::updateByBlockedGibbsSamplingHidden(ConditionalRBM &rbm) {
    for (int j = 0; j < rbm.getHiddenSize(); j++) {
        updateByGibbsSamplingHidden(rbm, j);
    }

    return rbm.nodes.h;
}
