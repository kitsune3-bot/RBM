#include "GeneralizedGRBMSampler.h"
#include <random>


GeneralizedGRBMSampler::GeneralizedGRBMSampler()
{
}


GeneralizedGRBMSampler::~GeneralizedGRBMSampler()
{
}

double GeneralizedGRBMSampler::gibbsSamplingVisible(GeneralizedGRBM &rbm, int vindex) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::normal_distribution<double> dist(rbm.meanVisible(vindex), sqrt(1 / rbm.params.lambda(vindex)));

    double value = dist(mt);
    return value;
}

double GeneralizedGRBMSampler::gibbsSamplingHidden(GeneralizedGRBM &rbm, int hindex) {
    std::vector<double> probs(rbm.getHiddenValueSetSize());
    auto hidset = rbm.splitHiddenSet();
    for (int i = 0; i < hidset.size(); i++) {
        probs[i] = rbm.condProbHid(hindex, hidset[i]);
    }

    std::random_device rd;
    std::mt19937 mt(rd());
    std::discrete_distribution<> dist(probs.begin(), probs.end());

    double value = hidset[dist(mt)];
    return value;
}

Eigen::VectorXd & GeneralizedGRBMSampler::blockedGibbsSamplingVisible(GeneralizedGRBM &rbm) {
    auto vect = rbm.nodes.v;

    for (int i = 0; i < rbm.getVisibleSize(); i++) {
        vect(i) = gibbsSamplingVisible(rbm, i);
    }

    return vect;
}

Eigen::VectorXd & GeneralizedGRBMSampler::blockedGibbsSamplingHidden(GeneralizedGRBM &rbm) {
    auto vect = rbm.nodes.h;

    for (int j = 0; j < rbm.getHiddenSize(); j++) {
        vect(j) = gibbsSamplingHidden(rbm, j);
    }

    return vect;
}

double GeneralizedGRBMSampler::updateByGibbsSamplingVisible(GeneralizedGRBM &rbm, int vindex) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::normal_distribution<double> dist(rbm.meanVisible(vindex), sqrt(1 / rbm.params.lambda(vindex)));

    double value = dist(mt);
    rbm.nodes.v(vindex) = value;
    return value;
}

double GeneralizedGRBMSampler::updateByGibbsSamplingHidden(GeneralizedGRBM &rbm, int hindex) {
    std::vector<double> probs(rbm.getHiddenValueSetSize());
    auto hidset = rbm.splitHiddenSet();
    for (int i = 0; i < hidset.size(); i++) {
        probs[i] = rbm.condProbHid(hindex, hidset[i]);
    }

    std::random_device rd;
    std::mt19937 mt(rd());
    std::discrete_distribution<> dist(probs.begin(), probs.end());

    int index = dist(mt);
    double value = hidset[index];
    rbm.nodes.h(hindex) = value;
    return value;
}

Eigen::VectorXd & GeneralizedGRBMSampler::updateByBlockedGibbsSamplingVisible(GeneralizedGRBM &rbm) {

    for (int i = 0; i < rbm.getVisibleSize(); i++) {
        updateByGibbsSamplingVisible(rbm, i);
    }

    return rbm.nodes.v;
}

Eigen::VectorXd & GeneralizedGRBMSampler::updateByBlockedGibbsSamplingHidden(GeneralizedGRBM &rbm) {
    for (int j = 0; j < rbm.getHiddenSize(); j++) {
        updateByGibbsSamplingHidden(rbm, j);
    }

    return rbm.nodes.h;
}
