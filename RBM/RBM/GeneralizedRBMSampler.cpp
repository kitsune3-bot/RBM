#include "GeneralizedRBMSampler.h"
#include "GeneralizedRBM.h"
#include <random>


GeneralizedRBMSampler::GeneralizedRBMSampler()
{
}


GeneralizedRBMSampler::~GeneralizedRBMSampler()
{
}



double GeneralizedRBMSampler::gibbsSamplingVisible(GeneralizedRBM & rbm, int vindex) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    double value = rbm.condProbVis(vindex, 0.0) < dist(mt) ? 0.0 : 1.0;
    return value;
}

double GeneralizedRBMSampler::gibbsSamplingHidden(GeneralizedRBM & rbm, int hindex) {
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

Eigen::VectorXd & GeneralizedRBMSampler::blockedGibbsSamplingVisible(GeneralizedRBM & rbm) {
    auto vect = rbm.nodes.v;

    for (int i = 0; i < rbm.getVisibleSize(); i++) {
        vect(i) = gibbsSamplingVisible(rbm, i);
    }

    return vect;
}

Eigen::VectorXd & GeneralizedRBMSampler::blockedGibbsSamplingHidden(GeneralizedRBM & rbm) {
    auto vect = rbm.nodes.h;

    for (int j = 0; j < rbm.getHiddenSize(); j++) {
        vect(j) = gibbsSamplingHidden(rbm, j);
    }

    return vect;
}

double GeneralizedRBMSampler::updateByGibbsSamplingVisible(GeneralizedRBM & rbm, int vindex) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    double value = rbm.condProbVis(vindex, 0.0) < dist(mt) ? 0.0 : 1.0;
    rbm.nodes.v(vindex) = value;
    return value;
}

double GeneralizedRBMSampler::updateByGibbsSamplingHidden(GeneralizedRBM & rbm, int hindex) {
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

Eigen::VectorXd & GeneralizedRBMSampler::updateByBlockedGibbsSamplingVisible(GeneralizedRBM & rbm) {

    for (int i = 0; i < rbm.getVisibleSize(); i++) {
        updateByGibbsSamplingVisible(rbm, i);
    }

    return rbm.nodes.v;
}

Eigen::VectorXd & GeneralizedRBMSampler::updateByBlockedGibbsSamplingHidden(GeneralizedRBM & rbm) {
    for (int j = 0; j < rbm.getHiddenSize(); j++) {
        updateByGibbsSamplingHidden(rbm, j);
    }

    return rbm.nodes.h;
}
