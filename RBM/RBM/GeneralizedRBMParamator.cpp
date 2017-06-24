#include "GeneralizedRBMParamator.h"
#include <random>



GeneralizedRBMParamator::GeneralizedRBMParamator()
{
}


GeneralizedRBMParamator::~GeneralizedRBMParamator()
{
}

GeneralizedRBMParamator::GeneralizedRBMParamator(size_t v_size, size_t h_size) {
    vSize = v_size;
    hSize = h_size;

    initParams();
}

void GeneralizedRBMParamator::initParams() {
    b.resize(vSize);
    b.setConstant(0.0);
    c.resize(hSize);
    c.setConstant(0.0);
    w.resize(vSize, hSize);
    w.setConstant(0.0);
}

void GeneralizedRBMParamator::initParamsRandom(double range_min, double range_max) {
    b.resize(vSize);
    c.resize(hSize);
    w.resize(vSize, hSize);

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(range_min, range_max);

    for (int i = 0; i < vSize; i++) {
        b(i) = dist(mt);

        for (int j = 0; j < hSize; j++) {
            w(i, j) = dist(mt);
        }
    }

    for (int j = 0; j < hSize; j++) {
        c(j) = dist(mt);
    }
}
