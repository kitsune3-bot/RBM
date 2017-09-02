#include "ConditionalRBMParamator.h"
#include <random>



ConditionalRBMParamator::ConditionalRBMParamator()
{
}


ConditionalRBMParamator::~ConditionalRBMParamator()
{
}

ConditionalRBMParamator::ConditionalRBMParamator(size_t v_size, size_t h_size, size_t x_size) {
    vSize = v_size;
    hSize = h_size;
    xSize = x_size;

    initParams();
}

void ConditionalRBMParamator::initParams() {
    b.resize(vSize);
    b.setConstant(0.0);
    c.resize(hSize);
    c.setConstant(0.0);
    hvW.resize(vSize, hSize);
    hvW.setConstant(0.0);
    xhW.resize(xSize, hSize);
    xhW.setConstant(0.0);
}

void ConditionalRBMParamator::initParamsRandom(double range_min, double range_max) {
    b.resize(vSize);
    c.resize(hSize);
    hvW.resize(hSize, vSize);
    xhW.resize(xSize, hSize);

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(range_min, range_max);

    for (int i = 0; i < vSize; i++) {
        b(i) = dist(mt);

        for (int j = 0; j < hSize; j++) {
            hvW(j, i) = dist(mt);
        }
    }

    for (int j = 0; j < hSize; j++) {
        c(j) = dist(mt);
        for (int k = 0; k < xSize; k++) {
            xhW(k, j) = dist(mt);
        }
    }
}
