#include "GBRBMParamator.h"
#include <random>



GBRBMParamator::GBRBMParamator()
{
}


GBRBMParamator::~GBRBMParamator()
{
}

GBRBMParamator::GBRBMParamator(size_t v_size, size_t h_size) {
    vSize = v_size;
    hSize = h_size;

    initParams();
}

void GBRBMParamator::initParams() {
    b.resize(vSize);
    b.setConstant(0.0);
    lambda.resize(vSize);
    lambda.setConstant(10.0);  // 逆分散は非負制約がある, 逆分散 = 10 -> 分散0.1
    c.resize(hSize);
    c.setConstant(0.0);
    w.resize(vSize, hSize);
    w.setConstant(0.0);
}

void GBRBMParamator::initParamsRandom(double range_min, double range_max) {
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

    // XXX: 逆分散は乱数使うと危ない
    lambda.resize(vSize);
    lambda.setConstant(10.0);  // 逆分散は非負制約がある, 逆分散 = 10 -> 分散0.1
}
