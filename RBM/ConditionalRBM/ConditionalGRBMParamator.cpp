#include "ConditionalGRBMParamator.h"
#include <random>



ConditionalGRBMParamator::ConditionalGRBMParamator()
{
}


ConditionalGRBMParamator::~ConditionalGRBMParamator()
{
}

ConditionalGRBMParamator::ConditionalGRBMParamator(size_t v_size, size_t h_size, size_t x_size) {
    vSize = v_size;
    hSize = h_size;
    xSize = x_size;

    initParams();
}

void ConditionalGRBMParamator::initParams() {
    b.resize(vSize);
    b.setConstant(0.0);
    lambda.resize(vSize);
    lambda.setConstant(10.0);  // 逆分散は非負制約がある, 逆分散 = 10 -> 分散0.1
    c.resize(hSize);
    c.setConstant(0.0);
    w.resize(vSize, hSize);
    w.setConstant(0.0);
    hxW.resize(hSize, xSize);
    hxW.setConstant(0.0);
}

void ConditionalGRBMParamator::initParamsRandom(double range_min, double range_max) {
    b.resize(vSize);
    c.resize(hSize);
    w.resize(vSize, hSize);
    hxW.resize(hSize, xSize);

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
       
        for (int k = 0; k < xSize; k++) {
            hxW(j, k) = dist(mt);
        }
    }

    // XXX: 逆分散は乱数使うと危ない
    lambda.resize(vSize);
    lambda.setConstant(10.0);  // 逆分散は非負制約がある, 逆分散 = 10 -> 分散0.1
}
