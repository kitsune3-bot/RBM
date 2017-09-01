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
    lambda.setConstant(1.0);  // 逆分散は非負制約がある, 逆分散 = 10 -> 分散0.1
    c.resize(hSize);
    c.setConstant(0.0);
    hvW.resize(hSize, vSize);
    hvW.setConstant(0.0);
    //vxW.resize(vSize, xSize);
    //vxW.setConstant(0.0);
    xhW.resize(xSize, hSize);
    xhW.setConstant(0.0);
}

void ConditionalGRBMParamator::initParamsRandom(double range_min, double range_max) {
    b.resize(vSize);
    c.resize(hSize);
    hvW.resize(hSize, vSize);
    //vxW.resize(vSize, xSize);
    xhW.resize(xSize, hSize);

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(range_min, range_max);

    for (int i = 0; i < vSize; i++) {
        b(i) = dist(mt);

        for (int j = 0; j < hSize; j++) {
            hvW(j, i) = dist(mt) * 100;
        }

        //for (int k = 0; k < xSize; k++) {
        //    vxW(i, k) = dist(mt);
        //}
    }

    for (int j = 0; j < hSize; j++) {
        c(j) = dist(mt);
       
        for (int k = 0; k < xSize; k++) {
            xhW(k, j) = dist(mt) * 100;
        }
    }

    // XXX: 逆分散は乱数使うと危ない
    lambda.resize(vSize);
    lambda.setConstant(100.0);  // 逆分散は非負制約がある, 逆分散 = 10 -> 分散0.1
}
