#include "ConditionalRBM.h"
#include "ConditionalRBMMath.h"
#include <cmath>


ConditionalRBM::ConditionalRBM()
{
}


ConditionalRBM::~ConditionalRBM()
{
}

ConditionalRBM::ConditionalRBM(size_t v_size, size_t h_size, size_t x_size) {
    vSize = v_size;
    hSize = h_size;
    xSize = x_size;

    // ノード確保
    nodes = ConditionalRBMNode(v_size, h_size, x_size);

    // パラメータ初期化
    params = ConditionalRBMParamator(v_size, h_size, x_size);
    params.initParamsRandom(0.01, 0.01);
}


// 可視変数の数を返す
size_t ConditionalRBM::getVisibleSize() {
    return vSize;
}

// 隠れ変数の数を返す
size_t ConditionalRBM::getHiddenSize() {
    return hSize;
}

// 条件変数の数を返す
size_t ConditionalRBM::getCondSize() {
    return xSize;
}

// 規格化を返します
double ConditionalRBM::getNormalConstant() {
    // 未実装なので保留
    throw;
}


// エネルギー関数を返します
double ConditionalRBM::getEnergy() {
    // まだ必要ないため実装保留
    throw;
}


// 自由エネルギーを返します
double ConditionalRBM::getFreeEnergy() {
    return -log(this->getNormalConstant());
}

// 隠れ変数の活性化関数的なもの
double ConditionalRBM::actHidJ(int hindex) {
    return ConditionalRBMMath::sigmoid(mu(hindex));
}

// 可視変数に関する外部磁場と相互作用
double ConditionalRBM::lambda(int vindex) {
    // TODO: 条件変数からの影響を受けるようにしたい

    double lam = params.b(vindex);

    // TODO: Eigen使ってるから内積計算で高速化できる
    for (int j = 0; j < hSize; j++) {
        lam += params.w(vindex, j) * nodes.h(j);
    }

    return lam;
}

// lambdaの可視変数に関する全ての実現値の総和
double ConditionalRBM::sumExpLambda(int vindex) {
    // {0, 1}での実装
    return 1.0 + exp(lambda(vindex));
}

// 隠れ変数に関する外部磁場と相互作用
double ConditionalRBM::mu(int hindex) {
    double mu = params.c(hindex);

    // TODO: Eigen使ってるから内積計算で高速化できる
    for (int i = 0; i < vSize; i++) {
        mu += params.w(i, hindex) * nodes.v(i);
    }

    // 条件変数からの影響
    for (int k = 0; k < xSize; k++) {
        mu += params.hxW(hindex, k) * nodes.x(k);
    }

    return mu;
}

// muの可視変数に関する全ての実現値の総和
double ConditionalRBM::sumExpMu(int hindex) {
    // {0, 1}での実装
    return 1.0 + exp(mu(hindex));
}

// 隠れ変数を条件で与えた可視変数の条件付き確率, P(v_i | h)
double ConditionalRBM::condProbVis(int vindex, double value) {
    double lam = lambda(vindex);
    return exp(lam * value) / sumExpLambda(vindex);
}

// 可視変数を条件で与えた隠れ変数の条件付き確率, P(h_j | v)
double ConditionalRBM::condProbHid(int hindex, double value) {
    double m = mu(hindex);
    return exp(m * value) / sumExpMu(hindex);
}

