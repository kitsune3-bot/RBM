#include "ConditionalGRBM.h"
#include "ConditionalRBMMath.h"
#include <cmath>


ConditionalGRBM::ConditionalGRBM()
{
}


ConditionalGRBM::~ConditionalGRBM()
{
}

ConditionalGRBM::ConditionalGRBM(size_t v_size, size_t h_size, size_t x_size) {
    vSize = v_size;
    hSize = h_size;
    xSize = x_size;

    // ノード確保
    nodes = ConditionalGRBMNode(v_size, h_size, x_size);

    // パラメータ初期化
    params = ConditionalGRBMParamator(v_size, h_size, x_size);
    params.initParamsRandom(-0.001, 0.001);
}


// 可視変数の数を返す
size_t ConditionalGRBM::getVisibleSize() {
    return vSize;
}

// 隠れ変数の数を返す
size_t ConditionalGRBM::getHiddenSize() {
    return hSize;
}

// 条件変数の数を返す
size_t ConditionalGRBM::getCondSize() {
    return xSize;
}

// 規格化を返します
double ConditionalGRBM::getNormalConstant() {
    // 未実装なので保留
    throw;
}


// エネルギー関数を返します
double ConditionalGRBM::getEnergy() {
    // まだ必要ないため実装保留
    throw;
}


// 自由エネルギーを返します
double ConditionalGRBM::getFreeEnergy() {
    return -log(this->getNormalConstant());
}

// 隠れ変数の活性化関数的なもの
double ConditionalGRBM::actHidJ(int hindex) {
    return ConditionalRBMMath::sigmoid(mu(hindex));
}

// 可視変数に関する外部磁場と相互作用
// ただし二乗の項を除く
double ConditionalGRBM::lambda(int vindex) {
    double lam = params.b(vindex);

    // TODO: 条件変数からの影響受けるようにしたい
    // FIXME: 因子化しないと計算量やばい
//    for (int k = 0; k < xSize; k++) {
//        lam += params.vxW(vindex, k) * nodes.x(k);
//    }


    // TODO: Eigen使ってるから内積計算で高速化できる
    for (int j = 0; j < hSize; j++) {
        lam += params.hvW(j, vindex) * nodes.h(j);
    }

    return lam;
}

// lambdaの可視変数に関する全ての実現値の総和
double ConditionalGRBM::sumExpLambda(int vindex) {
    // XXX: 使いません
    throw;
    return 1.0 + exp(lambda(vindex));
}

// v_iの平均
double ConditionalGRBM::meanVisible(int vindex) {
    double inv_var = params.lambda(vindex);;  // 逆分散
    double mean = lambda(vindex) / inv_var;
    return mean;
}


// v_iの条件付き確率の規格化定数
double ConditionalGRBM::integralExpVisible(int vindex) {
    return sqrt(2 * 3.141592 / params.lambda(vindex));
}


// 隠れ変数に関する外部磁場と相互作用
double ConditionalGRBM::mu(int hindex) {
    double mu = params.c(hindex);

    // 条件変数からの影響
    for (int k = 0; k < xSize; k++) {
        mu += params.xhW(k, hindex) * nodes.x(k);
    }

    // TODO: Eigen使ってるから内積計算で高速化できる
    for (int i = 0; i < vSize; i++) {
        mu += params.hvW(hindex, i) * nodes.v(i);
    }

    return mu;
}

// muの可視変数に関する全ての実現値の総和
double ConditionalGRBM::sumExpMu(int hindex) {
    // {0, 1}での実装
    return 1.0 + exp(mu(hindex));
}

// 隠れ変数を条件で与えた可視変数の条件付き確率, P(v_i | h)
double ConditionalGRBM::condProbVis(int vindex, double value) {
    double inv_var = params.lambda[vindex];  // 逆分散
    double mean = meanVisible(vindex);  // 可視変数期待値
    return exp(inv_var / 2.0 * pow(value - mean, 2)) / integralExpVisible(vindex);
}

// 可視変数を条件で与えた隠れ変数の条件付き確率, P(h_j | v)
double ConditionalGRBM::condProbHid(int hindex, double value) {
    double m = mu(hindex);
    return exp(m * value) / sumExpMu(hindex);
}

