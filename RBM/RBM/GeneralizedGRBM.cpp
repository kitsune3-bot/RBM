#include "GeneralizedGRBM.h"
#include "RBMMath.h"
#include <cmath>


GeneralizedGRBM::GeneralizedGRBM()
{
}


GeneralizedGRBM::~GeneralizedGRBM()
{
}

GeneralizedGRBM::GeneralizedGRBM(size_t v_size, size_t h_size) {
    vSize = v_size;
    hSize = h_size;

    // ノード確保
    nodes = GeneralizedGRBMNode(v_size, h_size);

    // パラメータ初期化
    params = GeneralizedGRBMParamator(v_size, h_size);
    params.initParamsRandom(-0.01, 0.01);

    // 区間分割
    hiddenValueSet = splitHiddenSet();
}


// 可視変数の数を返す
size_t GeneralizedGRBM::getVisibleSize() {
    return vSize;
}

// 隠れ変数の数を返す
size_t GeneralizedGRBM::getHiddenSize() {
    return hSize;
}


// 規格化を返します
double GeneralizedGRBM::getNormalConstant() {
    // 未実装なので保留
    throw;
}


// エネルギー関数を返します
double GeneralizedGRBM::getEnergy() {
    // まだ必要ないため実装保留
    throw;
}


// 自由エネルギーを返します
double GeneralizedGRBM::getFreeEnergy() {
    return -log(this->getNormalConstant());
}

// 隠れ変数の活性化関数的なもの
double GeneralizedGRBM::actHidJ(int hindex) {
    auto value_set = splitHiddenSet();
    double numer = 0.0;  // 分子
    double denom = sumExpMu(hindex);  // 分母

    for (auto & value : value_set) {
        numer += value * exp(mu(hindex) * value);
    }

    return numer / denom;
}

// 可視変数に関する外部磁場と相互作用
// ただし二乗の項を除く
double GeneralizedGRBM::lambda(int vindex) {
    double lam = params.b(vindex);

    // TODO: Eigen使ってるから内積計算で高速化できる
    for (int j = 0; j < hSize; j++) {
        lam += params.w(vindex, j) * nodes.h(j);
    }

    return lam;
}

// lambdaの可視変数に関する全ての実現値の総和
double GeneralizedGRBM::sumExpLambda(int vindex) {
    // XXX: 使いません
    throw;
    return 1.0 + exp(lambda(vindex));
}

// v_iの平均
double GeneralizedGRBM::meanVisible(int vindex) {
    double inv_var = params.lambda(vindex);;  // 逆分散
    double mean = lambda(vindex) / inv_var;
    return mean;
}


// v_iの条件付き確率の規格化定数
double GeneralizedGRBM::integralExpVisible(int vindex) {
    return sqrt(2 * 3.141592 / params.lambda(vindex));
}


// 隠れ変数に関する外部磁場と相互作用
double GeneralizedGRBM::mu(int hindex) {
    double mu = params.c(hindex);

    // TODO: Eigen使ってるから内積計算で高速化できる
    for (int i = 0; i < vSize; i++) {
        mu += params.w(i, hindex) * nodes.v(i);
    }

    return mu;
}

// muの可視変数に関する全ての実現値の総和
double GeneralizedGRBM::sumExpMu(int hindex) {
    double sum = 0.0;

    for (auto & value : hiddenValueSet) {
        sum += exp(mu(hindex) * value);
    }

    return sum;
}

// 隠れ変数を条件で与えた可視変数の条件付き確率, P(v_i | h)
double GeneralizedGRBM::condProbVis(int vindex, double value) {
    double lam = lambda(vindex);
    double inv_var = params.lambda[vindex];  // 逆分散
    double mean = meanVisible(vindex);  // 可視変数期待値
    return exp(inv_var / 2.0 * pow(value - mean, 2)) / integralExpVisible(vindex);
}

// 可視変数を条件で与えた隠れ変数の条件付き確率, P(h_j | v)
double GeneralizedGRBM::condProbHid(int hindex, double value) {
    double m = mu(hindex);
    double prob = exp(m * value) / sumExpMu(hindex);
    return prob;
}

std::vector<double> GeneralizedGRBM::splitHiddenSet() {
    std::vector<double> set(divSize + 1);

    auto x = [](double split_size, double i, double min, double max) {  // 分割関数[i=0,1,...,elems]
        return 1.0 / (split_size)* i * (max - min) + min;
    };

    for (int i = 0; i < set.size(); i++) set[i] = x(divSize, i, hMin, hMax);

    return set;
}

int GeneralizedGRBM::getHiddenValueSetSize() {
    return divSize + 1;
}

// 隠れ変数の取りうる最大値を取得
double GeneralizedGRBM::getHiddenMax() {
    return hMax;
}

// 隠れ変数の取りうる最大値を設定
void GeneralizedGRBM::setHiddenMax(double value) {
    hMax = value;

    // 区間分割
    hiddenValueSet = splitHiddenSet();
}

// 隠れ変数の取りうる最小値を取得
double GeneralizedGRBM::getHiddenMin() {
    return hMin;
}

// 隠れ変数の取りうる最小値を設定
void GeneralizedGRBM::setHiddenMin(double value) {
    hMin = value;

    // 区間分割
    hiddenValueSet = splitHiddenSet();
}

// 隠れ変数の区間分割数を返す
size_t GeneralizedGRBM::getHiddenDivSize() {
    return divSize;
}

// 隠れ変数の区間分割数を設定
void GeneralizedGRBM::setHiddenDiveSize(size_t div_size) {
    divSize = div_size;

    // 区間分割
    hiddenValueSet = splitHiddenSet();
}
