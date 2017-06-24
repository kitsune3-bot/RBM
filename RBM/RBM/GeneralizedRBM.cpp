﻿#include "GeneralizedRBM.h"
#include "RBMMath.h"
#include <cmath>


GeneralizedRBM::GeneralizedRBM()
{
}


GeneralizedRBM::~GeneralizedRBM()
{
}

// 可視変数の数を返す
size_t GeneralizedRBM::getVisibleSize() {
    return vSize;
}

// 隠れ変数の数を返す
size_t GeneralizedRBM::getHiddenSize() {
    return hSize;
}


// 規格化を返します
double GeneralizedRBM::getNormalConstant() {
    // 未実装なので保留
    throw;
}


// エネルギー関数を返します
double GeneralizedRBM::getEnergy() {
    // まだ必要ないため実装保留
    throw;
}


// 自由エネルギーを返します
double GeneralizedRBM::getFreeEnergy() {
    return -log(this->getNormalConstant());
}

// 隠れ変数の活性化関数的なもの
double GeneralizedRBM::actHidJ(int hindex) {
    auto value_set = splitHiddenSet();
    double numer = 0.0;  // 分子
    double denom = sumExpMu(hindex);  // 分母

    for (auto & value : value_set) {
        numer += value * exp(mu(hindex) * value);
    }

    return numer / denom;
}

// 可視変数に関する外部磁場と相互作用
double GeneralizedRBM::lambda(int vindex) {
    double lam = params.b(vindex);

    // TODO: Eigen使ってるから内積計算で高速化できる
    for (int j = 0; j < hSize; j++) {
        lam += params.w(vindex, j) * params.c(j);
    }

    return lam;
}

// lambdaの可視変数に関する全ての実現値の総和
double GeneralizedRBM::sumExpLambda(int vindex) {
    // {0, 1}での実装
    return 1.0 + exp(lambda(vindex));
}

// 隠れ変数に関する外部磁場と相互作用
double GeneralizedRBM::mu(int hindex) {
    double mu = params.c(hindex);

    // TODO: Eigen使ってるから内積計算で高速化できる
    for (int i = 0; i < vSize; i++) {
        mu += params.w(i, hindex) * params.b(i);
    }

    return mu;
}

// muの可視変数に関する全ての実現値の総和
double GeneralizedRBM::sumExpMu(int hindex) {
    double sum = 0.0;

    for (auto & value : hiddenValueSet) {
        sum += exp(mu(hindex) * value);
    }

    return sum;
}

// 隠れ変数を条件で与えた可視変数の条件付き確率, P(v_i | h)
double GeneralizedRBM::condProbVis(int vindex, double value) {
    double lam = lambda(vindex);
    return exp(lam * value) / sumExpLambda(lam);
}

// 可視変数を条件で与えた隠れ変数の条件付き確率, P(h_j | v)
double GeneralizedRBM::condProbHid(int hindex, double value) {
    double m = mu(hindex);
    return exp(m * value) / sumExpMu(m);
}

std::vector<double> GeneralizedRBM::splitHiddenSet() {
    std::vector<double> set(divSize + 1);

    auto x = [](double split_size, double i, double max, double min) {  // 分割関数[i=0,1,...,elems]
        return ((max - min)* i + (split_size + 1) * min) / (split_size + 1);
    };

    for (int i = 0; i < set.size(); i++) set[i] = x(divSize, i, -1.0, 1.0);

    return set;
}

int GeneralizedRBM::getHiddenValueSetSize() {
    return divSize + 1;
}
