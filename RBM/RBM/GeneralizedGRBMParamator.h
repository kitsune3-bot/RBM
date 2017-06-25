﻿#pragma once
#include "RBMParamatorBase.h"
#include <Eigen/Core>

class GeneralizedGRBMParamator : RBMParamatorBase {
private:
    size_t vSize;
    size_t hSize;
public:
    Eigen::VectorXd b;  // 可視変数のバイアス
    Eigen::VectorXd c;  // 隠れ変数のバイアス
    Eigen::MatrixXd w;  // 可視変数-隠れ変数間のカップリング
    Eigen::VectorXd lambda;  // 可視変数の逆分散


public:
    GeneralizedGRBMParamator();
    GeneralizedGRBMParamator(size_t vsize, size_t hsize);
    ~GeneralizedGRBMParamator();

    // 可視変数の総数を返す
    inline size_t getVisibleSize();

    // 隠れ変数の総数を返す
    inline size_t getHiddenSize();

    // 可視変数のバイアスを返す
    inline double getVisibleBias(int vindex);

    // 可視変数のバイアスベクトルを返す
    inline Eigen::VectorXd getVisibleBiasVector();

    // 可視変数の逆分散を返す
    inline double getVisibleLambda(int vindex);

    // 可視変数の逆分散ベクトルを返す
    inline Eigen::VectorXd getVisibleLambdaVector();

    // 隠れ変数のバイアスを返す
    inline double getHiddenBias(int hindex);

    // 隠れ変数のバイアスベクトルを返す
    inline Eigen::VectorXd getHiddenBiasVector();

    // ウェイトパラメータを返す
    inline double getWeight(int vindex, int hindex);

    // ウェイト行列を返す
    inline Eigen::MatrixXd getWeightMatrix();

    // 全てのパラメータを0で初期化
    void initParams();

    // 全てのパラメータを[min, max]の一様乱数で初期化
    void initParamsRandom(double range_min, double range_max);
};

// 可視変数の総数を返す
inline size_t GeneralizedGRBMParamator::getVisibleSize() {
    return vSize;
}

// 隠れ変数の総数を返す
inline size_t GeneralizedGRBMParamator::getHiddenSize() {
    return hSize;
}

// 可視変数のバイアスを返す
inline double GeneralizedGRBMParamator::getVisibleBias(int vindex) {
    return b(vindex);
}

// 可視変数のバイアスベクトルを返す
inline Eigen::VectorXd GeneralizedGRBMParamator::getVisibleBiasVector() {
    return b;
}

// 可視変数の逆分散を返す
inline double GeneralizedGRBMParamator::getVisibleLambda(int vindex) {
    return lambda(vindex);
}

// 可視変数の逆分散ベクトルを返す
inline Eigen::VectorXd GeneralizedGRBMParamator::getVisibleLambdaVector() {
    return lambda;
}

// 隠れ変数のバイアスを返す
inline double GeneralizedGRBMParamator::getHiddenBias(int hindex) {
    return c(hindex);
}

// 隠れ変数のバイアスベクトルを返す
inline Eigen::VectorXd GeneralizedGRBMParamator::getHiddenBiasVector() {
    return c;
}

// ウェイトパラメータを返す
inline double GeneralizedGRBMParamator::getWeight(int vindex, int hindex) {
    return w(vindex, hindex);
}

// ウェイト行列を返す
inline Eigen::MatrixXd GeneralizedGRBMParamator::getWeightMatrix() {
    return w;
}