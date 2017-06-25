#pragma once
#include "ConditionalRBMParamatorBase.h"
#include <Eigen/Core>

class ConditionalRBMParamator : ConditionalRBMParamatorBase {
private:
    size_t vSize;
    size_t hSize;
    size_t xSize;
public:
    Eigen::VectorXd b;  // 可視変数のバイアス
    Eigen::VectorXd c;  // 隠れ変数のバイアス
    Eigen::MatrixXd w;  // 可視変数-隠れ変数間のカップリング
    Eigen::MatrixXd hxW;  // 隠れ変数-条件変数間のカップリング


public:
    ConditionalRBMParamator();
    ConditionalRBMParamator(size_t vsize, size_t hsize, size_t xsize);
    ~ConditionalRBMParamator();

    // 可視変数の総数を返す
    inline size_t getVisibleSize();

    // 隠れ変数の総数を返す
    inline size_t getHiddenSize();

    // 条件変数の総数を返す
    inline size_t getCondSize();

    // 可視変数のバイアスを返す
    inline double getVisibleBias(int vindex);

    // 可視変数のバイアスベクトルを返す
    inline Eigen::VectorXd getVisibleBiasVector();

    // 隠れ変数のバイアスを返す
    inline double getHiddenBias(int hindex);

    // 隠れ変数のバイアスベクトルを返す
    inline Eigen::VectorXd getHiddenBiasVector();

    // ウェイトパラメータを返す
    inline double getWeight(int vindex, int hindex);

    // ウェイト行列を返す
    inline Eigen::MatrixXd getWeightMatrix();

    // 隠れ変数-条件変数間ウェイトパラメータを返す
    inline double getHXWeight(int hindex, int xindex);

    // 隠れ変数-条件変数間ウェイト行列を返す
    inline Eigen::MatrixXd getHXWeightMatrix();

    // 全てのパラメータを0で初期化
    void initParams();

    // 全てのパラメータを[min, max]の一様乱数で初期化
    void initParamsRandom(double range_min, double range_max);
};

// 可視変数の総数を返す
inline size_t ConditionalRBMParamator::getVisibleSize() {
    return vSize;
}

// 隠れ変数の総数を返す
inline size_t ConditionalRBMParamator::getHiddenSize() {
    return hSize;
}

// 隠れ変数の総数を返す
inline size_t ConditionalRBMParamator::getCondSize() {
    return xSize;
}

// 可視変数のバイアスを返す
inline double ConditionalRBMParamator::getVisibleBias(int vindex) {
    return b(vindex);
}

// 可視変数のバイアスベクトルを返す
inline Eigen::VectorXd ConditionalRBMParamator::getVisibleBiasVector() {
    return b;
}

// 隠れ変数のバイアスを返す
inline double ConditionalRBMParamator::getHiddenBias(int hindex) {
    return c(hindex);
}

// 隠れ変数のバイアスベクトルを返す
inline Eigen::VectorXd ConditionalRBMParamator::getHiddenBiasVector() {
    return c;
}

// ウェイトパラメータを返す
inline double ConditionalRBMParamator::getWeight(int vindex, int hindex) {
    return w(vindex, hindex);
}

// ウェイト行列を返す
inline Eigen::MatrixXd ConditionalRBMParamator::getWeightMatrix() {
    return w;
}

// 隠れ変数-条件変数間ウェイトパラメータを返す
inline double ConditionalRBMParamator::getHXWeight(int hindex, int xindex) {
    return hxW(hindex, xindex);
}

// 隠れ変数-条件変数間ウェイト行列を返す
inline Eigen::MatrixXd ConditionalRBMParamator::getHXWeightMatrix() {
    return hxW;
}