#pragma once
#include <Eigen/Core>

class RBMParamatorBase{
    // 可視変数の総数を返す
    virtual size_t getVisibleSize() = 0;

    // 隠れ変数の総数を返す
    virtual size_t getHiddenSize() = 0;

    // 可視変数のバイアスを返す
    virtual double getVisibleBias(int vindex) = 0;

    // 可視変数のバイアスベクトルを返す
    virtual Eigen::VectorXd getVisibleBiasVector() = 0;

    // 隠れ変数のバイアスを返す
    virtual double getHiddenBias(int hindex) = 0;

    // 隠れ変数のバイアスベクトルを返す
    virtual Eigen::VectorXd getHiddenBiasVector() = 0;

    // ウェイトパラメータを返す
    virtual double getWeight(int vindex, int hindex) = 0;

    // ウェイト行列を返す
    virtual Eigen::MatrixXd getWeightMatrix() = 0;

    // 全てのパラメータを0で初期化
    virtual void initParams() = 0;

    // 全てのパラメータを[min, max]の一様乱数で初期化
    virtual void initParamsRandom(double range_min, double range_max) = 0;
};