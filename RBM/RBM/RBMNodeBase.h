#pragma once
#include "Eigen/Core"
#include <vector>

class RBMNodeBase {

public:
    RBMNodeBase() = default;
    ~RBMNodeBase() = default;

    // 可視変数の総数を返す
    virtual size_t getVisibleSize() = 0;

    // 隠れ変数の総数を返す
    virtual size_t getHiddenSize() = 0;

    // 可視層の全てを返す
    virtual Eigen::VectorXd getVisibleLayer() = 0;

    // 隠れ層の全てを返す
    virtual Eigen::VectorXd getHiddenLayer() = 0;

    // 可視変数の値を返す
    virtual double getVisibleUnit(int vindex) = 0;

    // 隠れ変数の値を返す
    virtual double getHiddenUnit(int vindex) = 0;

    // 可視変数のノード番号集合を返す
    virtual std::vector<int> getVnodeIndexSet() = 0;

    // 隠れ変数のノード番号集合を返す
    virtual std::vector<int> getHnodeIndexSet() = 0;
};
