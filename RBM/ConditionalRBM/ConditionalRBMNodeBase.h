#pragma once
#include "Eigen/Core"
#include <vector>

class ConditionalRBMNodeBase {

public:
    ConditionalRBMNodeBase() = default;
    ~ConditionalRBMNodeBase() = default;

    // 可視変数の総数を返す
    virtual size_t getVisibleSize() = 0;

    // 隠れ変数の総数を返す
    virtual size_t getHiddenSize() = 0;

    // 条件変数の総数を返す
    virtual size_t getCondSize() = 0;

    // 可視層の全てを返す
    virtual Eigen::VectorXd getVisibleLayer() = 0;

    // 隠れ層の全てを返す
    virtual Eigen::VectorXd getHiddenLayer() = 0;

    // 条件層の全てを返す
    virtual Eigen::VectorXd getCondLayer() = 0;

    // 可視変数の値を返す
    virtual double getVisibleUnit(int vindex) = 0;

    // 隠れ変数の値を返す
    virtual double getHiddenUnit(int hindex) = 0;

    // 条件変数の値を返す
    virtual double getCondUnit(int xindex) = 0;

    // 可視変数のノード番号集合を返す
    virtual std::vector<int> getVnodeIndexSet() = 0;

    // 隠れ変数のノード番号集合を返す
    virtual std::vector<int> getHnodeIndexSet() = 0;

    // 条件変数のノード番号集合を返す
    virtual std::vector<int> getXnodeIndexSet() = 0;
};
