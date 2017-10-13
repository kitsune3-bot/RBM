#pragma once
#include "../RBMNodeBase.h"
#include "Eigen/Core"
#include <vector>
#include <numeric>


class GBRBMNode : RBMNodeBase {
private:
    size_t vSize = 0;
    size_t hSize = 0;
public:
    Eigen::VectorXd v;
    Eigen::VectorXd h;

public:
    GBRBMNode() = default;
    GBRBMNode(size_t v_size, size_t h_size);
    ~GBRBMNode() = default;

    // 可視変数の総数を返す
    inline size_t getVisibleSize();

    // 隠れ変数の総数を返す
    inline size_t getHiddenSize();

    // 可視層の全てを返す
    inline Eigen::VectorXd getVisibleLayer();

    // 隠れ層の全てを返す
    inline Eigen::VectorXd getHiddenLayer();

    // 可視変数の値を返す
    inline double getVisibleUnit(int vindex);

    // 隠れ変数の値を返す
    inline double getHiddenUnit(int vindex);

    // 可視変数のノード番号集合を返す
    std::vector<int> getVnodeIndexSet();

    // 隠れ変数のノード番号集合を返す
    std::vector<int> getHnodeIndexSet();
};



