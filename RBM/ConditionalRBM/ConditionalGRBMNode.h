#pragma once
#include "ConditionalRBMNodeBase.h"
#include "Eigen/Core"
#include <vector>

class ConditionalGRBMNode : ConditionalRBMNodeBase {
private:
    size_t vSize = 0;
    size_t hSize = 0;
    size_t xSize = 0;
public:
    Eigen::VectorXd v;
    Eigen::VectorXd h;
    Eigen::VectorXd x;

public:
    ConditionalGRBMNode();
    ConditionalGRBMNode(size_t v_size, size_t h_size, size_t x_size);
    ~ConditionalGRBMNode();

    // 可視変数の総数を返す
    inline size_t getVisibleSize();

    // 隠れ変数の総数を返す
    inline size_t getHiddenSize();

    // 条件変数の総数を返す
    inline size_t getCondSize();

    // 可視層の全てを返す
    inline Eigen::VectorXd getVisibleLayer();

    // 隠れ層の全てを返す
    inline Eigen::VectorXd getHiddenLayer();

    // 条件層の全てを返す
    inline Eigen::VectorXd getCondLayer();

    // 可視変数の値を返す
    inline double getVisibleUnit(int vindex);

    // 隠れ変数の値を返す
    inline double getHiddenUnit(int vindex);

    // 条件変数の値を返す
    inline double getCondUnit(int vindex);

    // 可視変数のノード番号集合を返す
    std::vector<int> getVnodeIndexSet();

    // 隠れ変数のノード番号集合を返す
    std::vector<int> getHnodeIndexSet();

    // 条件変数のノード番号集合を返す
    std::vector<int> getXnodeIndexSet();
};

// 可視変数の総数を返す
inline size_t ConditionalGRBMNode::getVisibleSize() {
    return vSize;
}

// 隠れ変数の総数を返す
inline size_t ConditionalGRBMNode::getHiddenSize() {
    return hSize;
}

// 条件変数の総数を返す
inline size_t ConditionalGRBMNode::getCondSize() {
    return xSize;
}

// 可視層の全てを返す
inline Eigen::VectorXd ConditionalGRBMNode::getVisibleLayer() {
    return v;
}

// 隠れ層の全てを返す
inline Eigen::VectorXd ConditionalGRBMNode::getHiddenLayer() {
    return h;
}

// 条件層の全てを返す
inline Eigen::VectorXd ConditionalGRBMNode::getCondLayer() {
    return x;
}

// 可視変数の値を返す
inline double ConditionalGRBMNode::getVisibleUnit(int vindex) {
    return v(vindex);
}

// 隠れ変数の値を返す
inline double ConditionalGRBMNode::getHiddenUnit(int hindex) {
    return h(hindex);
}

// 条件変数の値を返す
inline double ConditionalGRBMNode::getCondUnit(int xindex) {
    return x(xindex);
}
