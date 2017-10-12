#pragma once
#include "../RBMNodeBase.h"
#include "Eigen/Core"
#include <vector>
#include <numeric>

class GeneralizedGRBMNode : RBMNodeBase {
private:
    size_t vSize = 0;
    size_t hSize = 0;
public:
    Eigen::VectorXd v;
    Eigen::VectorXd h;

public:
    GeneralizedGRBMNode() = default;
    GeneralizedGRBMNode(size_t v_size, size_t h_size);
    ~GeneralizedGRBMNode() = default;

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

// 可視変数の総数を返す
inline size_t GeneralizedGRBMNode::getVisibleSize() {
    return vSize;
}

// 隠れ変数の総数を返す
inline size_t GeneralizedGRBMNode::getHiddenSize() {
    return hSize;
}

// 可視層の全てを返す
inline Eigen::VectorXd GeneralizedGRBMNode::getVisibleLayer() {
    return v;
}

// 隠れ層の全てを返す
inline Eigen::VectorXd GeneralizedGRBMNode::getHiddenLayer() {
    return h;
}

// 可視変数の値を返す
inline double GeneralizedGRBMNode::getVisibleUnit(int vindex) {
    return v(vindex);
}

// 隠れ変数の値を返す
inline double GeneralizedGRBMNode::getHiddenUnit(int hindex) {
    return h(hindex);
}




inline GeneralizedGRBMNode::GeneralizedGRBMNode(size_t v_size, size_t h_size) {
	vSize = v_size;
	hSize = h_size;
	v.setConstant(v_size, 0.0);
	h.setConstant(h_size, 0.0);
}

inline std::vector<int> GeneralizedGRBMNode::getVnodeIndexSet() {
	std::vector<int> set(vSize);
	std::iota(set.begin(), set.end(), 0);

	return set;
}

inline std::vector<int> GeneralizedGRBMNode::getHnodeIndexSet() {
	std::vector<int> set(hSize);
	std::iota(set.begin(), set.end(), 0);

	return set;
}
