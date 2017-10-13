#include "GeneralizedRBMNode.h"


// 可視変数の総数を返す
size_t GeneralizedRBMNode::getVisibleSize() {
	return vSize;
}

// 隠れ変数の総数を返す
size_t GeneralizedRBMNode::getHiddenSize() {
	return hSize;
}

// 可視層の全てを返す
Eigen::VectorXd GeneralizedRBMNode::getVisibleLayer() {
	return v;
}

// 隠れ層の全てを返す
Eigen::VectorXd GeneralizedRBMNode::getHiddenLayer() {
	return h;
}

// 可視変数の値を返す
double GeneralizedRBMNode::getVisibleUnit(int vindex) {
	return v(vindex);
}

// 隠れ変数の値を返す
double GeneralizedRBMNode::getHiddenUnit(int hindex) {
	return h(hindex);
}

GeneralizedRBMNode::GeneralizedRBMNode(size_t v_size, size_t h_size) {
	vSize = v_size;
	hSize = h_size;
	v.setConstant(v_size, 0.0);
	h.setConstant(h_size, 0.0);
}

std::vector<int> GeneralizedRBMNode::getVnodeIndexSet() {
	std::vector<int> set(vSize);
	std::iota(set.begin(), set.end(), 0);

	return set;
}

std::vector<int> GeneralizedRBMNode::getHnodeIndexSet() {
	std::vector<int> set(hSize);
	std::iota(set.begin(), set.end(), 0);

	return set;
}
