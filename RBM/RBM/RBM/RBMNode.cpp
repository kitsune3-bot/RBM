#include "RBMNode.h"


// 可視変数の総数を返す
size_t RBMNode::getVisibleSize() {
	return vSize;
}

// 隠れ変数の総数を返す
size_t RBMNode::getHiddenSize() {
	return hSize;
}

// 可視層の全てを返す
Eigen::VectorXd RBMNode::getVisibleLayer() {
	return v;
}

// 隠れ層の全てを返す
Eigen::VectorXd RBMNode::getHiddenLayer() {
	return h;
}

// 可視変数の値を返す
double RBMNode::getVisibleUnit(int vindex) {
	return v(vindex);
}

// 隠れ変数の値を返す
double RBMNode::getHiddenUnit(int hindex) {
	return h(hindex);
}

RBMNode::RBMNode(size_t v_size, size_t h_size) {
	vSize = v_size;
	hSize = h_size;
	v.setConstant(v_size, 0.0);
	h.setConstant(h_size, 0.0);
}

std::vector<int> RBMNode::getVnodeIndexSet() {
	std::vector<int> set(vSize);
	std::iota(set.begin(), set.end(), 0);

	return set;
}

std::vector<int> RBMNode::getHnodeIndexSet() {
	std::vector<int> set(hSize);
	std::iota(set.begin(), set.end(), 0);

	return set;
}
