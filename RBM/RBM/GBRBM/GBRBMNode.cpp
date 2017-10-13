#include "GBRBMNode.h"

GBRBMNode::GBRBMNode(size_t v_size, size_t h_size) {
	vSize = v_size;
	hSize = h_size;
	v.setConstant(v_size, 0.0);
	h.setConstant(h_size, 0.0);
}

std::vector<int> GBRBMNode::getVnodeIndexSet() {
	std::vector<int> set(vSize);
	std::iota(set.begin(), set.end(), 0);

	return set;
}

std::vector<int> GBRBMNode::getHnodeIndexSet() {
	std::vector<int> set(hSize);
	std::iota(set.begin(), set.end(), 0);

	return set;
}

// 可視変数の総数を返す
size_t GBRBMNode::getVisibleSize() {
	return vSize;
}

// 隠れ変数の総数を返す
size_t GBRBMNode::getHiddenSize() {
	return hSize;
}

// 可視層の全てを返す
Eigen::VectorXd GBRBMNode::getVisibleLayer() {
	return v;
}

// 隠れ層の全てを返す
Eigen::VectorXd GBRBMNode::getHiddenLayer() {
	return h;
}

// 可視変数の値を返す
double GBRBMNode::getVisibleUnit(int vindex) {
	return v(vindex);
}

// 隠れ変数の値を返す
double GBRBMNode::getHiddenUnit(int hindex) {
	return h(hindex);
}

