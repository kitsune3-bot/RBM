#include "GeneralizedFullSparseRBMNode.h"


// 可視変数の総数を返す
size_t GeneralizedFullSparseRBMNode::getVisibleSize() {
	return vSize;
}

// 隠れ変数の総数を返す
size_t GeneralizedFullSparseRBMNode::getHiddenSize() {
	return hSize;
}

// 可視層の全てを返す
Eigen::VectorXd GeneralizedFullSparseRBMNode::getVisibleLayer() {
	return v;
}

// 隠れ層の全てを返す
Eigen::VectorXd GeneralizedFullSparseRBMNode::getHiddenLayer() {
	return h;
}

// 可視変数の値を返す
double GeneralizedFullSparseRBMNode::getVisibleUnit(int vindex) {
	return v(vindex);
}

// 隠れ変数の値を返す
double GeneralizedFullSparseRBMNode::getHiddenUnit(int hindex) {
	return h(hindex);
}




GeneralizedFullSparseRBMNode::GeneralizedFullSparseRBMNode(size_t v_size, size_t h_size) {
	vSize = v_size;
	hSize = h_size;
	v.setConstant(v_size, 0.0);
	h.setConstant(h_size, 0.0);
}

std::vector<int> GeneralizedFullSparseRBMNode::getVnodeIndexSet() {
	std::vector<int> set(vSize);
	std::iota(set.begin(), set.end(), 0);

	return set;
}

std::vector<int> GeneralizedFullSparseRBMNode::getHnodeIndexSet() {
	std::vector<int> set(hSize);
	std::iota(set.begin(), set.end(), 0);

	return set;
}
