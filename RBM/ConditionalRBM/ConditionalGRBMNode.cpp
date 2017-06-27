#include "ConditionalGRBMNode.h"
#include <numeric>


ConditionalGRBMNode::ConditionalGRBMNode()
{
}


ConditionalGRBMNode::~ConditionalGRBMNode()
{
}

ConditionalGRBMNode::ConditionalGRBMNode(size_t v_size, size_t h_size, size_t x_size) {
    vSize = v_size;
    hSize = h_size;
    xSize = x_size;
    v.setConstant(v_size, 0.0);
    h.setConstant(h_size, 0.0);
    x.setConstant(x_size, 0.0);
}

std::vector<int> ConditionalGRBMNode::getVnodeIndexSet() {
    std::vector<int> set(vSize);
    std::iota(set.begin(), set.end(), 0);

    return set;
}

std::vector<int> ConditionalGRBMNode::getHnodeIndexSet() {
    std::vector<int> set(hSize);
    std::iota(set.begin(), set.end(), 0);

    return set;
}

std::vector<int> ConditionalGRBMNode::getXnodeIndexSet() {
    std::vector<int> set(xSize);
    std::iota(set.begin(), set.end(), 0);

    return set;
}
