#pragma once
#include "RBMBase.h"
#include "GeneralizedRBMNode.h"
#include "GeneralizedRBMParamator.h"
#include <vector>

class GeneralizedRBM :
    public RBMBase
{
private:
    size_t vSize = 0;
    size_t hSize = 0;
    size_t divSize = 1;  // 隠れ変数の区間分割数
    std::vector <double> hiddenValueSet;  // 隠れ変数の取りうる値

public:
    GeneralizedRBMParamator params;
    GeneralizedRBMNode nodes;

public:
    GeneralizedRBM();
    GeneralizedRBM(size_t v_size, size_t h_size);
    ~GeneralizedRBM();

    // 可視変数の数を返す
    size_t getVisibleSize();

    // 隠れ変数の数を返す
    size_t getHiddenSize();

    // 規格化を返します
    double getNormalConstant();

    // エネルギー関数を返します
    double getEnergy();

    // 自由エネルギーを返します
    double getFreeEnergy();

    // 隠れ変数の活性化関数的なもの
    double actHidJ(int hindex);

    // 可視変数に関する外部磁場と相互作用
    double lambda(int vindex);

    // exp(lambda)の可視変数に関する全ての実現値の総和
    double sumExpLambda(int vindex);

    // 隠れ変数に関する外部磁場と相互作用
    double mu(int hindex);

    // exp(mu)の可視変数に関する全ての実現値の総和
    double sumExpMu(int hindex);

    // 隠れ変数を条件で与えた可視変数の条件付き確率, P(v_i | h)
    double condProbVis(int vindex, double value);

    // 可視変数を条件で与えた隠れ変数の条件付き確率, P(h_j | v)
    double condProbHid(int hindex, double value);

    // 隠れ変数の取りうる値を返す [-1, +1]をdivSizeで分割
    std::vector<double> splitHiddenSet();

    // 隠れ変数の取りうるパターン数
    int getHiddenValueSetSize();
};

