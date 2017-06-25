#pragma once
class ConditionalRBMBase
{
public:
    ConditionalRBMBase() = default;
    ~ConditionalRBMBase() = default;

    // 可視変数の数を返す
    virtual size_t getVisibleSize() = 0;

    // 隠れ変数の数を返す
    virtual size_t getHiddenSize() = 0;

    // 条件変数の数を返す
    virtual size_t getCondSize() = 0;

    // 規格化を返します
    virtual double getNormalConstant() = 0;

    // エネルギー関数を返します
    virtual double getEnergy() = 0;

    // 自由エネルギーを返します
    virtual double getFreeEnergy() = 0;

    // 隠れ変数の活性化関数的なもの
    virtual double actHidJ(int hindex) = 0;

    // 可視変数に関する外部磁場と相互作用
    virtual double lambda(int vindex) = 0;

    // exp(lambda)の可視変数に関する全ての実現値の総和
    virtual double sumExpLambda(int vindex) = 0;

    // 隠れ変数に関する外部磁場と相互作用
    virtual double mu(int hindex) = 0;

    // exp(mu)の可視変数に関する全ての実現値の総和
    virtual double sumExpMu(int hindex) = 0;

    // 隠れ変数を条件で与えた可視変数の条件付き確率, P(v_i | h, x)
    virtual double condProbVis(int vindex, double value) = 0;

    // 可視変数を条件で与えた隠れ変数の条件付き確率, P(h_j | v, x)
    virtual double condProbHid(int hindex, double value) = 0;
};

