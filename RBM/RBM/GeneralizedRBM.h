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
	double hMin = 0.0;
	double hMax = 1.0;
	size_t divSize = 1;  // 隠れ変数の区間分割数
	bool realFlag = false;
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

	// 可視変数の確率(隠れ変数周辺化済み)
	double probVis(std::vector<double> data);

	// 隠れ変数を条件で与えた可視変数の条件付き確率, P(v_i | h)
	double condProbVis(int vindex, double value);

	// 可視変数を条件で与えた隠れ変数の条件付き確率, P(h_j | v)
	double condProbHid(int hindex, double value);

	// 可視変数の期待値, E[v_i]
	double expectedValueVis(int vindex);

	// 隠れ変数の期待値, E[h_j]
	double expectedValueHid(int hindex);

	// 可視変数の期待値, E[v_i h_j]
	double expectedValueVisHis(int vindex, int hindex);

	//
	//appendix methods
	//

	// 隠れ変数の取りうる値を返す
	std::vector<double> splitHiddenSet();

	// 隠れ変数の取りうるパターン数
	int getHiddenValueSetSize();

	// 隠れ変数の取りうる最大値を取得
	double getHiddenMax();

	// 隠れ変数の取りうる最大値を設定
	void setHiddenMax(double value);

	// 隠れ変数の取りうる最小値を取得
	double getHiddenMin();

	// 隠れ変数の取りうる最小値を設定
	void setHiddenMin(double value);

	// 隠れ変数の区間分割数を返す
	size_t getHiddenDivSize();

	// 隠れ変数の区間分割数を設定
	void setHiddenDiveSize(size_t div_size);

	// 隠れ変数を連続値にするか離散値にするか
	void setRealHiddenValue(bool flag);
};

