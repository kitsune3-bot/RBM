#pragma once
#include "../RBMBase.h"
#include "GeneralizedFullSparseRBMNode.h"
#include "GeneralizedFullSparseRBMParamator.h"
#include <vector>
#include "../RBMMath.h"
#include "../StateCounter.h"
#include <cmath>


class GeneralizedFullSparseRBM {
protected:
	size_t vSize = 0;
	size_t hSize = 0;
	double hMin = 0.0;
	double hMax = 1.0;
	size_t divSize = 1;  // 隠れ変数の区間分割数
	bool realFlag = false;
	std::vector <double> hiddenValueSet;  // 隠れ変数の取りうる値

public:
	GeneralizedFullSparseRBMParamator params;
	GeneralizedFullSparseRBMNode nodes;

public:
	GeneralizedFullSparseRBM() = default;
	GeneralizedFullSparseRBM(size_t v_size, size_t h_size);
	~GeneralizedFullSparseRBM() = default;

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

	// 隠れ変数の活性化関数的なもの
	double actHidJ(int hindex, double mu, double mu_star);

	// 可視変数に関する外部磁場と相互作用
	double lambda(int vindex);

	// exp(lambda)の可視変数に関する全ての実現値の総和
	double sumExpLambda(int vindex);

	// 隠れ変数に関する外部磁場と相互作用
	double mu(int hindex);

	// 隠れ変数に関する外部磁場と相互作用(一括計算)
	Eigen::VectorXd muVect();

	// 隠れ変数に関するスパースな外部磁場と相互作用
	double muStar(int hindex);

	// 隠れ変数に関するスパースな外部磁場と相互作用(一括計算)
	Eigen::VectorXd muStarVect();



	double sumHExpMuSparse(Eigen::VectorXd & mu_vect);

	// exp(mu+lambda)の可視変数に関する全ての実現値の総和
	double miniNormalizeConstantHidden(int hindex);

	// exp(mu+lambda)の可視変数に関する全ての実現値の総和
	double miniNormalizeConstantHidden(int hindex, double mu);

	// 可視変数の確率(隠れ変数周辺化済み)
	double probVis(std::vector<double> & data);

	// 可視変数の確率(隠れ変数周辺化済み, 分配関数使いまわし)
	double probVis(std::vector<double> & data, double normalize_constant);

	// 隠れ変数を条件で与えた可視変数の条件付き確率, P(v_i | h)
	double condProbVis(int vindex, double value);

	// 可視変数を条件で与えた隠れ変数の条件付き確率, P(h_j | v)
	double condProbHid(int hindex, double value);

	// 可視変数の期待値, E[v_i]
	double expectedValueVis(int vindex);

	// 可視変数の期待値, E[v_i](分配関数使いまわし)
	double expectedValueVis(int vindex, double normalize_constant);

	// 隠れ変数の期待値, E[h_j]
	double expectedValueHid(int hindex);

	// 隠れ変数の期待値, E[h_j](分配関数使いまわし)
	double expectedValueHid(int hindex, double normalize_constant);

	// 可視変数の期待値, E[v_i h_j]
	double expectedValueVisHid(int vindex, int hindex);

	// 可視変数の期待値, E[v_i h_j](分配関数使いまわし)
	double expectedValueVisHid(int vindex, int hindex, double normalize_constant);

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

	// 隠れ変数の期待値, E[ abs(h_j)]
	double expectedValueAbsHid(int hindex);

	// 隠れ変数の期待値, E[ abs(h_j)](分配関数使いまわし)
	double expectedValueAbsHid(int hindex, double normalize_constant);

	// 隠れ変数の(Abs)した活性化関数的なもの
	double actHidSparseJ(int hindex);

	// 隠れ変数の(Abs)した活性化関数的なもの
	double actHidSparseJ(int hindex, double mu, double mu_star);

	bool isRealHiddenValue();
};
