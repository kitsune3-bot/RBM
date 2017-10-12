#pragma once
#include <limits>
#include "../RBMBase.h"
#include "GeneralizedGRBMParamator.h"
#include "GeneralizedGRBMNode.h"
#include "../RBMMath.h"
#include "../StateCounter.h"
#include <cmath>


class GeneralizedGRBM : RBMBase {
private:
    size_t vSize = 0;
    size_t hSize = 0;
    double hMin = 0.0;
    double hMax = 1.0;
    size_t divSize = 1;  // 隠れ変数の区間分割数
	bool realFlag = false;
    std::vector <double> hiddenValueSet;  // 隠れ変数の取りうる値

public:
    GeneralizedGRBMParamator params;
    GeneralizedGRBMNode nodes;

public:
    GeneralizedGRBM() = default;
    GeneralizedGRBM(size_t v_size, size_t h_size);
    ~GeneralizedGRBM() = default;

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
    // 二乗の項は無視
    double lambda(int vindex);

    // exp(lambda)の可視変数に関する全ての実現値の総和
    // Gaussian Unitでは使いません
    double sumExpLambda(int vindex);

    // v_iの条件付き確率の規格化定数
    double integralExpVisible(int vindex);

    // v_iの平均
    double meanVisible(int vindex);

    // 隠れ変数に関する外部磁場と相互作用
    double mu(int hindex);

    // exp(mu)の可視変数に関する全ての実現値の総和
    double sumExpMu(int hindex);

    // 隠れ変数を条件で与えた可視変数の条件付き確率, P(v_i | h)
    double condProbVis(int vindex, double value);

    // 可視変数を条件で与えた隠れ変数の条件付き確率, P(h_j | v)
    double condProbHid(int hindex, double value);

	// 可視変数の期待値, E[v_i]
	double expectedValueVis(int vindex);

	// 隠れ変数の期待値, E[h_j]
	double expectedValueHid(int hindex);


    //
    // Generalized Method
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
	//void setRealHiddenValue(bool flag);
};



inline GeneralizedGRBM::GeneralizedGRBM(size_t v_size, size_t h_size) {
	vSize = v_size;
	hSize = h_size;

	// ノード確保
	nodes = GeneralizedGRBMNode(v_size, h_size);

	// パラメータ初期化
	params = GeneralizedGRBMParamator(v_size, h_size);
	params.initParamsRandom(-0.01, 0.01);

	// 区間分割
	hiddenValueSet = splitHiddenSet();
}


// 可視変数の数を返す
inline size_t GeneralizedGRBM::getVisibleSize() {
	return vSize;
}

// 隠れ変数の数を返す
inline size_t GeneralizedGRBM::getHiddenSize() {
	return hSize;
}


// 規格化を返します
inline double GeneralizedGRBM::getNormalConstant() {
	throw;
	// 以下全部間違ってます///

	StateCounter<std::vector<int>> sc(std::vector<int>(vSize, this->divSize + 1));  // 可視変数Vの状態カウンター
	auto state_map = this->splitHiddenSet();  // 状態->値変換写像


	double z = 0.0;
	auto max_count = sc.getMaxCount();
	for (int i = 0; i < max_count; i++, sc++) {
		// TODO: ここに状態数の計算を記述せよ

		// FIXME: stlのコピーは遅いぞ
		auto v_state = sc.getState();
		for (int i = 0; i < vSize; i++) {
			this->nodes.v(i) = state_map[v_state[i]];
		}

		// 項計算
		double term = exp(nodes.getVisibleLayer().dot(params.b));
		for (int j = 0; j < hSize; j++) {
			term *= 1 + exp(mu(j));
		}

		z += term;
	}

	return z;
}


// エネルギー関数を返します
inline double GeneralizedGRBM::getEnergy() {
	// まだ必要ないため実装保留
	throw;
}


// 自由エネルギーを返します
inline double GeneralizedGRBM::getFreeEnergy() {
	return -log(this->getNormalConstant());
}

// 隠れ変数の活性化関数的なもの
inline double GeneralizedGRBM::actHidJ(int hindex) {
	auto value_set = splitHiddenSet();
	double numer = 0.0;  // 分子
	double denom = sumExpMu(hindex);  // 分母

	for (auto & value : value_set) {
		numer += value * exp(mu(hindex) * value);
	}

	return numer / denom;
}

// 可視変数に関する外部磁場と相互作用
// ただし二乗の項を除く
inline double GeneralizedGRBM::lambda(int vindex) {
	double lam = params.b(vindex);

	// TODO: Eigen使ってるから内積計算で高速化できる
	for (int j = 0; j < hSize; j++) {
		lam += params.w(vindex, j) * nodes.h(j);
	}

	return lam;
}

// lambdaの可視変数に関する全ての実現値の総和
inline double GeneralizedGRBM::sumExpLambda(int vindex) {
	// XXX: 使いません
	throw;
	return 1.0 + exp(lambda(vindex));
}

// v_iの平均
inline double GeneralizedGRBM::meanVisible(int vindex) {
	double inv_var = params.lambda(vindex);;  // 逆分散
	double mean = lambda(vindex) / inv_var;
	return mean;
}


// v_iの条件付き確率の規格化定数
inline double GeneralizedGRBM::integralExpVisible(int vindex) {
	return sqrt(2 * 3.141592 / params.lambda(vindex));
}


// 隠れ変数に関する外部磁場と相互作用
inline double GeneralizedGRBM::mu(int hindex) {
	double mu = params.c(hindex);

	// TODO: Eigen使ってるから内積計算で高速化できる
	for (int i = 0; i < vSize; i++) {
		mu += params.w(i, hindex) * nodes.v(i);
	}

	return mu;
}

// muの可視変数に関する全ての実現値の総和
inline double GeneralizedGRBM::sumExpMu(int hindex) {
	double sum = 0.0;

	for (auto & value : hiddenValueSet) {
		sum += exp(mu(hindex) * value);
	}

	return sum;
}

// 隠れ変数を条件で与えた可視変数の条件付き確率, P(v_i | h)
inline double GeneralizedGRBM::condProbVis(int vindex, double value) {
	double lam = lambda(vindex);
	double inv_var = params.lambda[vindex];  // 逆分散
	double mean = meanVisible(vindex);  // 可視変数期待値
	return exp(inv_var / 2.0 * pow(value - mean, 2)) / integralExpVisible(vindex);
}

// 可視変数を条件で与えた隠れ変数の条件付き確率, P(h_j | v)
inline double GeneralizedGRBM::condProbHid(int hindex, double value) {
	double m = mu(hindex);
	double prob = exp(m * value) / sumExpMu(hindex);
	return prob;
}

inline std::vector<double> GeneralizedGRBM::splitHiddenSet() {
	std::vector<double> set(divSize + 1);

	auto x = [](double split_size, double i, double min, double max) {  // 分割関数[i=0,1,...,elems]
		return 1.0 / (split_size)* i * (max - min) + min;
	};

	for (int i = 0; i < set.size(); i++) set[i] = x(divSize, i, hMin, hMax);

	return set;
}

inline int GeneralizedGRBM::getHiddenValueSetSize() {
	return divSize + 1;
}

// 隠れ変数の取りうる最大値を取得
inline double GeneralizedGRBM::getHiddenMax() {
	return hMax;
}

// 隠れ変数の取りうる最大値を設定
inline void GeneralizedGRBM::setHiddenMax(double value) {
	hMax = value;

	// 区間分割
	hiddenValueSet = splitHiddenSet();
}

// 隠れ変数の取りうる最小値を取得
inline double GeneralizedGRBM::getHiddenMin() {
	return hMin;
}

// 隠れ変数の取りうる最小値を設定
inline void GeneralizedGRBM::setHiddenMin(double value) {
	hMin = value;

	// 区間分割
	hiddenValueSet = splitHiddenSet();
}

// 隠れ変数の区間分割数を返す
inline size_t GeneralizedGRBM::getHiddenDivSize() {
	return divSize;
}

// 隠れ変数の区間分割数を設定
inline void GeneralizedGRBM::setHiddenDiveSize(size_t div_size) {
	divSize = div_size;

	// 区間分割
	hiddenValueSet = splitHiddenSet();
}

