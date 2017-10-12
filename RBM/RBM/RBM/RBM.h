#pragma once
#include "../RBMBase.h"
#include "RBMParamator.h"
#include "RBMNode.h"
#include "../RBMMath.h"
#include "../StateCounter.h"
#include <cmath>
#include <vector>
#include <numeric>

//
// Restricted Boltzmann Machine(variables \in {0, 1})
//
class RBM : RBMBase {
private:
	size_t vSize = 0;
	size_t hSize = 0;

public:
	RBMParamator params;
	RBMNode nodes;

public:
	RBM() = default;
	RBM(size_t v_size, size_t h_size);
	~RBM() = default;

	// 可視変数の数を返す
	size_t getVisibleSize();

	// 隠れ変数の数を返す
	size_t getHiddenSize();

	// 規格化定数を返します
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

	// 可視変数の期待値, E[v_i]
	double expectedValueVis(int vindex);

	// 隠れ変数の期待値, E[h_j]
	double expectedValueHid(int hindex);
};




inline RBM::RBM(size_t v_size, size_t h_size) {
	vSize = v_size;
	hSize = h_size;

	// ノード確保
	nodes = RBMNode(v_size, h_size);

	// パラメータ初期化
	params = RBMParamator(v_size, h_size);
	params.initParamsRandom(0.01, 0.01);
}


// 可視変数の数を返す
inline size_t RBM::getVisibleSize() {
	return vSize;
}

// 隠れ変数の数を返す
inline size_t RBM::getHiddenSize() {
	return hSize;
}


// 規格化を返します
inline double RBM::getNormalConstant() {
	StateCounter<std::vector<int>> sc(std::vector<int>(vSize, 2));  // 可視変数Vの状態カウンター
	int v_state_map[] = { 0, 1 };  // 可視変数の状態->値変換写像


	double z = 0.0;
	auto max_count = sc.getMaxCount();
	for (int i = 0; i < max_count; i++, sc++) {
		// FIXME: stlのコピーは遅いぞ
		auto v_state = sc.getState();
		for (int i = 0; i < vSize; i++) {
			this->nodes.v(i) = v_state_map[v_state[i]];
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
inline double RBM::getEnergy() {
	// まだ必要ないため実装保留
	throw;
}


// 自由エネルギーを返します
inline double RBM::getFreeEnergy() {
	return -log(this->getNormalConstant());
}

// 隠れ変数の活性化関数的なもの
inline double RBM::actHidJ(int hindex) {
	return RBMMath::sigmoid(mu(hindex));
}

// 可視変数に関する外部磁場と相互作用
inline double RBM::lambda(int vindex) {
	double lam = params.b(vindex);

	// TODO: Eigen使ってるから内積計算で高速化できる
	for (int j = 0; j < hSize; j++) {
		lam += params.w(vindex, j) * nodes.h(j);
	}

	return lam;
}

// lambdaの可視変数に関する全ての実現値の総和
inline double RBM::sumExpLambda(int vindex) {
	// {0, 1}での実装
	return 1.0 + exp(lambda(vindex));
}

// 隠れ変数に関する外部磁場と相互作用
inline double RBM::mu(int hindex) {
	double mu = params.c(hindex);

	// TODO: Eigen使ってるから内積計算で高速化できる
	for (int i = 0; i < vSize; i++) {
		mu += params.w(i, hindex) * nodes.v(i);
	}

	return mu;
}

// muの可視変数に関する全ての実現値の総和
inline double RBM::sumExpMu(int hindex) {
	// {0, 1}での実装
	return 1.0 + exp(mu(hindex));
}

// 隠れ変数を条件で与えた可視変数の条件付き確率, P(v_i | h)
inline double RBM::condProbVis(int vindex, double value) {
	double lam = lambda(vindex);
	return exp(lam * value) / sumExpLambda(vindex);
}

// 可視変数を条件で与えた隠れ変数の条件付き確率, P(h_j | v)
inline double RBM::condProbHid(int hindex, double value) {
	double m = mu(hindex);
	return exp(m * value) / sumExpMu(hindex);
}

