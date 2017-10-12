#include "GBRBM.h"
#include "../RBMMath.h"
#include "../StateCounter.h"
#include <cmath>


GBRBM::GBRBM()
{
}


GBRBM::~GBRBM()
{
}

GBRBM::GBRBM(size_t v_size, size_t h_size) {
    vSize = v_size;
    hSize = h_size;

    // ノード確保
    nodes = GBRBMNode(v_size, h_size);

    // パラメータ初期化
    params = GBRBMParamator(v_size, h_size);
    params.initParamsRandom(-0.01, 0.01);
}


// 可視変数の数を返す
size_t GBRBM::getVisibleSize() {
    return vSize;
}

// 隠れ変数の数を返す
size_t GBRBM::getHiddenSize() {
    return hSize;
}


// 規格化を返します
double GBRBM::getNormalConstant() {
	throw;
	// 以下全部間違ってます///

	StateCounter<std::vector<int>> sc(std::vector<int>(vSize, 2));  // 可視変数Vの状態カウンター

	double z = 0.0;
	auto max_count = sc.getMaxCount();
	for (int i = 0; i < max_count; i++, sc++) {
		// TODO: ここに状態数の計算を記述せよ
		int state_map[] = { 0, 1 };  // 状態->値変換写像

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
double GBRBM::getEnergy() {
    // まだ必要ないため実装保留
    throw;
}


// 自由エネルギーを返します
double GBRBM::getFreeEnergy() {
    return -log(this->getNormalConstant());
}

// 隠れ変数の活性化関数的なもの
double GBRBM::actHidJ(int hindex) {
    return RBMMath::sigmoid(mu(hindex));
}

// 可視変数に関する外部磁場と相互作用
// ただし二乗の項を除く
double GBRBM::lambda(int vindex) {
    double lam = params.b(vindex);

    // TODO: Eigen使ってるから内積計算で高速化できる
    for (int j = 0; j < hSize; j++) {
        lam += params.w(vindex, j) * nodes.h(j);
    }

    return lam;
}

// lambdaの可視変数に関する全ての実現値の総和
double GBRBM::sumExpLambda(int vindex) {
    // XXX: 使いません
    throw;
    return 1.0 + exp(lambda(vindex));
}

// v_iの平均
double GBRBM::meanVisible(int vindex) {
    double inv_var = params.lambda(vindex);;  // 逆分散
    double mean = lambda(vindex) / inv_var;
    return mean;
}


// v_iの条件付き確率の規格化定数
double GBRBM::integralExpVisible(int vindex) {
    return sqrt(2 * 3.141592 / params.lambda(vindex));
}


// 隠れ変数に関する外部磁場と相互作用
double GBRBM::mu(int hindex) {
    double mu = params.c(hindex);

    // TODO: Eigen使ってるから内積計算で高速化できる
    for (int i = 0; i < vSize; i++) {
        mu += params.w(i, hindex) * nodes.v(i);
    }

    return mu;
}

// muの可視変数に関する全ての実現値の総和
double GBRBM::sumExpMu(int hindex) {
    // {0, 1}での実装
    return 1.0 + exp(mu(hindex));
}

// 隠れ変数を条件で与えた可視変数の条件付き確率, P(v_i | h)
double GBRBM::condProbVis(int vindex, double value) {
    double lam = lambda(vindex);
    double inv_var = params.lambda[vindex];  // 逆分散
    double mean = meanVisible(vindex);  // 可視変数期待値
    return exp(inv_var / 2.0 * pow(value - mean, 2)) / integralExpVisible(vindex);
}

// 可視変数を条件で与えた隠れ変数の条件付き確率, P(h_j | v)
double GBRBM::condProbHid(int hindex, double value) {
    double m = mu(hindex);
    return exp(m * value) / sumExpMu(hindex);
}

