#include "GeneralizedRBM.h"
#include "RBMMath.h"
#include "StateCounter.h"
#include <cmath>


GeneralizedRBM::GeneralizedRBM()
{
}


GeneralizedRBM::~GeneralizedRBM()
{
}

GeneralizedRBM::GeneralizedRBM(size_t v_size, size_t h_size) {
	vSize = v_size;
	hSize = h_size;

	// ノード確保
	nodes = GeneralizedRBMNode(v_size, h_size);

	// パラメータ初期化
	params = GeneralizedRBMParamator(v_size, h_size);
	params.initParamsRandom(0.01, 0.01);

	// 区間分割
	hiddenValueSet = splitHiddenSet();
}


// 可視変数の数を返す
size_t GeneralizedRBM::getVisibleSize() {
	return vSize;
}

// 隠れ変数の数を返す
size_t GeneralizedRBM::getHiddenSize() {
	return hSize;
}


// 規格化を返します
double GeneralizedRBM::getNormalConstant() {
	StateCounter<std::vector<int>> sc(std::vector<int>(vSize, 2));  // 可視変数Vの状態カウンター
	int v_state_map[] = { 0, 1 };  // 可視変数の状態->値変換写像

	double z = 0.0;
	auto max_count = sc.getMaxCount();
	for (int c = 0; c < max_count; c++, sc++) {
		// FIXME: stlのコピーは遅いぞ
		auto v_state = sc.getState();
		for (int i = 0; i < vSize; i++) {
			this->nodes.v(i) = v_state_map[v_state[i]];
		}

		// 項計算
		double term = exp(nodes.getVisibleLayer().dot(params.b));
		for (int j = 0; j < hSize; j++) {
			auto mu_j = mu(j);

			auto sum_h_j_discrete = [&](double mu_j) {
				double sum = 0.0;

				for (auto & h_val : this->hiddenValueSet) {
					sum += exp(mu_j * h_val);
				}

				return sum;
			};

			auto sum_h_j_real = [&](double mu_j) {
				double sum = (exp(hMax * mu_j) - exp(hMin * mu_j)) / mu_j;

				return sum;
			};

			auto sum_h_j = realFlag ? sum_h_j_real(mu_j) : sum_h_j_discrete(mu_j);

			term *= sum_h_j;
		}

		z += term;
	}

	return z;
}


// エネルギー関数を返します
double GeneralizedRBM::getEnergy() {
	// まだ必要ないため実装保留
	throw;
}


// 自由エネルギーを返します
double GeneralizedRBM::getFreeEnergy() {
	return -log(this->getNormalConstant());
}

// 隠れ変数の活性化関数的なもの
double GeneralizedRBM::actHidJ(int hindex) {
	auto value_set = splitHiddenSet();
	double numer = 0.0;  // 分子
	double denom = sumExpMu(hindex);  // 分母

	for (auto & value : value_set) {
		numer += value * exp(mu(hindex) * value);
	}

	return numer / denom;
}

// 可視変数に関する外部磁場と相互作用
double GeneralizedRBM::lambda(int vindex) {
	double lam = params.b(vindex);

	// TODO: Eigen使ってるから内積計算で高速化できる
	for (int j = 0; j < hSize; j++) {
		lam += params.w(vindex, j) * nodes.h(j);
	}

	return lam;
}

// lambdaの可視変数に関する全ての実現値の総和
double GeneralizedRBM::sumExpLambda(int vindex) {
	// {0, 1}での実装
	return 1.0 + exp(lambda(vindex));
}

// 隠れ変数に関する外部磁場と相互作用
double GeneralizedRBM::mu(int hindex) {
	double mu = params.c(hindex);

	// TODO: Eigen使ってるから内積計算で高速化できる
	for (int i = 0; i < vSize; i++) {
		mu += params.w(i, hindex) * nodes.v(i);
	}

	return mu;
}

// muの可視変数に関する全ての実現値の総和
double GeneralizedRBM::sumExpMu(int hindex) {
	double sum = 0.0;

	for (auto & value : hiddenValueSet) {
		sum += exp(mu(hindex) * value);
	}

	return sum;
}

// 可視変数の確率(隠れ変数周辺化済み)
double GeneralizedRBM::probVis(std::vector<double> & data) {
	// 分配関数
	double z = getNormalConstant();

	for (int i = 0; i < getVisibleSize(); i++) {
		this->nodes.v(i) = data[i];
	}

	// bとvの内積
	auto b_dot_v = [&]() {
		return nodes.getVisibleLayer().dot(params.b);
	};

	// 隠れ変数h_jの値の総和計算
	auto sum_h_j = [&](int j) {
		auto mu_j = mu(j);

		// 離散型
		auto sum_h_j_discrete = [&](double mu_j) {
			double sum = 0.0;

			for (auto & h_val : this->hiddenValueSet) {
				sum += exp(mu_j * h_val);
			}

			return sum;
		};

		// 連続型
		auto sum_h_j_real = [&](double mu_j) {
			double sum = (exp(hMax * mu_j) - exp(hMin * mu_j)) / mu_j;

			return sum;
		};

		auto value = realFlag ? sum_h_j_real(mu_j) : sum_h_j_discrete(mu_j);

		return value;
	};

	double value = exp(b_dot_v()) / z;

	for (int j = 0; j < hSize; j++) {
		value *= sum_h_j(j);
	}

	return value;
}


// 隠れ変数を条件で与えた可視変数の条件付き確率, P(v_i | h)
double GeneralizedRBM::condProbVis(int vindex, double value) {
	double lam = lambda(vindex);
	return exp(lam * value) / sumExpLambda(vindex);
}

// 可視変数を条件で与えた隠れ変数の条件付き確率, P(h_j | v)
double GeneralizedRBM::condProbHid(int hindex, double value) {
	double m = mu(hindex);
	double prob = exp(m * value) / sumExpMu(hindex);
	return prob;
}

std::vector<double> GeneralizedRBM::splitHiddenSet() {
	std::vector<double> set(divSize + 1);

	auto x = [](double split_size, double i, double min, double max) {  // 分割関数[i=0,1,...,elems]
		return 1.0 / (split_size)* i * (max - min) + min;
	};

	for (int i = 0; i < set.size(); i++) set[i] = x(divSize, i, hMin, hMax);

	return set;
}

int GeneralizedRBM::getHiddenValueSetSize() {
	return divSize + 1;
}

// 隠れ変数の取りうる最大値を取得
double GeneralizedRBM::getHiddenMax() {
	return hMax;
}

// 隠れ変数の取りうる最大値を設定
void GeneralizedRBM::setHiddenMax(double value) {
	hMax = value;

	// 区間分割
	hiddenValueSet = splitHiddenSet();
}

// 隠れ変数の取りうる最小値を取得
double GeneralizedRBM::getHiddenMin() {
	return hMin;
}

// 隠れ変数の取りうる最小値を設定
void GeneralizedRBM::setHiddenMin(double value) {
	hMin = value;

	// 区間分割
	hiddenValueSet = splitHiddenSet();
}

// 隠れ変数の区間分割数を返す
size_t GeneralizedRBM::getHiddenDivSize() {
	return divSize;
}

// 隠れ変数の区間分割数を設定
void GeneralizedRBM::setHiddenDiveSize(size_t div_size) {
	divSize = div_size;

	// 区間分割
	hiddenValueSet = splitHiddenSet();
}

void GeneralizedRBM::setRealHiddenValue(bool flag) {
	realFlag = flag;
}



// 可視変数の期待値, E[v_i]
double GeneralizedRBM::expectedValueVis(int vindex) {
	StateCounter<std::vector<int>> sc(std::vector<int>(vSize, 2));  // 可視変数Vの状態カウンター
	int v_state_map[] = { 0, 1 };  // 可視変数の状態->値変換写像

	auto z = getNormalConstant();  // 分配関数
	
	// bとvの内積
	auto b_dot_v = [&](){
		return nodes.getVisibleLayer().dot(params.b);
	};

	// 隠れ変数h_jの値の総和計算
	auto sum_h_j = [&](int j) {
		auto mu_j = mu(j);

		// 離散型
		auto sum_h_j_discrete = [&](double mu_j) {
			double sum = 0.0;

			for (auto & h_val : this->hiddenValueSet) {
				sum += exp(mu_j * h_val);
			}

			return sum;
		};

		// 連続型
		auto sum_h_j_real = [&](double mu_j) {
			double sum = (exp(hMax * mu_j) - exp(hMin * mu_j)) / mu_j;

			return sum;
		};

		auto value = realFlag ? sum_h_j_real(mu_j) : sum_h_j_discrete(mu_j);

		return value;
	};

	double value = 1.0 / z;

	auto max_count = sc.getMaxCount();
	for (int c = 0; c < max_count; c++, sc++) {
		// 項計算の前処理
		// FIXME: stlのコピーは遅いぞ
		auto v_state = sc.getState();

		// FIXME: v_i == 0 ときそのままcontinueしたほうが速いぞ
		//
		for (int i = 0; i < vSize; i++) {
			this->nodes.v(i) = v_state_map[v_state[i]];
		}

		// 項計算
		double term =this->nodes.v(vindex) * exp(b_dot_v());

		for (int j = 0; j < hSize; j++) {
			term *= sum_h_j(j);
		}

		value += term;
	}

	return value;
}


// 隠れ変数の期待値, E[h_j]
double GeneralizedRBM::expectedValueHid(int hindex) {
	StateCounter<std::vector<int>> sc(std::vector<int>(vSize, 2));  // 可視変数Vの状態カウンター
	int v_state_map[] = { 0, 1 };  // 可視変数の状態->値変換写像

	auto z = getNormalConstant();  // 分配関数

								   // bとvの内積
	auto b_dot_v = [&]() {
		return nodes.getVisibleLayer().dot(params.b);
	};

	// sum( h_j exp(mu_j h_j))
	auto sum_h_j = [&](int j) {
		auto mu_j = mu(j);

		// 離散型
		auto sum_h_j_discrete = [&](double mu_j) {
			double sum = 0.0;

			for (auto & h_val : this->hiddenValueSet) {
				sum += h_val * exp(mu_j * h_val);
			}

			return sum;
		};

		// 連続型
		auto sum_h_j_real = [&](double mu_j) {
			double sum = ( ( hMax * exp(hMax * mu_j) - hMin * exp(hMin * mu_j) ) / mu_j) - ( ( exp(hMax * mu_j) - exp(hMin * mu_j ) ) / (mu_j * mu_j) );

			return sum;
		};

		auto value = realFlag ? sum_h_j_real(mu_j) : sum_h_j_discrete(mu_j);

		return value;
	};

	// 隠れ変数h_lの値の総和計算
	auto sum_h_l = [&](int l) {
		auto mu_l = mu(l);

		// 離散型
		auto sum_h_l_discrete = [&](double mu_l) {
			double sum = 0.0;

			for (auto & h_val : this->hiddenValueSet) {
				sum += exp(mu_l * h_val);
			}

			return sum;
		};

		// 連続型
		auto sum_h_l_real = [&](double mu_l) {
			double sum = (exp(hMax * mu_l) - exp(hMin * mu_l)) / mu_l;

			return sum;
		};

		auto value = realFlag ? sum_h_l_real(mu_l) : sum_h_l_discrete(mu_l);

		return value;
	};


	double value = 1.0 / z;
	auto max_count = sc.getMaxCount();
	for (int c = 0; c < max_count; c++, sc++) {
		// FIXME: stlのコピーは遅いぞ
		auto v_state = sc.getState();

		// FIXME: v_i == 0 ときそのままcontinueしたほうが速いぞ

		for (int i = 0; i < vSize; i++) {
			this->nodes.v(i) = v_state_map[v_state[i]];
		}

		// 項計算
		double term = exp(b_dot_v());

		term *= sum_h_j(hindex);

		for (int l = 0; l < hSize; l++) {
			if (l == hindex) continue;

			term *= sum_h_l(l);
		}

		value += term;
	}

	return value;
}


// 隠れ変数の期待値, E[v_i h_j]
double GeneralizedRBM::expectedValueVisHid(int vindex, int hindex) {
	throw;
	// まだ未実装

	StateCounter<std::vector<int>> sc(std::vector<int>(vSize, 2));  // 可視変数Vの状態カウンター
	int v_state_map[] = { 0, 1 };  // 可視変数の状態->値変換写像

	auto z = getNormalConstant();  // 分配関数

								   // bとvの内積
	auto b_dot_v = [&]() {
		return nodes.getVisibleLayer().dot(params.b);
	};

	// sum( h_j exp(mu_j h_j))
	auto sum_h_j = [&](int j) {
		auto mu_j = mu(j);

		// 離散型
		auto sum_h_j_discrete = [&](double mu_j) {
			double sum = 0.0;

			for (auto & h_val : this->hiddenValueSet) {
				sum += h_val * exp(mu_j * h_val);
			}

			return sum;
		};

		// 連続型
		auto sum_h_j_real = [&](double mu_j) {
			double sum = ((hMax * exp(hMax * mu_j) - hMin * exp(hMin * mu_j)) / mu_j) - ((exp(hMax * mu_j) - exp(hMin * mu_j)) / (mu_j * mu_j));

			return sum;
		};

		auto value = realFlag ? sum_h_j_real(mu_j) : sum_h_j_discrete(mu_j);

		return value;
	};

	// 隠れ変数h_lの値の総和計算
	auto sum_h_l = [&](int j) {
		auto mu_j = mu(j);

		// 離散型
		auto sum_h_j_discrete = [&](double mu_j) {
			double sum = 0.0;

			for (auto & h_val : this->hiddenValueSet) {
				sum += exp(mu_j * h_val);
			}

			return sum;
		};

		// 連続型
		auto sum_h_j_real = [&](double mu_j) {
			double sum = (exp(hMax * mu_j) - exp(hMin * mu_j)) / mu_j;

			return sum;
		};

		auto value = realFlag ? sum_h_j_real(mu_j) : sum_h_j_discrete(mu_j);

		return value;
	};
	
	double value = 1.0 / z;
	auto max_count = sc.getMaxCount();
	for (int c = 0; c < max_count; c++, sc++) {
		// FIXME: stlのコピーは遅いぞ
		auto v_state = sc.getState();

		// FIXME: v_i == 0 ときそのままcontinueしたほうが速いぞ

		for (int i = 0; i < vSize; i++) {
			this->nodes.v(i) = v_state_map[v_state[i]];
		}

		// 項計算
		double term = this->nodes.v(vindex) * exp(b_dot_v());

		term *= sum_h_j(hindex);

		for (int l = 0; l < hSize; l++) {
			if (l == hindex) continue;

			term *= sum_h_l(l);
		}

		value += term;
	}

	return value;
}
