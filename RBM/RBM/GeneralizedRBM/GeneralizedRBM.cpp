#include "GeneralizedRBM.h"


GeneralizedRBM::GeneralizedRBM(size_t v_size, size_t h_size) {
	vSize = v_size;
	hSize = h_size;

	// ノード確保
	nodes = GeneralizedRBMNode(v_size, h_size);

	// パラメータ初期化
	params = GeneralizedRBMParamator(v_size, h_size);
	params.initParamsRandom(-0.1, 0.1);

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
	auto mu = this->mu(hindex);
	auto value = this->actHidJ(hindex, mu);

	return value;
}

double GeneralizedRBM::actHidJ(int hindex, double mu)
{

	// 離散型
	auto discrete = [&]()
	{
		auto value_set = splitHiddenSet();
		auto mu_j = mu;
		double numer = 0.0;  // 分子
		double denom = miniNormalizeConstantHidden(hindex, mu_j);  // 分母
		for (auto & h_j : value_set) {
			numer += h_j * exp(mu_j * h_j);
		}

		auto value = numer / denom;

		return value;
	};

	// 連続型
	auto real = [&]() {
		auto mu_j = mu;
		// FIXME: 0除算の可能性あり, 要テイラー展開
		auto value = (hMax * exp(hMax * mu_j) - hMin * exp(hMin * mu_j)) / (exp(hMax * mu_j) - exp(hMin * mu_j)) - 1 / mu_j;

		return value;
	};

	auto value = realFlag ? real() : discrete();

	return value;
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

Eigen::VectorXd GeneralizedRBM::lambdaVect()
{
	Eigen::VectorXd lambda_vect(this->vSize);
	for (int i = 0; i < this->vSize; i++) {
		lambda_vect(i) = mu(i);
	}

	return lambda_vect;
}

// lambdaの可視変数に関する全ての実現値の総和
double GeneralizedRBM::sumExpLambda(int vindex) {
	// {0, 1}での実装
	return 1.0 + exp(lambda(vindex));
}

double GeneralizedRBM::sumExpLambda(int vindex, double lambda)
{
	// {0, 1}での実装
	return 1.0 + exp(lambda);
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

Eigen::VectorXd GeneralizedRBM::muVect()
{
	Eigen::VectorXd mu_vect(this->hSize);
	for (int j = 0; j < this->hSize; j++) {
		mu_vect(j) = mu(j);
	}

	return mu_vect;
}

double GeneralizedRBM::sumHExpMu(Eigen::VectorXd & mu_vect)
{
	double value = 1.0;
	for (int j = 0; j < this->hSize; j++) {
		value *= miniNormalizeConstantHidden(j, mu_vect(j));
	}

	return value;
}

// muの可視変数に関する全ての実現値の総和
double GeneralizedRBM::miniNormalizeConstantHidden(int hindex) {
	// 離散型
	auto sum_discrete = [&]() {
		double value = 0.0;
		double mu_j = mu(hindex);

		for (auto & h_j : hiddenValueSet) {
			value += exp(mu_j * h_j);
		}

		return value;
	};

	// 連続型
	auto sum_real = [&]() {
		double mu_j = mu(hindex);
		double value = (exp(hMax * mu_j) - exp(hMin * mu_j)) / mu_j;

		return value;
	};


	double sum = realFlag ? sum_real() : sum_discrete();

	return sum;
}

// muの可視変数に関する全ての実現値の総和
double GeneralizedRBM::miniNormalizeConstantHidden(int hindex, double mu) {
	// 離散型
	auto sum_discrete = [&]() {
		double value = 0.0;
		double mu_j = mu;

		for (auto & h_j : hiddenValueSet) {
			value += exp(mu_j * h_j);
		}

		return value;
	};

	// 連続型
	auto sum_real = [&]() {
		double mu_j = mu;
		double value = (exp(hMax * mu_j) - exp(hMin * mu_j)) / mu_j;

		return value;
	};


	double sum = realFlag ? sum_real() : sum_discrete();

	return sum;
}


// 可視変数の確率(隠れ変数周辺化済み)
double GeneralizedRBM::probVis(std::vector<double> & data) {
	// 分配関数
	double z = getNormalConstant();

	return probVis(data, z);
}

// 可視変数の確率(隠れ変数周辺化済み, 分配関数使いまわし)
double GeneralizedRBM::probVis(std::vector<double> & data, double normalize_constant) {
	auto & z = normalize_constant;

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
	return this->condProbVis(vindex, value, this->lambda(vindex));
}

double GeneralizedRBM::condProbVis(int vindex, double value, double lambda)
{
	auto prob = exp(lambda * value) / sumExpLambda(vindex, lambda);
	return prob;
}

// 可視変数を条件で与えた隠れ変数の条件付き確率, P(h_j | v)
double GeneralizedRBM::condProbHid(int hindex, double value) {
	double prob = this->condProbHid(hindex, value, this->mu(hindex));
	return prob;
}

double GeneralizedRBM::condProbHid(int hindex, double value, double mu)
{
	double mu_j = mu;
	double prob = exp(mu_j * value) / miniNormalizeConstantHidden(hindex, mu_j);
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
	auto z = getNormalConstant();  // 分配関数

	return expectedValueVis(vindex, z);
}

// 可視変数の期待値, E[v_i]
double GeneralizedRBM::expectedValueVis(int vindex, double normalize_constant) {
	// TODO: とりあえず可視変数は{0, 1}のボルツマンマシンなので則値代入してます
	StateCounter<std::vector<int>> sc(std::vector<int>(vSize, 2));  // 可視変数Vの状態カウンター
	int v_state_map[] = { 0, 1 };  // 可視変数の状態->値変換写像

	auto & z = normalize_constant;

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

	double value = 0.0;

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
		// bとvの内積
		auto b_dot_v = [&]() {
			return nodes.getVisibleLayer().dot(params.b);
		}();
		double term = this->nodes.v(vindex) * exp(b_dot_v);

		for (int j = 0; j < hSize; j++) {
			term *= sum_h_j(j);
		}

		value += term;

		// debug
		if (isinf(value) || isnan(value)) {
			volatile auto debug_value = value;
			throw;
		}
	}

	value = value / z;
	return value;
}

// 隠れ変数の期待値, E[h_j]
double GeneralizedRBM::expectedValueHid(int hindex) {
	auto z = getNormalConstant();  // 分配関数
	return expectedValueHid(hindex, z);
}

// 隠れ変数の期待値, E[h_j]
double GeneralizedRBM::expectedValueHid(int hindex, double normalize_constant) {
	StateCounter<std::vector<int>> sc(std::vector<int>(vSize, 2));  // 可視変数Vの状態カウンター
	int v_state_map[] = { 0, 1 };  // 可視変数の状態->値変換写像

	auto & z = normalize_constant;  // 分配関数

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


	double value = 0.0;
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

	// debug
	if (isinf(value) || isnan(value)) {
		volatile auto debug_value = value;
		throw;
	}

	value = value / z;
	return value;
}


// 可視変数と隠れ変数の期待値, E[v_i h_j]
double GeneralizedRBM::expectedValueVisHid(int vindex, int hindex) {
	auto z = getNormalConstant();  // 分配関数
	return expectedValueVisHid(vindex, hindex, z);
}

// 可視変数と隠れ変数の期待値, E[v_i h_j]
double GeneralizedRBM::expectedValueVisHid(int vindex, int hindex, double normalize_constant) {
	StateCounter<std::vector<int>> sc(std::vector<int>(vSize, 2));  // 可視変数Vの状態カウンター
	int v_state_map[] = { 0, 1 };  // 可視変数の状態->値変換写像

	auto & z = normalize_constant;  // 分配関数

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

	double value = 0.0;
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

	// debug
	if (isinf(value) || isnan(value)) {
		volatile auto debug_value = value;
		throw;
	}

	value = value / z;
	return value;
}

bool GeneralizedRBM::isRealHiddenValue() {
	return realFlag;
}
