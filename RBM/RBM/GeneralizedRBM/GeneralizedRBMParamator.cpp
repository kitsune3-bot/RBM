#include "GeneralizedRBMParamator.h"
#include <iostream>


GeneralizedRBMParamator::GeneralizedRBMParamator(size_t v_size, size_t h_size) {
	vSize = v_size;
	hSize = h_size;

	initParams();
}

void GeneralizedRBMParamator::initParams() {
	b.resize(vSize);
	b.setConstant(0.0);
	c.resize(hSize);
	c.setConstant(0.0);
	w.resize(vSize, hSize);
	w.setConstant(0.0);
}

void GeneralizedRBMParamator::initParamsRandom(double range_min, double range_max) {

	std::random_device rd;
	std::mt19937 mt(rd());
	this->initParamsRandom(range_min, range_max, mt());

}

void GeneralizedRBMParamator::initParamsRandom(double range_min, double range_max, int seed) {
	b.resize(vSize);
	c.resize(hSize);
	w.resize(vSize, hSize);

	std::mt19937 mt(seed);
	std::uniform_real_distribution<double> dist(range_min, range_max);

	for (int i = 0; i < vSize; i++) {
		b(i) = dist(mt);

		for (int j = 0; j < hSize; j++) {
			w(i, j) = dist(mt);
		}
	}

	for (int j = 0; j < hSize; j++) {
		c(j) = dist(mt);
	}
}

void GeneralizedRBMParamator::initParamsRandom(double range_min, double range_max, int seed) {
	b.resize(vSize);
	c.resize(hSize);
	w.resize(vSize, hSize);

	std::mt19937 mt(seed);
	std::uniform_real_distribution<double> dist(range_min, range_max);

	for (int i = 0; i < vSize; i++) {
		b(i) = dist(mt);

		for (int j = 0; j < hSize; j++) {
			w(i, j) = dist(mt);
		}
	}

	for (int j = 0; j < hSize; j++) {
		c(j) = dist(mt);
	}
}


void GeneralizedRBMParamator::initParamsXavier() {
	std::random_device seed_gen;
	std::mt19937 engine(seed_gen());

	this->initParamsXavier(engine());
}

void GeneralizedRBMParamator::initParamsXavier(int seed) {
	b.resize(vSize);
	c.resize(hSize);
	w.resize(vSize, hSize);

	std::mt19937 mt(seed);
	std::uniform_real_distribution<double> dist(-1.0, 1.0);

	auto epsiron = 0.00001;

	for (int i = 0; i < vSize; i++) {
		b(i) = dist(mt) * epsiron;

		for (int j = 0; j < hSize; j++) {
			w(i, j) = dist(mt) * sqrt(3.0 / vSize);
		}
	}

	for (int j = 0; j < hSize; j++) {
		c(j) = dist(mt) * epsiron;
	}

}



// 可視変数の総数を返す
size_t GeneralizedRBMParamator::getVisibleSize() {
	return vSize;
}

// 隠れ変数の総数を返す
size_t GeneralizedRBMParamator::getHiddenSize() {
	return hSize;
}

// 可視変数のバイアスを返す
double GeneralizedRBMParamator::getVisibleBias(int vindex) {
	return b(vindex);
}

// 可視変数のバイアスベクトルを返す
Eigen::VectorXd GeneralizedRBMParamator::getVisibleBiasVector() {
	return b;
}

// 隠れ変数のバイアスを返す
double GeneralizedRBMParamator::getHiddenBias(int hindex) {
	return c(hindex);
}

// 隠れ変数のバイアスベクトルを返す
Eigen::VectorXd GeneralizedRBMParamator::getHiddenBiasVector() {
	return c;
}

// ウェイトパラメータを返す
double GeneralizedRBMParamator::getWeight(int vindex, int hindex) {
	return w(vindex, hindex);
}

// ウェイト行列を返す
Eigen::MatrixXd GeneralizedRBMParamator::getWeightMatrix() {
	return w;
}

// パラメータ情報のシリアライズ
std::string GeneralizedRBMParamator::serialize() {
	nlohmann::json json;
	json["vSize"] = vSize;
	json["hSize"] = hSize;
	json["params"]["b"] = std::vector<double>(this->b.data(), this->b.data() + this->b.cols() * this->b.rows());
	json["params"]["c"] = std::vector<double>(this->c.data(), this->c.data() + this->c.cols() * this->c.rows());
	json["params"]["w"] = std::vector<double>(this->w.data(), this->w.data() + this->w.cols() * this->w.rows());

	return json.dump();
}

// パラメータ情報のデシリアライズ
void GeneralizedRBMParamator::deserialize(std::string js) {
	auto json = nlohmann::json::parse(js);;

	vSize = json["vSize"];
	hSize = json["hSize"];
	std::vector<double> tmp_b(json["params"]["b"].begin(), json["params"]["b"].end());
	std::vector<double> tmp_c(json["params"]["c"].begin(), json["params"]["c"].end());
	std::vector<double> tmp_w(json["params"]["w"].begin(), json["params"]["w"].end());

	this->b = Eigen::Map<Eigen::VectorXd>(tmp_b.data(), vSize);
	this->c = Eigen::Map<Eigen::VectorXd>(tmp_c.data(), hSize);
	this->w = Eigen::Map<Eigen::MatrixXd>(tmp_w.data(), vSize, hSize);
}

void GeneralizedRBMParamator::printParams()
{
	std::cout << "--- b ---" << std::endl;
	std::cout << this->b << std::endl;

	std::cout << "--- c ---" << std::endl;
	std::cout << this->c << std::endl;

	std::cout << "--- w ---" << std::endl;
	std::cout << this->w << std::endl;
}
