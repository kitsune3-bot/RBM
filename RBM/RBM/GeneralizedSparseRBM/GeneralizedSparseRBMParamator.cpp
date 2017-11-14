#include "GeneralizedSparseRBMParamator.h"


GeneralizedSparseRBMParamator::GeneralizedSparseRBMParamator(size_t v_size, size_t h_size) {
	vSize = v_size;
	hSize = h_size;

	initParams();
}

void GeneralizedSparseRBMParamator::initParams() {
	b.resize(vSize);
	b.setConstant(0.0);
	c.resize(hSize);
	c.setConstant(0.0);
	w.resize(vSize, hSize);
	w.setConstant(0.0);
	sparse.resize(hSize);
	sparse.setConstant(0.0);
}

void GeneralizedSparseRBMParamator::initParamsRandom(double range_min, double range_max) {
	b.resize(vSize);
	c.resize(hSize);
	w.resize(vSize, hSize);
	sparse.resize(hSize);

	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<double> dist(range_min, range_max);

	for (int i = 0; i < vSize; i++) {
		b(i) = dist(mt);

		for (int j = 0; j < hSize; j++) {
			w(i, j) = dist(mt);
		}
	}

	for (int j = 0; j < hSize; j++) {
		c(j) = dist(mt);
		sparse(j) = dist(mt);
	}
}

void GeneralizedSparseRBMParamator::initParamsXavier()
{
	b.resize(vSize);
	c.resize(hSize);
	w.resize(vSize, hSize);
	sparse.resize(hSize);

	b.setRandom() *= 0.00001;
	c.setRandom() *= 0.00001;
	w.setRandom() /= sqrt((vSize + hSize) / 2.0);
	sparse.setConstant(4.0);
}


// 可視変数の総数を返す
size_t GeneralizedSparseRBMParamator::getVisibleSize() {
	return vSize;
}

// 隠れ変数の総数を返す
size_t GeneralizedSparseRBMParamator::getHiddenSize() {
	return hSize;
}

// 可視変数のバイアスを返す
double GeneralizedSparseRBMParamator::getVisibleBias(int vindex) {
	return b(vindex);
}

// 可視変数のバイアスベクトルを返す
Eigen::VectorXd GeneralizedSparseRBMParamator::getVisibleBiasVector() {
	return b;
}

// 隠れ変数のバイアスを返す
double GeneralizedSparseRBMParamator::getHiddenBias(int hindex) {
	return c(hindex);
}

// 隠れ変数のバイアスベクトルを返す
Eigen::VectorXd GeneralizedSparseRBMParamator::getHiddenBiasVector() {
	return c;
}

// ウェイトパラメータを返す
double GeneralizedSparseRBMParamator::getWeight(int vindex, int hindex) {
	return w(vindex, hindex);
}

// ウェイト行列を返す
Eigen::MatrixXd GeneralizedSparseRBMParamator::getWeightMatrix() {
	return w;
}

// パラメータ情報のシリアライズ
std::string GeneralizedSparseRBMParamator::serialize() {
	nlohmann::json json;
	json["vSize"] = vSize;
	json["hSize"] = hSize;
	json["params"]["b"] = std::vector<double>(this->b.data(), this->b.data() + this->b.cols() * this->b.rows());
	json["params"]["c"] = std::vector<double>(this->c.data(), this->c.data() + this->c.cols() * this->c.rows());
	json["params"]["w"] = std::vector<double>(this->w.data(), this->w.data() + this->w.cols() * this->w.rows());
	json["params"]["sparse"] = std::vector<double>(this->sparse.data(), this->sparse.data() + this->sparse.cols() * this->sparse.rows());

	return json.dump();
}

// パラメータ情報のデシリアライズ
void GeneralizedSparseRBMParamator::deserialize(std::string js) {
	auto json = nlohmann::json::parse(js);;

	vSize = json["vSize"];
	hSize = json["hSize"];
	std::vector<double> tmp_b(json["params"]["b"].begin(), json["params"]["b"].end());
	std::vector<double> tmp_c(json["params"]["c"].begin(), json["params"]["c"].end());
	std::vector<double> tmp_w(json["params"]["w"].begin(), json["params"]["w"].end());
	std::vector<double> tmp_sparse(json["params"]["sparse"].begin(), json["params"]["sparse"].end());

	this->b = Eigen::Map<Eigen::VectorXd>(tmp_b.data(), vSize);
	this->c = Eigen::Map<Eigen::VectorXd>(tmp_c.data(), hSize);
	this->w = Eigen::Map<Eigen::MatrixXd>(tmp_w.data(), vSize, hSize);
	this->sparse = Eigen::Map<Eigen::VectorXd>(tmp_sparse.data(), hSize);
}

// 隠れ変数のスパースパラメータを返す
double GeneralizedSparseRBMParamator::getHiddenSparse(int hindex) {
	return sparse(hindex);
}

// 隠れ変数のスパースパラメータベクトルを返す
Eigen::VectorXd GeneralizedSparseRBMParamator::getHiddenSparseVector() {
	return sparse;
}
