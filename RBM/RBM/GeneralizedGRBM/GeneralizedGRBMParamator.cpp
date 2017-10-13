#include "GeneralizedGRBMParamator.h"


// 可視変数の総数を返す
size_t GeneralizedGRBMParamator::getVisibleSize() {
	return vSize;
}

// 隠れ変数の総数を返す
size_t GeneralizedGRBMParamator::getHiddenSize() {
	return hSize;
}

// 可視変数のバイアスを返す
double GeneralizedGRBMParamator::getVisibleBias(int vindex) {
	return b(vindex);
}

// 可視変数のバイアスベクトルを返す
Eigen::VectorXd GeneralizedGRBMParamator::getVisibleBiasVector() {
	return b;
}

// 可視変数の逆分散を返す
double GeneralizedGRBMParamator::getVisibleLambda(int vindex) {
	return lambda(vindex);
}

// 可視変数の逆分散ベクトルを返す
Eigen::VectorXd GeneralizedGRBMParamator::getVisibleLambdaVector() {
	return lambda;
}

// 隠れ変数のバイアスを返す
double GeneralizedGRBMParamator::getHiddenBias(int hindex) {
	return c(hindex);
}

// 隠れ変数のバイアスベクトルを返す
Eigen::VectorXd GeneralizedGRBMParamator::getHiddenBiasVector() {
	return c;
}

// ウェイトパラメータを返す
double GeneralizedGRBMParamator::getWeight(int vindex, int hindex) {
	return w(vindex, hindex);
}

// ウェイト行列を返す
Eigen::MatrixXd GeneralizedGRBMParamator::getWeightMatrix() {
	return w;
}

// パラメータ情報のシリアライズ
std::string GeneralizedGRBMParamator::serialize() {
	nlohmann::json json;
	json["vSize"] = vSize;
	json["hSize"] = hSize;
	json["params"]["b"] = std::vector<double>(this->b.data(), this->b.data() + this->b.cols() * this->b.rows());
	json["params"]["c"] = std::vector<double>(this->c.data(), this->c.data() + this->c.cols() * this->c.rows());
	json["params"]["w"] = std::vector<double>(this->w.data(), this->w.data() + this->w.cols() * this->w.rows());
	json["params"]["lambda"] = std::vector<double>(this->lambda.data(), this->lambda.data() + this->lambda.cols() * this->lambda.rows());

	return json.dump();
}

// パラメータ情報のデシリアライズ
void GeneralizedGRBMParamator::deserialize(std::string js) {
	auto json = nlohmann::json::parse(js);;

	vSize = json["vSize"];
	hSize = json["hSize"];
	std::vector<double> tmp_b(json["params"]["b"].begin(), json["params"]["b"].end());
	std::vector<double> tmp_c(json["params"]["c"].begin(), json["params"]["c"].end());
	std::vector<double> tmp_w(json["params"]["w"].begin(), json["params"]["w"].end());
	std::vector<double> tmp_lambda(json["params"]["lambda"].begin(), json["params"]["lambda"].end());

	this->b = Eigen::Map<Eigen::VectorXd>(tmp_b.data(), vSize);
	this->c = Eigen::Map<Eigen::VectorXd>(tmp_c.data(), hSize);
	this->w = Eigen::Map<Eigen::MatrixXd>(tmp_w.data(), vSize, hSize);
	this->lambda = Eigen::Map<Eigen::VectorXd>(tmp_lambda.data(), vSize);
}


GeneralizedGRBMParamator::GeneralizedGRBMParamator(size_t v_size, size_t h_size) {
	vSize = v_size;
	hSize = h_size;

	initParams();
}

void GeneralizedGRBMParamator::initParams() {
	b.resize(vSize);
	b.setConstant(0.0);
	lambda.resize(vSize);
	lambda.setConstant(10.0);  // 逆分散は非負制約がある, 逆分散 = 10 -> 分散0.1
	c.resize(hSize);
	c.setConstant(0.0);
	w.resize(vSize, hSize);
	w.setConstant(0.0);
}

void GeneralizedGRBMParamator::initParamsRandom(double range_min, double range_max) {
	b.resize(vSize);
	c.resize(hSize);
	w.resize(vSize, hSize);

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
	}

	// XXX: 逆分散は乱数使うと危ない
	lambda.resize(vSize);
	lambda.setConstant(10.0);  // 逆分散は非負制約がある, 逆分散 = 10 -> 分散0.1
}
