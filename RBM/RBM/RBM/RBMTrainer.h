#pragma once
#include "../Trainer.h"
#include "RBM.h"
#include "RBMSampler.h"
#include "Eigen/Core"
#include <vector>
#include "json.hpp"
#include <vector>
#include <numeric>
#include <random>

template<class OPTIMIZERTYPE>
class Trainer<RBM, OPTIMIZERTYPE> {
	struct Momentum {
		Eigen::VectorXd vBias;
		Eigen::VectorXd hBias;
		Eigen::MatrixXd weight;
	};

	struct Gradient {
		Eigen::VectorXd vBias;
		Eigen::VectorXd hBias;
		Eigen::MatrixXd weight;
	};

	struct DataMean {
		Eigen::VectorXd visible;
		Eigen::VectorXd hidden;
	};

	struct RBMExpected {
		Eigen::VectorXd visible;
		Eigen::VectorXd hidden;
	};

private:
	Momentum momentum;
	Gradient gradient;
	DataMean dataMean;
	RBMExpected sampleMean;
	int _trainCount = 0;


public:
	int epoch = 0;
	int batchSize = 1;
	int cdk = 0;
	double learningRate = 0.01;
	double momentumRate = 0.9;

public:
	Trainer() = default;
	Trainer(RBM & rbm);
	~Trainer() = default;

	// モーメンタムベクトル初期化
	void initMomentum(RBM & rbm);

	// 確保済みのモーメンタムベクトルを0初期化
	void initMomentum();

	// 勾配ベクトル初期化
	void initGradient(RBM & rbm);

	// 確保済みの勾配ベクトルを0初期化
	void initGradient();

	// データ平均ベクトルを初期化
	void initDataMean(RBM & rbm);

	// 確保済みデータ平均ベクトルを初期化
	void initDataMean();

	// サンプル平均ベクトルを初期化
	void initRBMExpected(RBM & rbm);

	// 確保済みサンプル平均ベクトルを初期化
	void initRBMExpected();

	// 学習
	void train(RBM & rbm, std::vector<std::vector<double>> & dataset);

	// 1回だけ学習
	void trainOnce(RBM & rbm, std::vector<std::vector<double>> & dataset);

	// CD計算
	void calcContrastiveDivergence(RBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes);

	// データ平均の計算
	void calcDataMean(RBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes);

	// サンプル平均の計算
	void calcRBMExpectedCD(RBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes);

	// 勾配の計算
	void calcGradient(RBM & rbm, std::vector<int> & data_indexes);

	// モーメンタム更新
	void updateMomentum(RBM & rbm);

	// 勾配更新
	void updateParams(RBM & rbm);

	// 学習情報出力(JSON)
	std::string trainInfoJson(RBM & rbm);

	// 学習情報から学習(JSON)
	void trainFromTrainInfo(RBM & rbm, std::string json);
};

template<class OPTIMIZERTYPE>
inline Trainer<RBM, OPTIMIZERTYPE>::Trainer(RBM & rbm) {
	initMomentum(rbm);
	initGradient(rbm);
	initDataMean(rbm);
	initRBMExpected(rbm);
}

template<class OPTIMIZERTYPE>
inline void Trainer<RBM, OPTIMIZERTYPE>::initMomentum(RBM & rbm) {
	momentum.vBias.setConstant(rbm.getVisibleSize(), 0.0);
	momentum.hBias.setConstant(rbm.getHiddenSize(), 0.0);
	momentum.weight.setConstant(rbm.getVisibleSize(), rbm.getHiddenSize(), 0.0);
}

template<class OPTIMIZERTYPE>
inline void Trainer<RBM, OPTIMIZERTYPE>::initMomentum() {
	momentum.vBias.setConstant(0.0);
	momentum.hBias.setConstant(0.0);
	momentum.weight.setConstant(0.0);
}

template<class OPTIMIZERTYPE>
inline void Trainer<RBM, OPTIMIZERTYPE>::initGradient(RBM & rbm) {
	gradient.vBias.setConstant(rbm.getVisibleSize(), 0.0);
	gradient.hBias.setConstant(rbm.getHiddenSize(), 0.0);
	gradient.weight.setConstant(rbm.getVisibleSize(), rbm.getHiddenSize(), 0.0);
}

template<class OPTIMIZERTYPE>
inline void Trainer<RBM, OPTIMIZERTYPE>::initGradient() {
	gradient.vBias.setConstant(0.0);
	gradient.hBias.setConstant(0.0);
	gradient.weight.setConstant(0.0);
}

template<class OPTIMIZERTYPE>
inline void Trainer<RBM, OPTIMIZERTYPE>::initDataMean(RBM & rbm) {
	dataMean.visible.setConstant(rbm.getVisibleSize(), 0.0);
	dataMean.hidden.setConstant(rbm.getHiddenSize(), 0.0);
}

template<class OPTIMIZERTYPE>
inline void Trainer<RBM, OPTIMIZERTYPE>::initDataMean() {
	dataMean.visible.setConstant(0.0);
	dataMean.hidden.setConstant(0.0);
}

template<class OPTIMIZERTYPE>
inline void Trainer<RBM, OPTIMIZERTYPE>::initRBMExpected(RBM & rbm) {
	sampleMean.visible.setConstant(rbm.getVisibleSize(), 0.0);
	sampleMean.hidden.setConstant(rbm.getHiddenSize(), 0.0);
}

template<class OPTIMIZERTYPE>
inline void Trainer<RBM, OPTIMIZERTYPE>::initRBMExpected() {
	sampleMean.visible.setConstant(0.0);
	sampleMean.hidden.setConstant(0.0);
}

template<class OPTIMIZERTYPE>
inline void Trainer<RBM, OPTIMIZERTYPE>::train(RBM & rbm, std::vector<std::vector<double>> & dataset) {
	for (int e = 0; e < epoch; e++) {
		trainOnce(rbm, dataset);
	}
}


// 1回だけ学習
template<class OPTIMIZERTYPE>
inline void Trainer<RBM, OPTIMIZERTYPE>::trainOnce(RBM & rbm, std::vector<std::vector<double>> & dataset) {

	// データインデックス集合
	std::vector<int> data_indexes(dataset.size());

	// ミニバッチ学習のためにデータインデックスをシャッフルする
	std::iota(data_indexes.begin(), data_indexes.end(), 0);
	std::shuffle(data_indexes.begin(), data_indexes.end(), std::mt19937());

	// ミニバッチ
	// バッチサイズの確認
	int batch_size = this->batchSize < dataset.size() ? dataset.size() : this->batchSize;

	// ミニバッチ学習に使うデータのインデックス集合
	std::vector<int> minibatch_indexes(batch_size);
	std::copy(data_indexes.begin(), data_indexes.begin() + batch_size, minibatch_indexes.begin());

	// Contrastive Divergence
	calcContrastiveDivergence(rbm, dataset, minibatch_indexes);

	// モーメンタムの更新
	updateMomentum(rbm);

	// 勾配の更新
	updateParams(rbm);

	// Trainer情報更新
	_trainCount++;
}

template<class OPTIMIZERTYPE>
inline void Trainer<RBM, OPTIMIZERTYPE>::calcContrastiveDivergence(RBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) {
	// データ平均の計算
	calcDataMean(rbm, dataset, data_indexes);

	// サンプル平均の計算(CD)
	calcRBMExpectedCD(rbm, dataset, data_indexes);

	// 勾配計算
	calcGradient(rbm, data_indexes);
}

template<class OPTIMIZERTYPE>
inline void Trainer<RBM, OPTIMIZERTYPE>::calcDataMean(RBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) {
	// 0埋め初期化
	initDataMean();

	for (auto & n : data_indexes) {
		auto & data = dataset[n];
		Eigen::VectorXd vect = Eigen::Map<Eigen::VectorXd>(data.data(), data.size());
		rbm.nodes.v = vect;

		dataMean.visible += vect;

		for (int j = 0; j < rbm.getHiddenSize(); j++) {
			dataMean.hidden(j) += rbm.actHidJ(j);
		}
	}

	dataMean.visible /= static_cast<double>(data_indexes.size());
	dataMean.hidden /= static_cast<double>(data_indexes.size());
}

template<class OPTIMIZERTYPE>
inline void Trainer<RBM, OPTIMIZERTYPE>::calcRBMExpectedCD(RBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) {
	// 0埋め初期化
	initRBMExpected();

	for (auto & n : data_indexes) {
		auto & data = dataset[n];
		Eigen::VectorXd vect = Eigen::Map<Eigen::VectorXd>(data.data(), data.size());

		// RBMの初期値設定
		rbm.nodes.v = vect;

		for (int j = 0; j < rbm.getHiddenSize(); j++) {
			rbm.nodes.h(j) = rbm.actHidJ(j);
		}

		// CD-K
		Sampler<RBM, OPTIMIZERTYPE> sampler;
		for (int k = 0; k < cdk; k++) {
			sampler.updateByBlockedGibbsSamplingVisible(rbm);
			sampler.updateByBlockedGibbsSamplingHidden(rbm);
		}

		// 結果を格納
		sampleMean.visible += rbm.nodes.v;
		sampleMean.hidden += rbm.nodes.h;
	}

	sampleMean.visible /= static_cast<double>(data_indexes.size());
	sampleMean.hidden /= static_cast<double>(data_indexes.size());
}

// 勾配の計算
template<class OPTIMIZERTYPE>
inline void Trainer<RBM, OPTIMIZERTYPE>::calcGradient(RBM & rbm, std::vector<int> & data_indexes) {
	// 勾配ベクトルリセット
	initGradient();

	for (int i = 0; i < rbm.getVisibleSize(); i++) {
		gradient.vBias(i) = dataMean.visible(i) - sampleMean.visible(i);

		for (int j = 0; j < rbm.getHiddenSize(); j++) {
			gradient.weight(i, j) = dataMean.visible(i) * dataMean.hidden(j) - sampleMean.visible(i) * sampleMean.hidden(j);
		}
	}

	for (int j = 0; j < rbm.getHiddenSize(); j++) {
		gradient.hBias(j) = dataMean.hidden(j) - sampleMean.hidden(j);
	}
}

template<class OPTIMIZERTYPE>
inline void Trainer<RBM, OPTIMIZERTYPE>::updateMomentum(RBM & rbm) {
	for (int i = 0; i < rbm.getVisibleSize(); i++) {
		momentum.vBias(i) = momentumRate * momentum.vBias(i) + learningRate * gradient.vBias(i);

		for (int j = 0; j < rbm.getHiddenSize(); j++) {
			momentum.weight(i, j) = momentumRate * momentum.weight(i, j) + learningRate * gradient.weight(i, j);
		}
	}

	for (int j = 0; j < rbm.getHiddenSize(); j++) {
		momentum.hBias(j) = momentumRate * momentum.hBias(j) + learningRate * gradient.hBias(j);
	}
}

// パラメータの更新
template<class OPTIMIZERTYPE>
inline void Trainer<RBM, OPTIMIZERTYPE>::updateParams(RBM & rbm) {
	for (int i = 0; i < rbm.getVisibleSize(); i++) {
		rbm.params.b(i) += momentum.vBias(i);

		for (int j = 0; j < rbm.getHiddenSize(); j++) {
			rbm.params.w(i, j) += momentum.weight(i, j);
		}
	}

	for (int j = 0; j < rbm.getHiddenSize(); j++) {
		rbm.params.c(j) += momentum.hBias(j);
	}
}

// 学習情報出力(JSON)
template<class OPTIMIZERTYPE>
inline std::string Trainer<RBM, OPTIMIZERTYPE>::trainInfoJson(RBM & rbm) {
	auto js = nlohmann::json();
	js["rbm"] = nlohmann::json::parse(rbm.params.serialize());
	js["trainCount"] = _trainCount;
	js["learningRate"] = learningRate;
	js["momentumRate"] = momentumRate;
	js["cdk"] = cdk;

	return js.dump();
}

template<class OPTIMIZERTYPE>
inline void Trainer<RBM, OPTIMIZERTYPE>::trainFromTrainInfo(RBM & rbm, std::string json) {
	auto js = nlohmann::json::parse(json);
	rbm.params.deserialize(js["rbm"].dump());
	_trainCount = js["trainCount"];
	learningRate = js["learningRate"];
	momentumRate = js["momentumRate"];
	cdk = js["cdk"];
}

