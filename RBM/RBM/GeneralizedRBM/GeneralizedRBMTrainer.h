#pragma once
#include "Eigen/Core"
#include "../Trainer.h"
#include "GeneralizedRBM.h"
#include <vector>

template<>
class Trainer<GeneralizedRBM> {
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
		Eigen::VectorXd vBias;
		Eigen::VectorXd hBias;
		Eigen::MatrixXd weight;
	};

	struct RBMExpected {
		Eigen::VectorXd vBias;
		Eigen::VectorXd hBias;
		Eigen::MatrixXd weight;
	};

private:
	Momentum momentum;
	Gradient gradient;
	DataMean dataMean;
	RBMExpected rbmexpected;
	int _trainCount = 0;


public:
	int epoch = 0;
	int batchSize = 1;
	int cdk = 0;
	double learningRate = 0.01;
	double momentumRate = 0.9;

public:
	Trainer() = default;
	Trainer(GeneralizedRBM & rbm);
	~Trainer() = default;

	// モーメンタムベクトル初期化
	void initMomentum(GeneralizedRBM & rbm);

	// 確保済みのモーメンタムベクトルを0初期化
	void initMomentum();

	// 勾配ベクトル初期化
	void initGradient(GeneralizedRBM & rbm);

	// 確保済みの勾配ベクトルを0初期化
	void initGradient();

	// データ平均ベクトルを初期化
	void initDataMean(GeneralizedRBM & rbm);

	// 確保済みデータ平均ベクトルを初期化
	void initDataMean();

	// サンプル平均ベクトルを初期化
	void initRBMExpected(GeneralizedRBM & rbm);

	// 確保済みサンプル平均ベクトルを初期化
	void initRBMExpected();

	// 学習
	void train(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset);

	void trainCD(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset);
	void trainExact(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset);

	// 1回だけ学習
	void trainOnce(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset);

	void trainOnceCD(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset);
	void trainOnceExact(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset);

	// CD計算
	void calcContrastiveDivergence(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes);

	// CD計算
	void calcExact(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes);

	// データ平均の計算
	void calcDataMean(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes);

	// サンプル平均の計算
	void calcRBMExpectedCD(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes);

	// サンプル平均の計算
	void calcRBMExpectedExact(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes);


	// 勾配の計算
	void calcGradient(GeneralizedRBM & rbm, std::vector<int> & data_indexes);

	// モーメンタム更新
	void updateMomentum(GeneralizedRBM & rbm);

	// 勾配更新
	void updateParams(GeneralizedRBM & rbm);

	// 対数尤度関数
	double logLikeliHood(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset);

	// 学習情報出力(JSON)
	std::string trainInfoJson(GeneralizedRBM & rbm);

	// 学習情報から学習(JSON)
	void trainFromTrainInfo(GeneralizedRBM & rbm, std::string json);
};

inline Trainer<GeneralizedRBM>::Trainer(GeneralizedRBM & rbm) {
	initMomentum(rbm);
	initGradient(rbm);
	initDataMean(rbm);
	initRBMExpected(rbm);
}

inline void Trainer<GeneralizedRBM>::initMomentum(GeneralizedRBM & rbm) {
	momentum.vBias.setConstant(rbm.getVisibleSize(), 0.0);
	momentum.hBias.setConstant(rbm.getHiddenSize(), 0.0);
	momentum.weight.setConstant(rbm.getVisibleSize(), rbm.getHiddenSize(), 0.0);
}

inline void Trainer<GeneralizedRBM>::initMomentum() {
	momentum.vBias.setConstant(0.0);
	momentum.hBias.setConstant(0.0);
	momentum.weight.setConstant(0.0);
}

inline void Trainer<GeneralizedRBM>::initGradient(GeneralizedRBM & rbm) {
	gradient.vBias.setConstant(rbm.getVisibleSize(), 0.0);
	gradient.hBias.setConstant(rbm.getHiddenSize(), 0.0);
	gradient.weight.setConstant(rbm.getVisibleSize(), rbm.getHiddenSize(), 0.0);
}

inline void Trainer<GeneralizedRBM>::initGradient() {
	gradient.vBias.setConstant(0.0);
	gradient.hBias.setConstant(0.0);
	gradient.weight.setConstant(0.0);
}

inline void Trainer<GeneralizedRBM>::initDataMean(GeneralizedRBM & rbm) {
	dataMean.vBias.setConstant(rbm.getVisibleSize(), 0.0);
	dataMean.hBias.setConstant(rbm.getHiddenSize(), 0.0);
	dataMean.weight.setConstant(rbm.getVisibleSize(), rbm.getHiddenSize(), 0.0);
}

inline void Trainer<GeneralizedRBM>::initDataMean() {
	dataMean.vBias.setConstant(0.0);
	dataMean.hBias.setConstant(0.0);
	dataMean.weight.setConstant(0.0);
}

inline void Trainer<GeneralizedRBM>::initRBMExpected(GeneralizedRBM & rbm) {
	rbmexpected.vBias.setConstant(rbm.getVisibleSize(), 0.0);
	rbmexpected.hBias.setConstant(rbm.getHiddenSize(), 0.0);
	rbmexpected.weight.setConstant(rbm.getVisibleSize(), rbm.getHiddenSize(), 0.0);
}

inline void Trainer<GeneralizedRBM>::initRBMExpected() {
	rbmexpected.vBias.setConstant(0.0);
	rbmexpected.hBias.setConstant(0.0);
	rbmexpected.weight.setConstant(0.0);
}

inline void Trainer<GeneralizedRBM>::train(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset) {
	for (int e = 0; e < epoch; e++) {
		trainOnce(rbm, dataset);
	}
}

inline void Trainer<GeneralizedRBM>::trainCD(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset) {
	for (int e = 0; e < epoch; e++) {
		trainOnceCD(rbm, dataset);
	}
}

inline void Trainer<GeneralizedRBM>::trainExact(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset) {
	for (int e = 0; e < epoch; e++) {
		trainOnceExact(rbm, dataset);
	}
}

// FIXME: CDとExactをフラグで切り分けられるように
// 1回だけ学習
inline void Trainer<GeneralizedRBM>::trainOnce(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset) {
	// 勾配初期化
	initGradient();

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

inline void Trainer<GeneralizedRBM>::trainOnceCD(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset) {
	// 勾配初期化
	initGradient();

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

inline void Trainer<GeneralizedRBM>::trainOnceExact(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset) {
	// 勾配初期化
	initGradient();

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
	calcExact(rbm, dataset, minibatch_indexes);

	// モーメンタムの更新
	updateMomentum(rbm);

	// 勾配の更新
	updateParams(rbm);

	// Trainer情報更新
	_trainCount++;
}

inline void Trainer<GeneralizedRBM>::calcContrastiveDivergence(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) {
	// データ平均の計算
	calcDataMean(rbm, dataset, data_indexes);

	// サンプル平均の計算(CD)
	calcRBMExpectedCD(rbm, dataset, data_indexes);

	// 勾配計算
	calcGradient(rbm, data_indexes);
}

inline void Trainer<GeneralizedRBM>::calcExact(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) {
	// データ平均の計算
	calcDataMean(rbm, dataset, data_indexes);

	// サンプル平均の計算(CD)
	calcRBMExpectedExact(rbm, dataset, data_indexes);

	// 勾配計算
	calcGradient(rbm, data_indexes);
}


inline void Trainer<GeneralizedRBM>::calcDataMean(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) {
	// 0埋め初期化
	initDataMean();

	for (auto & n : data_indexes) {
		auto & data = dataset[n];
		Eigen::VectorXd vect = Eigen::Map<Eigen::VectorXd>(data.data(), data.size());
		rbm.nodes.v = vect;

		for (int i = 0; i < rbm.getVisibleSize(); i++) {
			dataMean.vBias(i) += vect(i);

			for (int j = 0; j < rbm.getHiddenSize(); j++) {
				dataMean.weight(i, j) += vect(i) * rbm.actHidJ(j);
			}
		}

		for (int j = 0; j < rbm.getHiddenSize(); j++) {
			dataMean.hBias(j) += rbm.actHidJ(j);
		}
	}

	dataMean.vBias /= static_cast<double>(data_indexes.size());
	dataMean.hBias /= static_cast<double>(data_indexes.size());
	dataMean.weight /= static_cast<double>(data_indexes.size());
}

inline void Trainer<GeneralizedRBM>::calcRBMExpectedCD(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) {
	// 0埋め初期化
	initRBMExpected();

	for (auto & n : data_indexes) {
		auto & data = dataset[n];
		Eigen::VectorXd vect = Eigen::Map<Eigen::VectorXd>(data.data(), data.size());

		// GeneralizedRBMの初期値設定
		rbm.nodes.v = vect;

		for (int j = 0; j < rbm.getHiddenSize(); j++) {
			rbm.nodes.h(j) = rbm.actHidJ(j);
		}

		// CD-K
		Sampler<GeneralizedRBM> sampler;
		for (int k = 0; k < cdk; k++) {
			sampler.updateByBlockedGibbsSamplingVisible(rbm);
			sampler.updateByBlockedGibbsSamplingHidden(rbm);
		}

		// 結果を格納
		rbmexpected.vBias += rbm.nodes.v;
		rbmexpected.hBias += rbm.nodes.h;

		for (int i = 0; i < rbm.getVisibleSize(); i++) {
			for (int j = 0; j < rbm.getHiddenSize(); j++) {
				rbmexpected.weight(i, j) += rbm.nodes.v(i) * rbm.nodes.h(j);
			}
		}
	}

	rbmexpected.vBias /= static_cast<double>(data_indexes.size());
	rbmexpected.hBias /= static_cast<double>(data_indexes.size());
	rbmexpected.weight /= static_cast<double>(data_indexes.size());
}

inline void Trainer<GeneralizedRBM>::calcRBMExpectedExact(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) {
	// 0埋め初期化
	initRBMExpected();

	// FIXME: 分配関数の計算がネックになる。 規格化定数使いまわしで高速化可能
	for (int i = 0; i < rbm.getVisibleSize(); i++) {
		rbmexpected.vBias(i) = rbm.expectedValueVis(i);

		for (int j = 0; j < rbm.getHiddenSize(); j++) {
			rbmexpected.weight(i, j) = rbm.expectedValueVisHid(i, j);
		}
	}

	// FIXME: 分配関数の計算がネックになる。 規格化定数使いまわしで高速化可能
	for (int j = 0; j < rbm.getHiddenSize(); j++) {
		rbmexpected.hBias(j) = rbm.expectedValueHid(j);
	}
}


// 勾配の計算
inline void Trainer<GeneralizedRBM>::calcGradient(GeneralizedRBM & rbm, std::vector<int> & data_indexes) {
	// 勾配ベクトルリセット
	initGradient();

	for (int i = 0; i < rbm.getVisibleSize(); i++) {
		gradient.vBias(i) = dataMean.vBias(i) - rbmexpected.vBias(i);

		for (int j = 0; j < rbm.getHiddenSize(); j++) {
			gradient.weight(i, j) = dataMean.weight(i, j) - rbmexpected.weight(i, j);
		}
	}

	for (int j = 0; j < rbm.getHiddenSize(); j++) {
		gradient.hBias(j) = dataMean.hBias(j) - rbmexpected.hBias(j);
	}
}

inline void Trainer<GeneralizedRBM>::updateMomentum(GeneralizedRBM & rbm) {
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
inline void Trainer<GeneralizedRBM>::updateParams(GeneralizedRBM & rbm) {
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


// 対数尤度関数
inline double Trainer<GeneralizedRBM>::logLikeliHood(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset) {
	double value = 0.0;

	for (auto & data : dataset) {
		// FIXME: 分配関数使いまわしで高速化可能
		auto prob = rbm.probVis(data);
		value += log(prob);
	}

	return value;
}

// 学習情報出力(JSON)
inline std::string Trainer<GeneralizedRBM>::trainInfoJson(GeneralizedRBM & rbm) {
	auto js = nlohmann::json();
	js["rbm"] = nlohmann::json::parse(rbm.params.serialize());
	js["trainCount"] = _trainCount;
	js["learningRate"] = learningRate;
	js["momentumRate"] = momentumRate;
	js["cdk"] = cdk;
	js["divSize"] = rbm.getHiddenDivSize();
	js["realFlag"] = rbm.isRealHiddenValue();

	return js.dump();
}

inline void Trainer<GeneralizedRBM>::trainFromTrainInfo(GeneralizedRBM & rbm, std::string json) {
	auto js = nlohmann::json::parse(json);
	rbm.params.deserialize(js["rbm"].dump());
	_trainCount = js["trainCount"];
	learningRate = js["learningRate"];
	momentumRate = js["momentumRate"];
	cdk = js["cdk"];
	rbm.setHiddenDiveSize(js["divSize"]);
	rbm.setRealHiddenValue(js["realFlag"]);
}
