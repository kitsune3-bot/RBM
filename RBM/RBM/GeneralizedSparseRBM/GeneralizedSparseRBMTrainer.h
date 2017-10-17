#pragma once
#include "Eigen/Core"
#include "../Trainer.h"
#include "GeneralizedSparseRBM.h"
#include "GeneralizedSparseRBMSampler.h"
#include <vector>


template<class OPTIMIZERTYPE>
class Trainer<GeneralizedSparseRBM, OPTIMIZERTYPE>{
	struct Momentum {
		Eigen::VectorXd vBias;
		Eigen::VectorXd hBias;
		Eigen::MatrixXd weight;
		Eigen::VectorXd hSparse;
	};

	struct Gradient {
		Eigen::VectorXd vBias;
		Eigen::VectorXd hBias;
		Eigen::MatrixXd weight;
		Eigen::VectorXd hSparse;
	};

	struct DataMean {
		Eigen::VectorXd vBias;
		Eigen::VectorXd hBias;
		Eigen::MatrixXd weight;
		Eigen::VectorXd hSparse;
	};

	struct RBMExpected {
		Eigen::VectorXd vBias;
		Eigen::VectorXd hBias;
		Eigen::MatrixXd weight;
		Eigen::VectorXd hSparse;
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
	Trainer(GeneralizedSparseRBM & rbm);
	~Trainer() = default;

	// モーメンタムベクトル初期化
	void initMomentum(GeneralizedSparseRBM & rbm);

	// 確保済みのモーメンタムベクトルを0初期化
	void initMomentum();

	// 勾配ベクトル初期化
	void initGradient(GeneralizedSparseRBM & rbm);

	// 確保済みの勾配ベクトルを0初期化
	void initGradient();

	// データ平均ベクトルを初期化
	void initDataMean(GeneralizedSparseRBM & rbm);

	// 確保済みデータ平均ベクトルを初期化
	void initDataMean();

	// サンプル平均ベクトルを初期化
	void initRBMExpected(GeneralizedSparseRBM & rbm);

	// 確保済みサンプル平均ベクトルを初期化
	void initRBMExpected();

	// 学習
	void train(GeneralizedSparseRBM & rbm, std::vector<std::vector<double>> & dataset);

	void trainCD(GeneralizedSparseRBM & rbm, std::vector<std::vector<double>> & dataset);
	void trainExact(GeneralizedSparseRBM & rbm, std::vector<std::vector<double>> & dataset);


	// 1回だけ学習
	void trainOnce(GeneralizedSparseRBM & rbm, std::vector<std::vector<double>> & dataset);

	void trainOnceCD(GeneralizedSparseRBM & rbm, std::vector<std::vector<double>> & dataset);
	void trainOnceExact(GeneralizedSparseRBM & rbm, std::vector<std::vector<double>> & dataset);


	// CD計算
	void calcContrastiveDivergence(GeneralizedSparseRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes);

	// CD計算
	void calcExact(GeneralizedSparseRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes);

	// データ平均の計算
	void calcDataMean(GeneralizedSparseRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes);

	// サンプル平均の計算
	void calcRBMExpectedCD(GeneralizedSparseRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes);

	// サンプル平均の計算
	void calcRBMExpectedExact(GeneralizedSparseRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes);


	// 勾配の計算
	void calcGradient(GeneralizedSparseRBM & rbm, std::vector<int> & data_indexes);

	// モーメンタム更新
	void updateMomentum(GeneralizedSparseRBM & rbm);

	// 勾配更新
	void updateParams(GeneralizedSparseRBM & rbm);

	// 対数尤度関数
	double logLikeliHood(GeneralizedSparseRBM & rbm, std::vector<std::vector<double>> & dataset);

	// 学習情報出力(JSON)
	std::string trainInfoJson(GeneralizedSparseRBM & rbm);

	// 学習情報から学習(JSON)
	void trainFromTrainInfo(GeneralizedSparseRBM & rbm, std::string json);
};

template<class OPTIMIZERTYPE>
inline Trainer<GeneralizedSparseRBM, OPTIMIZERTYPE>::Trainer(GeneralizedSparseRBM & rbm) {
	initMomentum(rbm);
	initGradient(rbm);
	initDataMean(rbm);
	initRBMExpected(rbm);
}

template<class OPTIMIZERTYPE>
inline void Trainer<GeneralizedSparseRBM, OPTIMIZERTYPE>::initMomentum(GeneralizedSparseRBM & rbm) {
	momentum.vBias.setConstant(rbm.getVisibleSize(), 0.0);
	momentum.hBias.setConstant(rbm.getHiddenSize(), 0.0);
	momentum.weight.setConstant(rbm.getVisibleSize(), rbm.getHiddenSize(), 0.0);
	momentum.hSparse.setConstant(rbm.getHiddenSize(), 0.0);
}

template<class OPTIMIZERTYPE>
inline void Trainer<GeneralizedSparseRBM, OPTIMIZERTYPE>::initMomentum() {
	momentum.vBias.setConstant(0.0);
	momentum.hBias.setConstant(0.0);
	momentum.weight.setConstant(0.0);
	momentum.hSparse.setConstant(0.0);
}

template<class OPTIMIZERTYPE>
inline void Trainer<GeneralizedSparseRBM, OPTIMIZERTYPE>::initGradient(GeneralizedSparseRBM & rbm) {
	gradient.vBias.setConstant(rbm.getVisibleSize(), 0.0);
	gradient.hBias.setConstant(rbm.getHiddenSize(), 0.0);
	gradient.hSparse.setConstant(rbm.getHiddenSize(), 0.0);
	gradient.weight.setConstant(rbm.getVisibleSize(), rbm.getHiddenSize(), 0.0);
}

template<class OPTIMIZERTYPE>
inline void Trainer<GeneralizedSparseRBM, OPTIMIZERTYPE>::initGradient() {
	gradient.vBias.setConstant(0.0);
	gradient.hBias.setConstant(0.0);
	gradient.hSparse.setConstant(0.0);
	gradient.weight.setConstant(0.0);
}

template<class OPTIMIZERTYPE>
inline void Trainer<GeneralizedSparseRBM, OPTIMIZERTYPE>::initDataMean(GeneralizedSparseRBM & rbm) {
	dataMean.vBias.setConstant(rbm.getVisibleSize(), 0.0);
	dataMean.hBias.setConstant(rbm.getHiddenSize(), 0.0);
	dataMean.weight.setConstant(rbm.getVisibleSize(), rbm.getHiddenSize(), 0.0);
	dataMean.hSparse.setConstant(rbm.getHiddenSize(), 0.0);
}

template<class OPTIMIZERTYPE>
inline void Trainer<GeneralizedSparseRBM, OPTIMIZERTYPE>::initDataMean() {
	dataMean.vBias.setConstant(0.0);
	dataMean.hBias.setConstant(0.0);
	dataMean.weight.setConstant(0.0);
	dataMean.hSparse.setConstant(0.0);
}

template<class OPTIMIZERTYPE>
inline void Trainer<GeneralizedSparseRBM, OPTIMIZERTYPE>::initRBMExpected(GeneralizedSparseRBM & rbm) {
	rbmexpected.vBias.setConstant(rbm.getVisibleSize(), 0.0);
	rbmexpected.hBias.setConstant(rbm.getHiddenSize(), 0.0);
	rbmexpected.weight.setConstant(rbm.getVisibleSize(), rbm.getHiddenSize(), 0.0);
	rbmexpected.hSparse.setConstant(rbm.getHiddenSize(), 0.0);
}

template<class OPTIMIZERTYPE>
inline void Trainer<GeneralizedSparseRBM, OPTIMIZERTYPE>::initRBMExpected() {
	rbmexpected.vBias.setConstant(0.0);
	rbmexpected.hBias.setConstant(0.0);
	rbmexpected.weight.setConstant(0.0);
	rbmexpected.hSparse.setConstant(0.0);
}

template<class OPTIMIZERTYPE>
inline void Trainer<GeneralizedSparseRBM, OPTIMIZERTYPE>::train(GeneralizedSparseRBM & rbm, std::vector<std::vector<double>> & dataset) {
	for (int e = 0; e < epoch; e++) {
		trainOnce(rbm, dataset);
	}
}

template<class OPTIMIZERTYPE>
inline void Trainer<GeneralizedSparseRBM, OPTIMIZERTYPE>::trainCD(GeneralizedSparseRBM & rbm, std::vector<std::vector<double>> & dataset) {
	for (int e = 0; e < epoch; e++) {
		trainOnceCD(rbm, dataset);
	}
}

template<class OPTIMIZERTYPE>
inline void Trainer<GeneralizedSparseRBM, OPTIMIZERTYPE>::trainExact(GeneralizedSparseRBM & rbm, std::vector<std::vector<double>> & dataset) {
	for (int e = 0; e < epoch; e++) {
		trainOnceExact(rbm, dataset);
	}
}


// FIXME: CDとExactをフラグで切り分けられるように
// 1回だけ学習
template<class OPTIMIZERTYPE>
inline void Trainer<GeneralizedSparseRBM, OPTIMIZERTYPE>::trainOnce(GeneralizedSparseRBM & rbm, std::vector<std::vector<double>> & dataset) {
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

template<class OPTIMIZERTYPE>
inline void Trainer<GeneralizedSparseRBM, OPTIMIZERTYPE>::trainOnceCD(GeneralizedSparseRBM & rbm, std::vector<std::vector<double>> & dataset) {
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

template<class OPTIMIZERTYPE>
inline void Trainer<GeneralizedSparseRBM, OPTIMIZERTYPE>::trainOnceExact(GeneralizedSparseRBM & rbm, std::vector<std::vector<double>> & dataset) {
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


template<class OPTIMIZERTYPE>
inline void Trainer<GeneralizedSparseRBM, OPTIMIZERTYPE>::calcContrastiveDivergence(GeneralizedSparseRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) {
	// データ平均の計算
	calcDataMean(rbm, dataset, data_indexes);

	// サンプル平均の計算(CD)
	calcRBMExpectedCD(rbm, dataset, data_indexes);

	// 勾配計算
	calcGradient(rbm, data_indexes);
}

template<class OPTIMIZERTYPE>
inline void Trainer<GeneralizedSparseRBM, OPTIMIZERTYPE>::calcExact(GeneralizedSparseRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) {
	// データ平均の計算
	calcDataMean(rbm, dataset, data_indexes);

	// サンプル平均の計算(CD)
	calcRBMExpectedExact(rbm, dataset, data_indexes);

	// 勾配計算
	calcGradient(rbm, data_indexes);
}


template<class OPTIMIZERTYPE>
inline void Trainer<GeneralizedSparseRBM, OPTIMIZERTYPE>::calcDataMean(GeneralizedSparseRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) {
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
			dataMean.hSparse(j) += rbm.actHidSparseJ(j);
		}
	}

	dataMean.vBias /= static_cast<double>(data_indexes.size());
	dataMean.hBias /= static_cast<double>(data_indexes.size());
	dataMean.weight /= static_cast<double>(data_indexes.size());
	dataMean.hSparse /= static_cast<double>(data_indexes.size());
}

template<class OPTIMIZERTYPE>
inline void Trainer<GeneralizedSparseRBM, OPTIMIZERTYPE>::calcRBMExpectedCD(GeneralizedSparseRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) {
	// 0埋め初期化
	initRBMExpected();

	for (auto & n : data_indexes) {
		auto & data = dataset[n];
		Eigen::VectorXd vect = Eigen::Map<Eigen::VectorXd>(data.data(), data.size());

		// GeneralizedSparseRBMの初期値設定
		rbm.nodes.v = vect;

		for (int j = 0; j < rbm.getHiddenSize(); j++) {
			rbm.nodes.h(j) = rbm.actHidJ(j);
		}

		// CD-K
		Sampler<GeneralizedSparseRBM, OPTIMIZERTYPE> sampler;
		for (int k = 0; k < cdk; k++) {
			sampler.updateByBlockedGibbsSamplingVisible(rbm);
			sampler.updateByBlockedGibbsSamplingHidden(rbm);
		}

		// 結果を格納
		rbmexpected.vBias += rbm.nodes.v;
		rbmexpected.hBias += rbm.nodes.h;

		for (int j = 0; j < rbm.getHiddenSize(); j++) {
			rbmexpected.hSparse(j) += abs(rbm.nodes.h(j));
		}

		for (int i = 0; i < rbm.getVisibleSize(); i++) {
			for (int j = 0; j < rbm.getHiddenSize(); j++) {
				rbmexpected.weight(i, j) += rbm.nodes.v(i) * rbm.nodes.h(j);
			}
		}
	}

	rbmexpected.vBias /= static_cast<double>(data_indexes.size());
	rbmexpected.hBias /= static_cast<double>(data_indexes.size());
	rbmexpected.weight /= static_cast<double>(data_indexes.size());
	rbmexpected.hSparse /= static_cast<double>(data_indexes.size());
}

template<class OPTIMIZERTYPE>
inline void Trainer<GeneralizedSparseRBM, OPTIMIZERTYPE>::calcRBMExpectedExact(GeneralizedSparseRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) {
	// 0埋め初期化
	initRBMExpected();

	auto z = rbm.getNormalConstant();

	for (int i = 0; i < rbm.getVisibleSize(); i++) {
		rbmexpected.vBias(i) = rbm.expectedValueVis(i, z);

		for (int j = 0; j < rbm.getHiddenSize(); j++) {
			rbmexpected.weight(i, j) = rbm.expectedValueVisHid(i, j, z);
		}
	}

	for (int j = 0; j < rbm.getHiddenSize(); j++) {
		rbmexpected.hBias(j) = rbm.expectedValueHid(j, z);
		rbmexpected.hSparse(j) = rbm.expectedValueAbsHid(j, z);
	}
}


// 勾配の計算
template<class OPTIMIZERTYPE>
inline void Trainer<GeneralizedSparseRBM, OPTIMIZERTYPE>::calcGradient(GeneralizedSparseRBM & rbm, std::vector<int> & data_indexes) {
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
		gradient.hSparse(j) = dataMean.hSparse(j) - rbmexpected.hSparse(j);
	}
}

template<class OPTIMIZERTYPE>
inline void Trainer<GeneralizedSparseRBM, OPTIMIZERTYPE>::updateMomentum(GeneralizedSparseRBM & rbm) {
	for (int i = 0; i < rbm.getVisibleSize(); i++) {
		momentum.vBias(i) = momentumRate * momentum.vBias(i) + learningRate * gradient.vBias(i);

		for (int j = 0; j < rbm.getHiddenSize(); j++) {
			momentum.weight(i, j) = momentumRate * momentum.weight(i, j) + learningRate * gradient.weight(i, j);
		}
	}

	for (int j = 0; j < rbm.getHiddenSize(); j++) {
		momentum.hBias(j) = momentumRate * momentum.hBias(j) + learningRate * gradient.hBias(j);
		momentum.hSparse(j) = momentumRate * momentum.hSparse(j) + learningRate * gradient.hSparse(j);
	}
}

// パラメータの更新
template<class OPTIMIZERTYPE>
inline void Trainer<GeneralizedSparseRBM, OPTIMIZERTYPE>::updateParams(GeneralizedSparseRBM & rbm) {
	for (int i = 0; i < rbm.getVisibleSize(); i++) {
		rbm.params.b(i) += momentum.vBias(i);

		for (int j = 0; j < rbm.getHiddenSize(); j++) {
			rbm.params.w(i, j) += momentum.weight(i, j);
		}
	}

	for (int j = 0; j < rbm.getHiddenSize(); j++) {
		rbm.params.c(j) += momentum.hBias(j);
		rbm.params.sparse(j) += momentum.hSparse(j);
	}
}


// 対数尤度関数
template<class OPTIMIZERTYPE>
inline double Trainer<GeneralizedSparseRBM, OPTIMIZERTYPE>::logLikeliHood(GeneralizedSparseRBM & rbm, std::vector<std::vector<double>> & dataset) {
	double value = 0.0;

	auto z = rbm.getNormalConstant();

	for (auto & data : dataset) {
		auto prob = rbm.probVis(data, z);
		value += log(prob);
	}

	return value;
}

// 学習情報出力(JSON)
template<class OPTIMIZERTYPE>
inline std::string Trainer<GeneralizedSparseRBM, OPTIMIZERTYPE>::trainInfoJson(GeneralizedSparseRBM & rbm) {
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

template<class OPTIMIZERTYPE>
inline void Trainer<GeneralizedSparseRBM, OPTIMIZERTYPE>::trainFromTrainInfo(GeneralizedSparseRBM & rbm, std::string json) {
	auto js = nlohmann::json::parse(json);
	rbm.params.deserialize(js["rbm"].dump());
	_trainCount = js["trainCount"];
	learningRate = js["learningRate"];
	momentumRate = js["momentumRate"];
	cdk = js["cdk"];
	rbm.setHiddenDiveSize(js["divSize"]);
	rbm.setRealHiddenValue(js["realFlag"]);
}
