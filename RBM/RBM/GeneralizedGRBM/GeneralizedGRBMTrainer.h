#pragma once
#include "../Trainer.h"
#include "GeneralizedGRBM.h"
#include "Eigen/Core"
#include <vector>
#include <numeric>
#include <random>

template<>
class Trainer<GeneralizedGRBM> {
    struct Momentum {
        Eigen::VectorXd vBias;
        Eigen::VectorXd vLambda;
        Eigen::VectorXd hBias;
        Eigen::MatrixXd weight;
    };

    struct Gradient {
        Eigen::VectorXd vBias;
        Eigen::VectorXd vLambda;
        Eigen::VectorXd hBias;
        Eigen::MatrixXd weight;
    };

    struct DataMean {
        Eigen::VectorXd visible;
        Eigen::VectorXd visible2;  // Gausiann Unit限定 
        Eigen::VectorXd hidden;
    };

    struct RBMExpected {
        Eigen::VectorXd visible;
        Eigen::VectorXd visible2;  // Gausiann Unit限定 
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
    Trainer(GeneralizedGRBM & rbm);
    ~Trainer();

    // モーメンタムベクトル初期化
    void initMomentum(RBMBase & rbm) { initMomentum(reinterpret_cast<GeneralizedGRBM &>(rbm)); }
    void initMomentum(GeneralizedGRBM & rbm);

    // 確保済みのモーメンタムベクトルを0初期化
    void initMomentum();

    // 勾配ベクトル初期化
    void initGradient(RBMBase & rbm) { initGradient(reinterpret_cast<GeneralizedGRBM &>(rbm)); }
    void initGradient(GeneralizedGRBM & rbm);

    // 確保済みの勾配ベクトルを0初期化
    void initGradient();

    // データ平均ベクトルを初期化
    void initDataMean(RBMBase & rbm) { initDataMean(reinterpret_cast<GeneralizedGRBM &>(rbm)); }
    void initDataMean(GeneralizedGRBM & rbm);

    // 確保済みデータ平均ベクトルを初期化
    void initDataMean();

    // サンプル平均ベクトルを初期化
    void initRBMExpected(RBMBase & rbm) { initRBMExpected(reinterpret_cast<GeneralizedGRBM &>(rbm)); }
    void initRBMExpected(GeneralizedGRBM & rbm);

    // 確保済みサンプル平均ベクトルを初期化
    void initRBMExpected();

    // 学習
    void train(RBMBase & rbm, std::vector<std::vector<double>> & dataset) { train(reinterpret_cast<GeneralizedGRBM &>(rbm), dataset); }
    void train(GeneralizedGRBM & rbm, std::vector<std::vector<double>> & dataset);

    // 1回だけ学習
    void trainOnce(RBMBase & rbm, std::vector<std::vector<double>> & dataset) { trainOnce(reinterpret_cast<GeneralizedGRBM &>(rbm), dataset); }
    void trainOnce(GeneralizedGRBM & rbm, std::vector<std::vector<double>> & dataset);

    // CD計算
    void calcContrastiveDivergence(RBMBase & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) { calcContrastiveDivergence(reinterpret_cast<GeneralizedGRBM &>(rbm), dataset, data_indexes); }
    void calcContrastiveDivergence(GeneralizedGRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes);

    // データ平均の計算
    void calcDataMean(RBMBase & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) { calcDataMean(reinterpret_cast<GeneralizedGRBM &>(rbm), dataset, data_indexes); }
    void calcDataMean(GeneralizedGRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes);

    // サンプル平均の計算
    void calcRBMExpectedCD(RBMBase & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) { calcRBMExpectedCD(reinterpret_cast<GeneralizedGRBM &>(rbm), dataset, data_indexes); }
    void calcRBMExpectedCD(GeneralizedGRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes);

    // 勾配の計算
    void calcGradient(RBMBase & rbm, std::vector<int> & data_indexes) { calcGradient(reinterpret_cast<GeneralizedGRBM &>(rbm), data_indexes); }
    void calcGradient(GeneralizedGRBM & rbm, std::vector<int> & data_indexes);

    // モーメンタム更新
    void updateMomentum(RBMBase & rbm) { updateMomentum(reinterpret_cast<GeneralizedGRBM &>(rbm)); }
    void updateMomentum(GeneralizedGRBM & rbm);

    // 勾配更新
    void updateParams(RBMBase & rbm) { updateParams(reinterpret_cast<GeneralizedGRBM &>(rbm)); }
    void updateParams(GeneralizedGRBM & rbm);

	// 学習情報出力(JSON)
	std::string trainInfoJson(RBMBase & rbm) { trainInfoJson(reinterpret_cast<GeneralizedGRBM &>(rbm)); };
	std::string trainInfoJson(GeneralizedGRBM & rbm);

	// 学習情報から学習(JSON)
	void trainFromTrainInfo(RBMBase & rbm, std::string json) { trainFromTrainInfo(reinterpret_cast<GeneralizedGRBM &>(rbm), json); };
	void trainFromTrainInfo(GeneralizedGRBM & rbm, std::string json);
};

Trainer<GeneralizedGRBM>::Trainer(GeneralizedGRBM & rbm) {
	initMomentum(rbm);
	initGradient(rbm);
	initDataMean(rbm);
	initRBMExpected(rbm);
}

void Trainer<GeneralizedGRBM>::initMomentum(GeneralizedGRBM & rbm) {
	momentum.vBias.setConstant(rbm.getVisibleSize(), 0.0);
	momentum.vLambda.setConstant(rbm.getVisibleSize(), 0.0);
	momentum.hBias.setConstant(rbm.getHiddenSize(), 0.0);
	momentum.weight.setConstant(rbm.getVisibleSize(), rbm.getHiddenSize(), 0.0);
}

void Trainer<GeneralizedGRBM>::initMomentum() {
	momentum.vBias.setConstant(0.0);
	momentum.vLambda.setConstant(0.0);
	momentum.hBias.setConstant(0.0);
	momentum.weight.setConstant(0.0);
}

void Trainer<GeneralizedGRBM>::initGradient(GeneralizedGRBM & rbm) {
	gradient.vBias.setConstant(rbm.getVisibleSize(), 0.0);
	gradient.vLambda.setConstant(rbm.getVisibleSize(), 0.0);
	gradient.hBias.setConstant(rbm.getHiddenSize(), 0.0);
	gradient.weight.setConstant(rbm.getVisibleSize(), rbm.getHiddenSize(), 0.0);
}

void Trainer<GeneralizedGRBM>::initGradient() {
	gradient.vBias.setConstant(0.0);
	gradient.vLambda.setConstant(0.0);
	gradient.hBias.setConstant(0.0);
	gradient.weight.setConstant(0.0);
}

void Trainer<GeneralizedGRBM>::initDataMean(GeneralizedGRBM & rbm) {
	dataMean.visible.setConstant(rbm.getVisibleSize(), 0.0);
	dataMean.visible2.setConstant(rbm.getVisibleSize(), 0.0);  // Gaussian Unit
	dataMean.hidden.setConstant(rbm.getHiddenSize(), 0.0);
}

void Trainer<GeneralizedGRBM>::initDataMean() {
	dataMean.visible.setConstant(0.0);
	dataMean.visible2.setConstant(0.0);  // Gaussian Unit
	dataMean.hidden.setConstant(0.0);
}

void Trainer<GeneralizedGRBM>::initRBMExpected(GeneralizedGRBM & rbm) {
	sampleMean.visible.setConstant(rbm.getVisibleSize(), 0.0);
	sampleMean.visible2.setConstant(rbm.getVisibleSize(), 0.0);  // Gaussian Unit
	sampleMean.hidden.setConstant(rbm.getHiddenSize(), 0.0);
}

void Trainer<GeneralizedGRBM>::initRBMExpected() {
	sampleMean.visible.setConstant(0.0);
	sampleMean.visible2.setConstant(0.0);  // Gaussian Unit
	sampleMean.hidden.setConstant(0.0);
}

void Trainer<GeneralizedGRBM>::train(GeneralizedGRBM & rbm, std::vector<std::vector<double>> & dataset) {
	for (int e = 0; e < epoch; e++) {
		trainOnce(rbm, dataset);
	}
}


// 1回だけ学習
void Trainer<GeneralizedGRBM>::trainOnce(GeneralizedGRBM & rbm, std::vector<std::vector<double>> & dataset) {

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

void Trainer<GeneralizedGRBM>::calcContrastiveDivergence(GeneralizedGRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) {
	// データ平均の計算
	calcDataMean(rbm, dataset, data_indexes);

	// サンプル平均の計算(CD)
	calcRBMExpectedCD(rbm, dataset, data_indexes);

	// 勾配計算
	calcGradient(rbm, data_indexes);
}

void Trainer<GeneralizedGRBM>::calcDataMean(GeneralizedGRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) {
	// 0埋め初期化
	initDataMean();

	for (auto & n : data_indexes) {
		auto & data = dataset[n];
		Eigen::VectorXd vect = Eigen::Map<Eigen::VectorXd>(data.data(), data.size());
		rbm.nodes.v = vect;

		dataMean.visible += vect;
		// Gausiann Unit限定 
		for (int i = 0; i < rbm.getVisibleSize(); i++) {
			dataMean.visible2(i) += vect(i) * vect(i) / 2.0;  // Gausiann Unit限定 
		}

		for (int j = 0; j < rbm.getHiddenSize(); j++) {
			dataMean.hidden(j) += rbm.actHidJ(j);
		}
	}

	dataMean.visible /= static_cast<double>(data_indexes.size());
	dataMean.hidden /= static_cast<double>(data_indexes.size());
}

void Trainer<GeneralizedGRBM>::calcRBMExpectedCD(GeneralizedGRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) {
	// 0埋め初期化
	initRBMExpected();

	for (auto & n : data_indexes) {
		auto & data = dataset[n];
		Eigen::VectorXd vect = Eigen::Map<Eigen::VectorXd>(data.data(), data.size());

		// GeneralizedGRBMの初期値設定
		rbm.nodes.v = vect;

		for (int j = 0; j < rbm.getHiddenSize(); j++) {
			rbm.nodes.h(j) = rbm.actHidJ(j);
		}

		// CD-K
		Sampler<GeneralizedGRBM> sampler;
		for (int k = 0; k < cdk; k++) {
			sampler.updateByBlockedGibbsSamplingVisible(rbm);
			sampler.updateByBlockedGibbsSamplingHidden(rbm);
		}

		// 結果を格納
		sampleMean.visible += rbm.nodes.v;
		sampleMean.hidden += rbm.nodes.h;

		// Gausiann Unit限定 
		for (int i = 0; i < rbm.getVisibleSize(); i++) {
			sampleMean.visible2(i) += rbm.nodes.v(i) * rbm.nodes.v(i) / 2.0;
		}
	}

	sampleMean.visible /= static_cast<double>(data_indexes.size());
	sampleMean.hidden /= static_cast<double>(data_indexes.size());
}

// 勾配の計算
void Trainer<GeneralizedGRBM>::calcGradient(GeneralizedGRBM & rbm, std::vector<int> & data_indexes) {
	// 勾配ベクトルリセット
	initGradient();

	for (int i = 0; i < rbm.getVisibleSize(); i++) {
		gradient.vBias(i) = dataMean.visible(i) - sampleMean.visible(i);
		//gradient.vLambda(i) = dataMean.visible2(i) - sampleMean.visible2(i);

		for (int j = 0; j < rbm.getHiddenSize(); j++) {
			gradient.weight(i, j) = dataMean.visible(i) * dataMean.hidden(j) - sampleMean.visible(i) * sampleMean.hidden(j);
		}
	}

	for (int j = 0; j < rbm.getHiddenSize(); j++) {
		gradient.hBias(j) = dataMean.hidden(j) - sampleMean.hidden(j);
	}
}

void Trainer<GeneralizedGRBM>::updateMomentum(GeneralizedGRBM & rbm) {
	for (int i = 0; i < rbm.getVisibleSize(); i++) {
		momentum.vBias(i) = momentumRate * momentum.vBias(i) + learningRate * gradient.vBias(i);
		//momentum.vLambda(i) = momentumRate * momentum.vLambda(i) + learningRate * gradient.vLambda(i);  // 非負制約満たすこと

		for (int j = 0; j < rbm.getHiddenSize(); j++) {
			momentum.weight(i, j) = momentumRate * momentum.weight(i, j) + learningRate * gradient.weight(i, j);
		}
	}

	for (int j = 0; j < rbm.getHiddenSize(); j++) {
		momentum.hBias(j) = momentumRate * momentum.hBias(j) + learningRate * gradient.hBias(j);
	}
}

// パラメータの更新
void Trainer<GeneralizedGRBM>::updateParams(GeneralizedGRBM & rbm) {
	for (int i = 0; i < rbm.getVisibleSize(); i++) {
		rbm.params.b(i) += momentum.vBias(i);
		//rbm.params.lambda(i) += momentum.vLambda(i);  // 非負制約を満たすこと

		for (int j = 0; j < rbm.getHiddenSize(); j++) {
			rbm.params.w(i, j) += momentum.weight(i, j);
		}
	}

	for (int j = 0; j < rbm.getHiddenSize(); j++) {
		rbm.params.c(j) += momentum.hBias(j);
	}
}

// 学習情報出力(JSON)
std::string Trainer<GeneralizedGRBM>::trainInfoJson(GeneralizedGRBM & rbm) {
	auto js = nlohmann::json();
	js["rbm"] = nlohmann::json::parse(rbm.params.serialize());
	js["trainCount"] = _trainCount;
	js["learningRate"] = learningRate;
	js["momentumRate"] = momentumRate;
	js["cdk"] = cdk;
	js["divSize"] = rbm.getHiddenDivSize();
	//js["realFlag"] = rbm.isRealHiddenValue();

	return js.dump();
}

void Trainer<GeneralizedGRBM>::trainFromTrainInfo(GeneralizedGRBM & rbm, std::string json) {
	auto js = nlohmann::json::parse(json);
	rbm.params.deserialize(js["rbm"].dump());
	_trainCount = js["trainCount"];
	learningRate = js["learningRate"];
	momentumRate = js["momentumRate"];
	cdk = js["cdk"];
	rbm.setHiddenDiveSize(js["divSize"]);
	//rbm.setRealHiddenValue(js["realFlag"]);
}