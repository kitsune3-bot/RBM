#include "GeneralizedRBMTrainer.h"
#include "GeneralizedRBM.h"
#include "GeneralizedRBMSampler.h"
#include <vector>
#include <numeric>
#include <random>

GeneralizedRBMTrainer::GeneralizedRBMTrainer()
{
}


GeneralizedRBMTrainer::~GeneralizedRBMTrainer()
{
}


GeneralizedRBMTrainer::GeneralizedRBMTrainer(GeneralizedRBM & rbm) {
    initMomentum(rbm);
    initGradient(rbm);
    initDataMean(rbm);
    initSampleMean(rbm);
}

void GeneralizedRBMTrainer::initMomentum(GeneralizedRBM & rbm) {
    momentum.vBias.setConstant(rbm.getVisibleSize(), 0.0);
    momentum.hBias.setConstant(rbm.getHiddenSize(), 0.0);
    momentum.weight.setConstant(rbm.getVisibleSize(), rbm.getHiddenSize(), 0.0);
}

void GeneralizedRBMTrainer::initMomentum() {
    momentum.vBias.setConstant(0.0);
    momentum.hBias.setConstant(0.0);
    momentum.weight.setConstant(0.0);
}

void GeneralizedRBMTrainer::initGradient(GeneralizedRBM & rbm) {
    gradient.vBias.setConstant(rbm.getVisibleSize(), 0.0);
    gradient.hBias.setConstant(rbm.getHiddenSize(), 0.0);
    gradient.weight.setConstant(rbm.getVisibleSize(), rbm.getHiddenSize(), 0.0);
}

void GeneralizedRBMTrainer::initGradient() {
    gradient.vBias.setConstant(0.0);
    gradient.hBias.setConstant(0.0);
    gradient.weight.setConstant(0.0);
}

void GeneralizedRBMTrainer::initDataMean(GeneralizedRBM & rbm) {
    dataMean.visible.setConstant(rbm.getVisibleSize(), 0.0);
    dataMean.hidden.setConstant(rbm.getHiddenSize(), 0.0);
}

void GeneralizedRBMTrainer::initDataMean() {
    dataMean.visible.setConstant(0.0);
    dataMean.hidden.setConstant(0.0);
}

void GeneralizedRBMTrainer::initSampleMean(GeneralizedRBM & rbm) {
    sampleMean.visible.setConstant(rbm.getVisibleSize(), 0.0);
    sampleMean.hidden.setConstant(rbm.getHiddenSize(), 0.0);
}

void GeneralizedRBMTrainer::initSampleMean() {
    sampleMean.visible.setConstant(0.0);
    sampleMean.hidden.setConstant(0.0);
}

void GeneralizedRBMTrainer::train(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset) {
    for (int e = 0; e < epoch; e++) {
        trainOnce(rbm, dataset);
    }
}


// 1回だけ学習
void GeneralizedRBMTrainer::trainOnce(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset) {
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

}

void GeneralizedRBMTrainer::calcContrastiveDivergence(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) {
    // データ平均の計算
    calcDataMean(rbm, dataset, data_indexes);

    // サンプル平均の計算(CD)
    calcSampleMeanCD(rbm, dataset, data_indexes);

    // 勾配計算
    calcGradient(rbm, data_indexes);
}

void GeneralizedRBMTrainer::calcExact(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) {
	// データ平均の計算
	calcDataMean(rbm, dataset, data_indexes);

	// サンプル平均の計算(CD)
	calcSampleMeanCD(rbm, dataset, data_indexes);

	// 勾配計算
	calcGradient(rbm, data_indexes);
}


void GeneralizedRBMTrainer::calcDataMean(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) {
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

void GeneralizedRBMTrainer::calcSampleMeanCD(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) {
    // 0埋め初期化
    initSampleMean();

    for (auto & n : data_indexes) {
        auto & data = dataset[n];
        Eigen::VectorXd vect = Eigen::Map<Eigen::VectorXd>(data.data(), data.size());

        // GeneralizedRBMの初期値設定
        rbm.nodes.v = vect;

        for (int j = 0; j < rbm.getHiddenSize(); j++) {
            rbm.nodes.h(j) = rbm.actHidJ(j);
        }

        // CD-K
        GeneralizedRBMSampler sampler;
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

void GeneralizedRBMTrainer::calcSampleMeanExact(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) {
	// 0埋め初期化
	initSampleMean();

	for (auto & n : data_indexes) {
		auto & data = dataset[n];
		Eigen::VectorXd vect = Eigen::Map<Eigen::VectorXd>(data.data(), data.size());

		// GeneralizedRBMの初期値設定
		rbm.nodes.v = vect;

		for (int j = 0; j < rbm.getHiddenSize(); j++) {
			rbm.nodes.h(j) = rbm.actHidJ(j);
		}

		// CD-K
		GeneralizedRBMSampler sampler;
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
void GeneralizedRBMTrainer::calcGradient(GeneralizedRBM & rbm, std::vector<int> & data_indexes) {
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

void GeneralizedRBMTrainer::updateMomentum(GeneralizedRBM & rbm) {
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
void GeneralizedRBMTrainer::updateParams(GeneralizedRBM & rbm) {
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
