#include "GeneralizedRBMTrainer.h"
#include "GeneralizedRBM.h"
#include "GeneralizedRBMSampler.h"
#include "rbmutil.h"
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
    initRBMExpected(rbm);
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
    dataMean.vBias.setConstant(rbm.getVisibleSize(), 0.0);
	dataMean.hBias.setConstant(rbm.getHiddenSize(), 0.0);
	dataMean.weight.setConstant(rbm.getVisibleSize(), rbm.getHiddenSize(), 0.0);
}

void GeneralizedRBMTrainer::initDataMean() {
    dataMean.vBias.setConstant(0.0);
	dataMean.hBias.setConstant(0.0);
	dataMean.weight.setConstant(0.0);
}

void GeneralizedRBMTrainer::initRBMExpected(GeneralizedRBM & rbm) {
	rbmexpected.vBias.setConstant(rbm.getVisibleSize(), 0.0);
	rbmexpected.hBias.setConstant(rbm.getHiddenSize(), 0.0);
	rbmexpected.weight.setConstant(rbm.getVisibleSize(), rbm.getHiddenSize(), 0.0);
}

void GeneralizedRBMTrainer::initRBMExpected() {
	rbmexpected.vBias.setConstant(0.0);
	rbmexpected.hBias.setConstant(0.0);
	rbmexpected.weight.setConstant(0.0);
}

void GeneralizedRBMTrainer::train(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset) {
    for (int e = 0; e < epoch; e++) {
        trainOnce(rbm, dataset);
    }
}

void GeneralizedRBMTrainer::trainCD(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset) {
	for (int e = 0; e < epoch; e++) {
		trainOnceCD(rbm, dataset);
	}
}

void GeneralizedRBMTrainer::trainExact(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset) {
	for (int e = 0; e < epoch; e++) {
		trainOnceExact(rbm, dataset);
	}
}


// FIXME: CDとExactをフラグで切り分けられるように
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

void GeneralizedRBMTrainer::trainOnceCD(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset) {
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

void GeneralizedRBMTrainer::trainOnceExact(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset) {
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

}


void GeneralizedRBMTrainer::calcContrastiveDivergence(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) {
    // データ平均の計算
    calcDataMean(rbm, dataset, data_indexes);

    // サンプル平均の計算(CD)
    calcRBMExpectedCD(rbm, dataset, data_indexes);

    // 勾配計算
    calcGradient(rbm, data_indexes);
}

void GeneralizedRBMTrainer::calcExact(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) {
	// データ平均の計算
	calcDataMean(rbm, dataset, data_indexes);

	// サンプル平均の計算(CD)
	calcRBMExpectedExact(rbm, dataset, data_indexes);

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

		for (int i = 0; i < rbm.getVisibleSize(); i++) {
			dataMean.vBias(i) = vect(i);

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

void GeneralizedRBMTrainer::calcRBMExpectedCD(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) {
    // 0埋め初期化
    initRBMExpected();

    for (auto & n : data_indexes) {
        auto & data = dataset[n];
        Eigen::VectorXd vect = Eigen::Map<Eigen::VectorXd>(data.data(), data.size());

        // GeneralizedRBMの初期値設定
        rbm.nodes.v = vect;

        //for (int j = 0; j < rbm.getHiddenSize(); j++) {
        //    rbm.nodes.h(j) = rbm.actHidJ(j);
        //}

        // CD-K
        GeneralizedRBMSampler sampler;
		sampler.updateByBlockedGibbsSamplingHidden(rbm);
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

void GeneralizedRBMTrainer::calcRBMExpectedExact(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) {
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
void GeneralizedRBMTrainer::calcGradient(GeneralizedRBM & rbm, std::vector<int> & data_indexes) {
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

	//std::cout << "------Gradient------" << std::endl;
	//std::cout << dataMean.vBias.transpose() << std::endl;
	//std::cout << rbmexpected.vBias.transpose() << std::endl;
	//std::cout << dataMean.hBias.transpose() << std::endl;
	//std::cout << rbmexpected.hBias.transpose() << std::endl;
	
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


// 対数尤度関数
double GeneralizedRBMTrainer::logLikeliHood(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset) {
	double value = 0.0;

	for (auto & data : dataset) {
		// FIXME: 分配関数使いまわしで高速化可能
		auto prob = rbm.probVis(data);
		value += log(prob);
	}

	return value;
}
