#include "GeneralizedSparseRBMTrainer.h"
#include "GeneralizedSparseRBM.h"
#include "GeneralizedSparseRBMSampler.h"
#include <vector>
#include <numeric>
#include <random>

GeneralizedSparseRBMTrainer::GeneralizedSparseRBMTrainer()
{
}


GeneralizedSparseRBMTrainer::~GeneralizedSparseRBMTrainer()
{
}


GeneralizedSparseRBMTrainer::GeneralizedSparseRBMTrainer(GeneralizedSparseRBM & rbm) {
	initMomentum(rbm);
	initGradient(rbm);
	initDataMean(rbm);
	initRBMExpected(rbm);
}

void GeneralizedSparseRBMTrainer::initMomentum(GeneralizedSparseRBM & rbm) {
	momentum.vBias.setConstant(rbm.getVisibleSize(), 0.0);
	momentum.hBias.setConstant(rbm.getHiddenSize(), 0.0);
	momentum.weight.setConstant(rbm.getVisibleSize(), rbm.getHiddenSize(), 0.0);
	momentum.hSparse.setConstant(rbm.getHiddenSize(), 0.0);
}

void GeneralizedSparseRBMTrainer::initMomentum() {
	momentum.vBias.setConstant(0.0);
	momentum.hBias.setConstant(0.0);
	momentum.weight.setConstant(0.0);
	momentum.hSparse.setConstant(0.0);
}

void GeneralizedSparseRBMTrainer::initGradient(GeneralizedSparseRBM & rbm) {
	gradient.vBias.setConstant(rbm.getVisibleSize(), 0.0);
	gradient.hBias.setConstant(rbm.getHiddenSize(), 0.0);
	gradient.hSparse.setConstant(rbm.getHiddenSize(), 0.0);
	gradient.weight.setConstant(rbm.getVisibleSize(), rbm.getHiddenSize(), 0.0);
}

void GeneralizedSparseRBMTrainer::initGradient() {
	gradient.vBias.setConstant(0.0);
	gradient.hBias.setConstant(0.0);
	gradient.hSparse.setConstant(0.0);
	gradient.weight.setConstant(0.0);
}

void GeneralizedSparseRBMTrainer::initDataMean(GeneralizedSparseRBM & rbm) {
	dataMean.vBias.setConstant(rbm.getVisibleSize(), 0.0);
	dataMean.hBias.setConstant(rbm.getHiddenSize(), 0.0);
	dataMean.weight.setConstant(rbm.getVisibleSize(), rbm.getHiddenSize(), 0.0);
	dataMean.hSparse.setConstant(rbm.getHiddenSize(), 0.0);
}

void GeneralizedSparseRBMTrainer::initDataMean() {
	dataMean.vBias.setConstant(0.0);
	dataMean.hBias.setConstant(0.0);
	dataMean.weight.setConstant(0.0);
	dataMean.hSparse.setConstant(0.0);
}

void GeneralizedSparseRBMTrainer::initRBMExpected(GeneralizedSparseRBM & rbm) {
	rbmexpected.vBias.setConstant(rbm.getVisibleSize(), 0.0);
	rbmexpected.hBias.setConstant(rbm.getHiddenSize(), 0.0);
	rbmexpected.weight.setConstant(rbm.getVisibleSize(), rbm.getHiddenSize(), 0.0);
	rbmexpected.hSparse.setConstant(rbm.getHiddenSize(), 0.0);
}

void GeneralizedSparseRBMTrainer::initRBMExpected() {
	rbmexpected.vBias.setConstant(0.0);
	rbmexpected.hBias.setConstant(0.0);
	rbmexpected.weight.setConstant(0.0);
	rbmexpected.hSparse.setConstant(0.0);
}

void GeneralizedSparseRBMTrainer::train(GeneralizedSparseRBM & rbm, std::vector<std::vector<double>> & dataset) {
	for (int e = 0; e < epoch; e++) {
		trainOnce(rbm, dataset);
	}
}

void GeneralizedSparseRBMTrainer::trainCD(GeneralizedSparseRBM & rbm, std::vector<std::vector<double>> & dataset) {
	for (int e = 0; e < epoch; e++) {
		trainOnceCD(rbm, dataset);
	}
}

void GeneralizedSparseRBMTrainer::trainExact(GeneralizedSparseRBM & rbm, std::vector<std::vector<double>> & dataset) {
	for (int e = 0; e < epoch; e++) {
		trainOnceExact(rbm, dataset);
	}
}


// FIXME: CDとExactをフラグで切り分けられるように
// 1回だけ学習
void GeneralizedSparseRBMTrainer::trainOnce(GeneralizedSparseRBM & rbm, std::vector<std::vector<double>> & dataset) {
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

void GeneralizedSparseRBMTrainer::trainOnceCD(GeneralizedSparseRBM & rbm, std::vector<std::vector<double>> & dataset) {
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

void GeneralizedSparseRBMTrainer::trainOnceExact(GeneralizedSparseRBM & rbm, std::vector<std::vector<double>> & dataset) {
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


void GeneralizedSparseRBMTrainer::calcContrastiveDivergence(GeneralizedSparseRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) {
	// データ平均の計算
	calcDataMean(rbm, dataset, data_indexes);

	// サンプル平均の計算(CD)
	calcRBMExpectedCD(rbm, dataset, data_indexes);

	// 勾配計算
	calcGradient(rbm, data_indexes);
}

void GeneralizedSparseRBMTrainer::calcExact(GeneralizedSparseRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) {
	// データ平均の計算
	calcDataMean(rbm, dataset, data_indexes);

	// サンプル平均の計算(CD)
	calcRBMExpectedExact(rbm, dataset, data_indexes);

	// 勾配計算
	calcGradient(rbm, data_indexes);
}


void GeneralizedSparseRBMTrainer::calcDataMean(GeneralizedSparseRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) {
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

void GeneralizedSparseRBMTrainer::calcRBMExpectedCD(GeneralizedSparseRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) {
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
		GeneralizedSparseRBMSampler sampler;
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

void GeneralizedSparseRBMTrainer::calcRBMExpectedExact(GeneralizedSparseRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) {
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
		rbmexpected.hSparse(j) = rbm.expectedValueAbsHid(j);
	}
}


// 勾配の計算
void GeneralizedSparseRBMTrainer::calcGradient(GeneralizedSparseRBM & rbm, std::vector<int> & data_indexes) {
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

void GeneralizedSparseRBMTrainer::updateMomentum(GeneralizedSparseRBM & rbm) {
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
void GeneralizedSparseRBMTrainer::updateParams(GeneralizedSparseRBM & rbm) {
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
double GeneralizedSparseRBMTrainer::logLikeliHood(GeneralizedSparseRBM & rbm, std::vector<std::vector<double>> & dataset) {
	double value = 0.0;

	for (auto & data : dataset) {
		// FIXME: 分配関数使いまわしで高速化可能
		auto prob = rbm.probVis(data);
		value += log(prob);
	}

	return value;
}

// 学習情報出力(JSON)
std::string GeneralizedSparseRBMTrainer::trainInfoJson(GeneralizedSparseRBM & rbm) {
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

void GeneralizedSparseRBMTrainer::trainFromTrainInfo(GeneralizedSparseRBM & rbm, std::string json) {
	auto js = nlohmann::json::parse(json);
	rbm.params.deserialize(js["rbm"].dump());
	_trainCount = js["trainCount"];
	learningRate = js["learningRate"];
	momentumRate = js["momentumRate"];
	cdk = js["cdk"];
	rbm.setHiddenDiveSize(js["divSize"]);
	rbm.setRealHiddenValue(js["realFlag"]);
}
