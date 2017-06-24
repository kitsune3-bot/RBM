#include "GBRBMTrainer.h"
#include "GBRBM.h"
#include "GBRBMSampler.h"
#include <vector>
#include <numeric>
#include <random>

GBRBMTrainer::GBRBMTrainer()
{
}


GBRBMTrainer::~GBRBMTrainer()
{
}


GBRBMTrainer::GBRBMTrainer(GBRBM & rbm) {
    initMomentum(rbm);
    initGradient(rbm);
    initDataMean(rbm);
    initSampleMean(rbm);
}

void GBRBMTrainer::initMomentum(GBRBM & rbm) {
    momentum.vBias.setConstant(rbm.getVisibleSize(), 0.0);
    momentum.vLambda.setConstant(rbm.getVisibleSize(), 0.0);
    momentum.hBias.setConstant(rbm.getHiddenSize(), 0.0);
    momentum.weight.setConstant(rbm.getVisibleSize(), rbm.getHiddenSize(), 0.0);
}

void GBRBMTrainer::initMomentum() {
    momentum.vBias.setConstant(0.0);
    momentum.vLambda.setConstant(0.0);
    momentum.hBias.setConstant(0.0);
    momentum.weight.setConstant(0.0);
}

void GBRBMTrainer::initGradient(GBRBM & rbm) {
    gradient.vBias.setConstant(rbm.getVisibleSize(), 0.0);
    gradient.vLambda.setConstant(rbm.getVisibleSize(), 0.0);
    gradient.hBias.setConstant(rbm.getHiddenSize(), 0.0);
    gradient.weight.setConstant(rbm.getVisibleSize(), rbm.getHiddenSize(), 0.0);
}

void GBRBMTrainer::initGradient() {
    gradient.vBias.setConstant(0.0);
    gradient.vLambda.setConstant(0.0);
    gradient.hBias.setConstant(0.0);
    gradient.weight.setConstant(0.0);
}

void GBRBMTrainer::initDataMean(GBRBM & rbm) {
    dataMean.visible.setConstant(rbm.getVisibleSize(), 0.0);
    dataMean.hidden.setConstant(rbm.getHiddenSize(), 0.0);
}

void GBRBMTrainer::initDataMean() {
    dataMean.visible.setConstant(0.0);
    dataMean.hidden.setConstant(0.0);
}

void GBRBMTrainer::initSampleMean(GBRBM & rbm) {
    sampleMean.visible.setConstant(rbm.getVisibleSize(), 0.0);
    sampleMean.hidden.setConstant(rbm.getHiddenSize(), 0.0);
}

void GBRBMTrainer::initSampleMean() {
    sampleMean.visible.setConstant(0.0);
    sampleMean.hidden.setConstant(0.0);
}

void GBRBMTrainer::train(GBRBM & rbm, std::vector<std::vector<double>> & dataset) {
    for (int e = 0; e < epoch; e++) {
        trainOnce(rbm, dataset);
    }
}


// 1回だけ学習
void GBRBMTrainer::trainOnce(GBRBM & rbm, std::vector<std::vector<double>> & dataset) {

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

void GBRBMTrainer::calcContrastiveDivergence(GBRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) {
    // データ平均の計算
    calcDataMean(rbm, dataset, data_indexes);

    // サンプル平均の計算(CD)
    calcSampleMean(rbm, dataset, data_indexes);

    // 勾配計算
    calcGradient(rbm, data_indexes);
}

void GBRBMTrainer::calcDataMean(GBRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) {
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

void GBRBMTrainer::calcSampleMean(GBRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) {
    // 0埋め初期化
    initSampleMean();

    for (auto & n : data_indexes) {
        auto & data = dataset[n];
        Eigen::VectorXd vect = Eigen::Map<Eigen::VectorXd>(data.data(), data.size());

        // GBRBMの初期値設定
        rbm.nodes.v = vect;

        for (int j = 0; j < rbm.getHiddenSize(); j++) {
            rbm.nodes.h(j) = rbm.actHidJ(j);
        }

        // CD-K
        GBRBMSampler sampler;
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
void GBRBMTrainer::calcGradient(GBRBM & rbm, std::vector<int> & data_indexes) {
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

void GBRBMTrainer::updateMomentum(GBRBM & rbm) {
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
void GBRBMTrainer::updateParams(GBRBM & rbm) {
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
