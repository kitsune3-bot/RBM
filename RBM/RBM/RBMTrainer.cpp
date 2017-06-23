#include "RBMTrainer.h"
#include "RBM.h"
#include "RBMSampler.h"
#include <vector>
#include <numeric>
#include <random>

RBMTrainer::RBMTrainer()
{
}


RBMTrainer::~RBMTrainer()
{
}


RBMTrainer::RBMTrainer(RBM *rbm) {
    initMomentum(rbm);
    initGradient(rbm);
    initDataMean(rbm);
    initSampleMean(rbm);
}

void RBMTrainer::initMomentum(RBM *rbm) {
    momentum.vBias.setConstant(rbm->getVisibleSize(), 0.0);
    momentum.hBias.setConstant(rbm->getHiddenSize(), 0.0);
    momentum.weight.setConstant(rbm->getVisibleSize, rbm->getHiddenSize(), 0.0);
}

void RBMTrainer::initMomentum() {
    momentum.vBias.setConstant(0.0);
    momentum.hBias.setConstant(0.0);
    momentum.weight.setConstant(0.0);
}

void RBMTrainer::initGradient(RBM *rbm) {
    gradient.vBias.setConstant(rbm->getVisibleSize(), 0.0);
    gradient.hBias.setConstant(rbm->getHiddenSize(), 0.0);
    gradient.weight.setConstant(rbm->getVisibleSize, rbm->getHiddenSize(), 0.0);
}

void RBMTrainer::initGradient() {
    gradient.vBias.setConstant(0.0);
    gradient.hBias.setConstant(0.0);
    gradient.weight.setConstant(0.0);
}

void RBMTrainer::initDataMean(RBM *rbm) {
    dataMean.visible.setConstant(rbm->getVisibleSize(), 0.0);
    dataMean.hidden.setConstant(rbm->getHiddenSize(), 0.0);
}

void RBMTrainer::initDataMean() {
    dataMean.visible.setConstant(0.0);
    dataMean.hidden.setConstant(0.0);
}

void RBMTrainer::initSampleMean(RBM *rbm) {
    sampleMean.visible.setConstant(rbm->getVisibleSize(), 0.0);
    sampleMean.hidden.setConstant(rbm->getHiddenSize(), 0.0);
}

void RBMTrainer::initSampleMean() {
    sampleMean.visible.setConstant(0.0);
    sampleMean.hidden.setConstant(0.0);
}

// 1回だけ学習
void RBMTrainer::trainOnce(RBM *rbm,  std::vector<std::vector<double>> & dataset) {
    
    // データインデックス集合
    std::vector<int> data_indexes(dataset.size());
    
    // ミニバッチ学習のためにデータインデックスをシャッフルする
    std::iota(data_indexes.begin(), data_indexes.end(), 0);
    std::shuffle(data_indexes.begin(), data_indexes.end(), std::mt19937());
    
    // ミニバッチ
    // バッチサイズの確認
    int batch_size = batch_size < dataset.size() ? dataset.size() : batch_size;

    // ミニバッチ学習に使うデータのインデックス集合
    std::vector<int> minibatch_indexes(batch_size);
    std::copy(data_indexes.begin(), data_indexes.begin() + batch_size, minibatch_indexes);

    // Contrastive Divergence
    calcContrastiveDivergence(rbm, dataset, minibatch_indexes);

    // モーメンタムの更新
    updateMomentum();

    // 勾配の更新
    updateParams(rbm);

}

void RBMTrainer::calcContrastiveDivergence(RBM *rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) {
    // データ平均の計算
    calcDataMean(rbm, dataset, data_indexes);

    // サンプル平均の計算(CD)
    calcSampleMean(rbm, dataset, data_indexes);

    // 勾配計算
    calcGradient(data_indexes);
}

void RBMTrainer::calcDataMean(RBM *rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) {
    // 0埋め初期化
    initDataMean();

    for (auto & n : data_indexes) {
        auto & data = dataset[n];
        Eigen::VectorXd vect = Eigen::Map<Eigen::VectorXd>(data.data(), data.size());
        rbm->nodes.v = vect;

        dataMean.visible += vect;

        for (int j = 0; j < rbm->getHiddenSize; j++) {
            dataMean.hidden(j) += rbm->actHidJ(j);
        }
    }

    dataMean.visible /= static_cast<double>(data_indexes.size());
    dataMean.hidden /= static_cast<double>(data_indexes.size());
}

void RBMTrainer::calcSampleMean(RBM *rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) {
    // 0埋め初期化
    initSampleMean();

    for (auto & n : data_indexes) {
        auto & data = dataset[n];
        Eigen::VectorXd vect = Eigen::Map<Eigen::VectorXd>(data.data(), data.size());

        // RBMの初期値設定
        rbm->nodes.v = vect;

        for (int j = 0; j < rbm->getHiddenSize; j++) {
            dataMean.hidden(j) += rbm->actHidJ(j);
        }

        // CD-K
        RBMSampler sampler;
        for (int k = 0; k < cdk; k++) {
            sampler.updateByBlockedGibbsSamplingVisible(rbm);
            sampler.updateByBlockedGibbsSamplingHidden(rbm);
        }

        // 結果を格納
        dataMean.visible += rbm->nodes.v;
        dataMean.hidden += rbm->nodes.h;
    }

    dataMean.visible /= static_cast<double>(data_indexes.size());
    dataMean.hidden /= static_cast<double>(data_indexes.size());
}