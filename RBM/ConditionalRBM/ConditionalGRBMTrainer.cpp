#include "ConditionalGRBMTrainer.h"
#include "ConditionalGRBM.h"
#include "ConditionalGRBMSampler.h"
#include <vector>
#include <numeric>
#include <random>

ConditionalGRBMTrainer::ConditionalGRBMTrainer()
{
}


ConditionalGRBMTrainer::~ConditionalGRBMTrainer()
{
}


ConditionalGRBMTrainer::ConditionalGRBMTrainer(ConditionalGRBM & rbm) {
    initMomentum(rbm);
    initGradient(rbm);
    initDataMean(rbm);
    initSampleMean(rbm);
}

void ConditionalGRBMTrainer::initMomentum(ConditionalGRBM & rbm) {
    momentum.vBias.setConstant(rbm.getVisibleSize(), 0.0);
    momentum.vLambda.setConstant(rbm.getVisibleSize(), 0.0);
    momentum.hBias.setConstant(rbm.getHiddenSize(), 0.0);
    momentum.weight.setConstant(rbm.getVisibleSize(), rbm.getHiddenSize(), 0.0);
    momentum.hxWeight.setConstant(rbm.getHiddenSize(), rbm.getCondSize(), 0.0);
}

void ConditionalGRBMTrainer::initMomentum() {
    momentum.vBias.setConstant(0.0);
    momentum.vLambda.setConstant(0.0);
    momentum.hBias.setConstant(0.0);
    momentum.weight.setConstant(0.0);
    momentum.hxWeight.setConstant(0.0);
}

void ConditionalGRBMTrainer::initGradient(ConditionalGRBM & rbm) {
    gradient.vBias.setConstant(rbm.getVisibleSize(), 0.0);
    gradient.vLambda.setConstant(rbm.getVisibleSize(), 0.0);
    gradient.hBias.setConstant(rbm.getHiddenSize(), 0.0);
    gradient.weight.setConstant(rbm.getVisibleSize(), rbm.getHiddenSize(), 0.0);
    gradient.hxWeight.setConstant(rbm.getHiddenSize(), rbm.getCondSize(), 0.0);
}

void ConditionalGRBMTrainer::initGradient() {
    gradient.vBias.setConstant(0.0);
    gradient.vLambda.setConstant(0.0);
    gradient.hBias.setConstant(0.0);
    gradient.weight.setConstant(0.0);
    gradient.hxWeight.setConstant(0.0);
}

void ConditionalGRBMTrainer::initDataMean(ConditionalGRBM & rbm) {
    dataMean.visible.setConstant(rbm.getVisibleSize(), 0.0);
    dataMean.visible2.setConstant(rbm.getVisibleSize(), 0.0);  // Gaussian Unit
    dataMean.hidden.setConstant(rbm.getHiddenSize(), 0.0);
    dataMean.conditional.setConstant(rbm.getHiddenSize(), 0.0);
}

void ConditionalGRBMTrainer::initDataMean() {
    dataMean.visible.setConstant(0.0);
    dataMean.visible2.setConstant(0.0);  // Gaussian Unit
    dataMean.hidden.setConstant(0.0);
    dataMean.conditional.setConstant(0.0);
}

void ConditionalGRBMTrainer::initSampleMean(ConditionalGRBM & rbm) {
    sampleMean.visible.setConstant(rbm.getVisibleSize(), 0.0);
    sampleMean.visible2.setConstant(rbm.getVisibleSize(), 0.0);  // Gaussian Unit
    sampleMean.hidden.setConstant(rbm.getHiddenSize(), 0.0);
    sampleMean.conditional.setConstant(rbm.getHiddenSize(), 0.0);
}

void ConditionalGRBMTrainer::initSampleMean() {
    sampleMean.visible.setConstant(0.0);
    sampleMean.visible2.setConstant(0.0);  // Gaussian Unit
    sampleMean.hidden.setConstant(0.0);
    sampleMean.conditional.setConstant(0.0);
}

void ConditionalGRBMTrainer::train(ConditionalGRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<std::vector<double>> & cond_dataset) {
    for (int e = 0; e < epoch; e++) {
        trainOnce(rbm, dataset, cond_dataset);
    }
}


// 1回だけ学習
void ConditionalGRBMTrainer::trainOnce(ConditionalGRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<std::vector<double>> & cond_dataset) {

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
    calcContrastiveDivergence(rbm, dataset, cond_dataset, minibatch_indexes);

    // モーメンタムの更新
    updateMomentum(rbm);

    // 勾配の更新
    updateParams(rbm);

}

void ConditionalGRBMTrainer::calcContrastiveDivergence(ConditionalGRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<std::vector<double>> & cond_dataset, std::vector<int> & data_indexes) {
    // データ平均の計算
    calcDataMean(rbm, dataset, cond_dataset, data_indexes);

    // サンプル平均の計算(CD)
    calcSampleMean(rbm, dataset, cond_dataset, data_indexes);

    // 勾配計算
    calcGradient(rbm, data_indexes);
}

void ConditionalGRBMTrainer::calcDataMean(ConditionalGRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<std::vector<double>> & cond_dataset, std::vector<int> & data_indexes) {
    // 0埋め初期化
    initDataMean();

    for (auto & n : data_indexes) {
        auto & data = dataset[n];
        auto & cond_data = dataset[n];
        Eigen::VectorXd vect = Eigen::Map<Eigen::VectorXd>(data.data(), data.size());
        rbm.nodes.v = vect;
        Eigen::VectorXd cond_vect = Eigen::Map<Eigen::VectorXd>(cond_data.data(), cond_data.size());
        rbm.nodes.x = cond_vect;

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
    dataMean.conditional /= static_cast<double>(data_indexes.size());
}

void ConditionalGRBMTrainer::calcSampleMean(ConditionalGRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<std::vector<double>> & cond_dataset, std::vector<int> & data_indexes) {
    // 0埋め初期化
    initSampleMean();

    for (auto & n : data_indexes) {
        auto & data = dataset[n];
        auto & cond_data = cond_dataset[n];
        Eigen::VectorXd vect = Eigen::Map<Eigen::VectorXd>(data.data(), data.size());
        Eigen::VectorXd cond_vect = Eigen::Map<Eigen::VectorXd>(cond_data.data(), cond_data.size());

        // ConditionalGRBMの初期値設定
        rbm.nodes.v = vect;
        rbm.nodes.x = cond_vect;

        for (int j = 0; j < rbm.getHiddenSize(); j++) {
            rbm.nodes.h(j) = rbm.actHidJ(j);
        }

        // CD-K
        ConditionalGRBMSampler sampler;
        for (int k = 0; k < cdk; k++) {
            sampler.updateByBlockedGibbsSamplingVisible(rbm);
            sampler.updateByBlockedGibbsSamplingHidden(rbm);
        }

        // 結果を格納
        sampleMean.visible += rbm.nodes.v;
        sampleMean.hidden += rbm.nodes.h;
        sampleMean.hidden += cond_vect;

        // Gausiann Unit限定 
        for (int i = 0; i < rbm.getVisibleSize(); i++) {
            sampleMean.visible2(i) += rbm.nodes.v(i) * rbm.nodes.v(i) / 2.0;
        }
    }

    sampleMean.visible /= static_cast<double>(data_indexes.size());
    sampleMean.hidden /= static_cast<double>(data_indexes.size());
    sampleMean.conditional /= static_cast<double>(data_indexes.size());
}

// 勾配の計算
void ConditionalGRBMTrainer::calcGradient(ConditionalGRBM & rbm, std::vector<int> & data_indexes) {
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

        for (int k = 0; k < rbm.getCondSize(); k++) {
            gradient.hxWeight(j, k) = dataMean.hidden(j) * dataMean.conditional(k) - sampleMean.hidden(j) * sampleMean.conditional(k);
        }
    }
}

void ConditionalGRBMTrainer::updateMomentum(ConditionalGRBM & rbm) {
    for (int i = 0; i < rbm.getVisibleSize(); i++) {
        momentum.vBias(i) = momentumRate * momentum.vBias(i) + learningRate * gradient.vBias(i);
        //momentum.vLambda(i) = momentumRate * momentum.vLambda(i) + learningRate * gradient.vLambda(i);  // 非負制約満たすこと

        for (int j = 0; j < rbm.getHiddenSize(); j++) {
            momentum.weight(i, j) = momentumRate * momentum.weight(i, j) + learningRate * gradient.weight(i, j);
        }
    }

    for (int j = 0; j < rbm.getHiddenSize(); j++) {
        momentum.hBias(j) = momentumRate * momentum.hBias(j) + learningRate * gradient.hBias(j);

        for (int k = 0; k < rbm.getCondSize(); k++) {
            momentum.hxWeight(j, k) = momentumRate * momentum.hxWeight(j, k) + learningRate * gradient.hxWeight(j, k);
        }
    }
}

// パラメータの更新
void ConditionalGRBMTrainer::updateParams(ConditionalGRBM & rbm) {
    for (int i = 0; i < rbm.getVisibleSize(); i++) {
        rbm.params.b(i) += momentum.vBias(i);
        //rbm.params.lambda(i) += momentum.vLambda(i);  // 非負制約を満たすこと

        for (int j = 0; j < rbm.getHiddenSize(); j++) {
            rbm.params.w(i, j) += momentum.weight(i, j);
        }
    }

    for (int j = 0; j < rbm.getHiddenSize(); j++) {
        rbm.params.c(j) += momentum.hBias(j);

        for (int k = 0; k < rbm.getCondSize(); k++) {
            rbm.params.hxW(j, k) += momentum.hxWeight(j, k);
        }
    }
}
