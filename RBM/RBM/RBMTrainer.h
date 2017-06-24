#pragma once
#include "RBMTrainerBase.h"
#include <Eigen/Core>
#include <vector>

class RBM;

class RBMTrainer : RBMTrainerBase {
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

    struct SampleMean {
        Eigen::VectorXd visible;
        Eigen::VectorXd hidden;
    };

private:
    Momentum momentum;
    Gradient gradient;
    DataMean dataMean;
    SampleMean sampleMean;


public:
    int epoch = 0;
    int batchSize = 1;
    int cdk = 0;
    double learningRate = 0.01;
    double momentumRate = 0.9;

public:
    RBMTrainer();
    RBMTrainer(RBMBase & rbm) { RBMTrainer(reinterpret_cast<RBM &>(rbm)); }
    RBMTrainer(RBM & rbm);
    ~RBMTrainer();

    // モーメンタムベクトル初期化
    void initMomentum(RBMBase & rbm) { initMomentum(reinterpret_cast<RBM &>(rbm)); }
    void initMomentum(RBM & rbm);

    // 確保済みのモーメンタムベクトルを0初期化
    void initMomentum();

    // 勾配ベクトル初期化
    void initGradient(RBMBase & rbm) { initGradient(reinterpret_cast<RBM &>(rbm)); }
    void initGradient(RBM & rbm);

    // 確保済みの勾配ベクトルを0初期化
    void initGradient();

    // データ平均ベクトルを初期化
    void initDataMean(RBMBase & rbm) { initDataMean(reinterpret_cast<RBM &>(rbm)); }
    void initDataMean(RBM & rbm);

    // 確保済みデータ平均ベクトルを初期化
    void initDataMean();

    // サンプル平均ベクトルを初期化
    void initSampleMean(RBMBase & rbm) { initSampleMean(reinterpret_cast<RBM &>(rbm)); }
    void initSampleMean(RBM & rbm);

    // 確保済みサンプル平均ベクトルを初期化
    void initSampleMean();

    // 学習
    void train(RBMBase & rbm, std::vector<std::vector<double>> & dataset) { train(reinterpret_cast<RBM &>(rbm), dataset); }
    void train(RBM & rbm, std::vector<std::vector<double>> & dataset);

    // 1回だけ学習
    void trainOnce(RBMBase & rbm, std::vector<std::vector<double>> & dataset) { trainOnce(reinterpret_cast<RBM &>(rbm), dataset); }
    void trainOnce(RBM & rbm, std::vector<std::vector<double>> & dataset);

    // CD計算
    void calcContrastiveDivergence(RBMBase & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) { calcContrastiveDivergence(reinterpret_cast<RBM &>(rbm), dataset, data_indexes); }
    void calcContrastiveDivergence(RBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes);

    // データ平均の計算
    void calcDataMean(RBMBase & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) { calcDataMean(reinterpret_cast<RBM &>(rbm), dataset, data_indexes); }
    void calcDataMean(RBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes);

    // サンプル平均の計算
    void calcSampleMean(RBMBase & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) { calcSampleMean(reinterpret_cast<RBM &>(rbm), dataset, data_indexes); }
    void calcSampleMean(RBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes);

    // 勾配の計算
    void calcGradient(RBMBase & rbm, std::vector<int> & data_indexes) { calcGradient(reinterpret_cast<RBM &>(rbm), data_indexes); }
    void calcGradient(RBM & rbm, std::vector<int> & data_indexes);

    // モーメンタム更新
    void updateMomentum(RBMBase & rbm) { updateMomentum(reinterpret_cast<RBM &>(rbm)); }
    void updateMomentum(RBM & rbm);

    // 勾配更新
    void updateParams(RBMBase & rbm) { updateParams(reinterpret_cast<RBM &>(rbm)); }
    void updateParams(RBM & rbm);
};

