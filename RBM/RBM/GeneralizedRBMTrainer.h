#pragma once
#include "RBMTrainerBase.h"
#include <Eigen/Core>
#include <vector>

class GeneralizedRBM;

class GeneralizedRBMTrainer : RBMTrainerBase {
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
    GeneralizedRBMTrainer();
    GeneralizedRBMTrainer(RBMBase & rbm) { GeneralizedRBMTrainer(reinterpret_cast<GeneralizedRBM &>(rbm)); }
    GeneralizedRBMTrainer(GeneralizedRBM & rbm);
    ~GeneralizedRBMTrainer();

    // モーメンタムベクトル初期化
    void initMomentum(RBMBase & rbm) { initMomentum(reinterpret_cast<GeneralizedRBM &>(rbm)); }
    void initMomentum(GeneralizedRBM & rbm);

    // 確保済みのモーメンタムベクトルを0初期化
    void initMomentum();

    // 勾配ベクトル初期化
    void initGradient(RBMBase & rbm) { initGradient(reinterpret_cast<GeneralizedRBM &>(rbm)); }
    void initGradient(GeneralizedRBM & rbm);

    // 確保済みの勾配ベクトルを0初期化
    void initGradient();

    // データ平均ベクトルを初期化
    void initDataMean(RBMBase & rbm) { initDataMean(reinterpret_cast<GeneralizedRBM &>(rbm)); }
    void initDataMean(GeneralizedRBM & rbm);

    // 確保済みデータ平均ベクトルを初期化
    void initDataMean();

    // サンプル平均ベクトルを初期化
    void initSampleMean(RBMBase & rbm) { initSampleMean(reinterpret_cast<GeneralizedRBM &>(rbm)); }
    void initSampleMean(GeneralizedRBM & rbm);

    // 確保済みサンプル平均ベクトルを初期化
    void initSampleMean();

    // 学習
    void train(RBMBase & rbm, std::vector<std::vector<double>> & dataset) { train(reinterpret_cast<GeneralizedRBM &>(rbm), dataset); }
    void train(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset);

    // 1回だけ学習
    void trainOnce(RBMBase & rbm, std::vector<std::vector<double>> & dataset) { trainOnce(reinterpret_cast<GeneralizedRBM &>(rbm), dataset); }
    void trainOnce(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset);

    // CD計算
    void calcContrastiveDivergence(RBMBase & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) { calcContrastiveDivergence(reinterpret_cast<GeneralizedRBM &>(rbm), dataset, data_indexes); }
    void calcContrastiveDivergence(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes);

    // データ平均の計算
    void calcDataMean(RBMBase & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) { calcDataMean(reinterpret_cast<GeneralizedRBM &>(rbm), dataset, data_indexes); }
    void calcDataMean(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes);

    // サンプル平均の計算
    void calcSampleMean(RBMBase & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) { calcSampleMean(reinterpret_cast<GeneralizedRBM &>(rbm), dataset, data_indexes); }
    void calcSampleMean(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes);

    // 勾配の計算
    void calcGradient(RBMBase & rbm, std::vector<int> & data_indexes) { calcGradient(reinterpret_cast<GeneralizedRBM &>(rbm), data_indexes); }
    void calcGradient(GeneralizedRBM & rbm, std::vector<int> & data_indexes);

    // モーメンタム更新
    void updateMomentum(RBMBase & rbm) { updateMomentum(reinterpret_cast<GeneralizedRBM &>(rbm)); }
    void updateMomentum(GeneralizedRBM & rbm);

    // 勾配更新
    void updateParams(RBMBase & rbm) { updateParams(reinterpret_cast<GeneralizedRBM &>(rbm)); }
    void updateParams(GeneralizedRBM & rbm);
};

