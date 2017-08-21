#pragma once
#include "RBMTrainerBase.h"
#include <Eigen/Core>
#include <vector>

class GBRBM;

class GBRBMTrainer : RBMTrainerBase {
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


public:
    int epoch = 0;
    int batchSize = 1;
    int cdk = 0;
    double learningRate = 0.01;
    double momentumRate = 0.9;

public:
    GBRBMTrainer();
    GBRBMTrainer(RBMBase & rbm) { GBRBMTrainer(reinterpret_cast<GBRBM &>(rbm)); }
    GBRBMTrainer(GBRBM & rbm);
    ~GBRBMTrainer();

    // モーメンタムベクトル初期化
    void initMomentum(RBMBase & rbm) { initMomentum(reinterpret_cast<GBRBM &>(rbm)); }
    void initMomentum(GBRBM & rbm);

    // 確保済みのモーメンタムベクトルを0初期化
    void initMomentum();

    // 勾配ベクトル初期化
    void initGradient(RBMBase & rbm) { initGradient(reinterpret_cast<GBRBM &>(rbm)); }
    void initGradient(GBRBM & rbm);

    // 確保済みの勾配ベクトルを0初期化
    void initGradient();

    // データ平均ベクトルを初期化
    void initDataMean(RBMBase & rbm) { initDataMean(reinterpret_cast<GBRBM &>(rbm)); }
    void initDataMean(GBRBM & rbm);

    // 確保済みデータ平均ベクトルを初期化
    void initDataMean();

    // サンプル平均ベクトルを初期化
    void initRBMExpected(RBMBase & rbm) { initRBMExpected(reinterpret_cast<GBRBM &>(rbm)); }
    void initRBMExpected(GBRBM & rbm);

    // 確保済みサンプル平均ベクトルを初期化
    void initRBMExpected();

    // 学習
    void train(RBMBase & rbm, std::vector<std::vector<double>> & dataset) { train(reinterpret_cast<GBRBM &>(rbm), dataset); }
    void train(GBRBM & rbm, std::vector<std::vector<double>> & dataset);

    // 1回だけ学習
    void trainOnce(RBMBase & rbm, std::vector<std::vector<double>> & dataset) { trainOnce(reinterpret_cast<GBRBM &>(rbm), dataset); }
    void trainOnce(GBRBM & rbm, std::vector<std::vector<double>> & dataset);

    // CD計算
    void calcContrastiveDivergence(RBMBase & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) { calcContrastiveDivergence(reinterpret_cast<GBRBM &>(rbm), dataset, data_indexes); }
    void calcContrastiveDivergence(GBRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes);

    // データ平均の計算
    void calcDataMean(RBMBase & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) { calcDataMean(reinterpret_cast<GBRBM &>(rbm), dataset, data_indexes); }
    void calcDataMean(GBRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes);

    // サンプル平均の計算
    void calcRBMExpectedCD(RBMBase & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) { calcRBMExpectedCD(reinterpret_cast<GBRBM &>(rbm), dataset, data_indexes); }
    void calcRBMExpectedCD(GBRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes);

    // 勾配の計算
    void calcGradient(RBMBase & rbm, std::vector<int> & data_indexes) { calcGradient(reinterpret_cast<GBRBM &>(rbm), data_indexes); }
    void calcGradient(GBRBM & rbm, std::vector<int> & data_indexes);

    // モーメンタム更新
    void updateMomentum(RBMBase & rbm) { updateMomentum(reinterpret_cast<GBRBM &>(rbm)); }
    void updateMomentum(GBRBM & rbm);

    // 勾配更新
    void updateParams(RBMBase & rbm) { updateParams(reinterpret_cast<GBRBM &>(rbm)); }
    void updateParams(GBRBM & rbm);
};

