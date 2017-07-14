#pragma once
#include "ConditionalRBMTrainerBase.h"
#include "ConditionalGRBM.h"
#include <Eigen/Core>
#include <vector>


class ConditionalGRBMTrainer : ConditionalRBMTrainerBase {
    struct Momentum {
        Eigen::VectorXd vBias;
        Eigen::VectorXd vLambda;
        Eigen::VectorXd hBias;
        Eigen::MatrixXd weight;
        Eigen::MatrixXd vxWeight;
        Eigen::MatrixXd hxWeight;
    };

    struct Gradient {
        Eigen::VectorXd vBias;
        Eigen::VectorXd vLambda;
        Eigen::VectorXd hBias;
        Eigen::MatrixXd weight;
        Eigen::MatrixXd vxWeight;
        Eigen::MatrixXd hxWeight;
    };

    struct DataMean {
        Eigen::VectorXd visible;
        Eigen::VectorXd visible2;  // Gausiann Unit限定 
        Eigen::VectorXd hidden;
        Eigen::VectorXd conditional;
    };

    struct SampleMean {
        Eigen::VectorXd visible;
        Eigen::VectorXd visible2;  // Gausiann Unit限定 
        Eigen::VectorXd hidden;
        Eigen::VectorXd conditional;
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
    ConditionalGRBMTrainer();
    ConditionalGRBMTrainer(ConditionalRBMBase & rbm) { ConditionalGRBMTrainer(reinterpret_cast<ConditionalGRBM &>(rbm)); }

    ConditionalGRBMTrainer(ConditionalGRBM & rbm);
    ~ConditionalGRBMTrainer();

    // モーメンタムベクトル初期化
    void initMomentum(ConditionalRBMBase & rbm) { initMomentum(reinterpret_cast<ConditionalGRBM &>(rbm)); }
    void initMomentum(ConditionalGRBM & rbm);

    // 確保済みのモーメンタムベクトルを0初期化
    void initMomentum();

    // 勾配ベクトル初期化
    void initGradient(ConditionalRBMBase & rbm) { initGradient(reinterpret_cast<ConditionalGRBM &>(rbm)); }
    void initGradient(ConditionalGRBM & rbm);

    // 確保済みの勾配ベクトルを0初期化
    void initGradient();

    // データ平均ベクトルを初期化
    void initDataMean(ConditionalRBMBase & rbm) { initDataMean(reinterpret_cast<ConditionalGRBM &>(rbm)); }
    void initDataMean(ConditionalGRBM & rbm);

    // 確保済みデータ平均ベクトルを初期化
    void initDataMean();

    // サンプル平均ベクトルを初期化
    void initSampleMean(ConditionalRBMBase & rbm) { initSampleMean(reinterpret_cast<ConditionalGRBM &>(rbm)); }
    void initSampleMean(ConditionalGRBM & rbm);

    // 確保済みサンプル平均ベクトルを初期化
    void initSampleMean();

    // 学習
    void train(ConditionalRBMBase & rbm, std::vector<std::vector<double>> & dataset, std::vector<std::vector<double>> & cond_dataset) { train(reinterpret_cast<ConditionalGRBM &>(rbm), dataset, cond_dataset); }
    void train(ConditionalGRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<std::vector<double>> & cond_dataset);

    // 1回だけ学習
    void trainOnce(ConditionalRBMBase & rbm, std::vector<std::vector<double>> & dataset, std::vector<std::vector<double>> & cond_dataset) { trainOnce(reinterpret_cast<ConditionalGRBM &>(rbm), dataset, cond_dataset); }
    void trainOnce(ConditionalGRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<std::vector<double>> & cond_dataset);

    // CD計算
    void calcContrastiveDivergence(ConditionalRBMBase & rbm, std::vector<std::vector<double>> & dataset, std::vector<std::vector<double>> & cond_dataset, std::vector<int> & data_indexes) { calcContrastiveDivergence(reinterpret_cast<ConditionalGRBM &>(rbm), dataset, cond_dataset, data_indexes); }
    void calcContrastiveDivergence(ConditionalGRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<std::vector<double>> & cond_dataset, std::vector<int> & data_indexes);

    // データ平均の計算
    void calcDataMean(ConditionalRBMBase & rbm, std::vector<std::vector<double>> & dataset, std::vector<std::vector<double>> & cond_dataset, std::vector<int> & data_indexes) { calcDataMean(reinterpret_cast<ConditionalGRBM &>(rbm), dataset, cond_dataset, data_indexes); }
    void calcDataMean(ConditionalGRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<std::vector<double>> & cond_dataset, std::vector<int> & data_indexes);

    // サンプル平均の計算
    void calcSampleMean(ConditionalRBMBase & rbm, std::vector<std::vector<double>> & dataset, std::vector<std::vector<double>> & cond_dataset, std::vector<int> & data_indexes) { calcSampleMean(reinterpret_cast<ConditionalGRBM &>(rbm), dataset, cond_dataset, data_indexes); }
    void calcSampleMean(ConditionalGRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<std::vector<double>> & cond_dataset, std::vector<int> & data_indexes);

    // 勾配の計算
    void calcGradient(ConditionalRBMBase & rbm, std::vector<int> & data_indexes) { calcGradient(reinterpret_cast<ConditionalGRBM &>(rbm), data_indexes); }
    void calcGradient(ConditionalGRBM & rbm, std::vector<int> & data_indexes);

    // モーメンタム更新
    void updateMomentum(ConditionalRBMBase & rbm) { updateMomentum(reinterpret_cast<ConditionalGRBM &>(rbm)); }
    void updateMomentum(ConditionalGRBM & rbm);

    // 勾配更新
    void updateParams(ConditionalRBMBase & rbm) { updateParams(reinterpret_cast<ConditionalGRBM &>(rbm)); }
    void updateParams(ConditionalGRBM & rbm);
};

