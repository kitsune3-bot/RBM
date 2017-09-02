#pragma once
#include "ConditionalRBMTrainerBase.h"
#include "Eigen/Core"
#include <vector>

class ConditionalRBM;

class ConditionalRBMTrainer : ConditionalRBMTrainerBase {
    struct Momentum {
        Eigen::VectorXd vBias;
        Eigen::VectorXd hBias;
        Eigen::MatrixXd hvWeight;
        Eigen::MatrixXd xhWeight;
    };

    struct Gradient {
        Eigen::VectorXd vBias;
        Eigen::VectorXd hBias;
		Eigen::MatrixXd hvWeight;
		Eigen::MatrixXd xhWeight;
	};

    struct DataMean {
		Eigen::VectorXd vBias;
		Eigen::VectorXd hBias;
		Eigen::MatrixXd hvWeight;
		Eigen::MatrixXd xhWeight;
	};

    struct SampleMean {
		Eigen::VectorXd vBias;
		Eigen::VectorXd hBias;
		Eigen::MatrixXd hvWeight;
		Eigen::MatrixXd xhWeight;
	};

private:
    Momentum momentum;
    Gradient gradient;
    DataMean dataMean;
    SampleMean rbmExpected;


public:
    int epoch = 0;
    int batchSize = 1;
    int cdk = 0;
    double learningRate = 0.01;
    double momentumRate = 0.9;

public:
    ConditionalRBMTrainer();
    ConditionalRBMTrainer(ConditionalRBMBase & rbm) { ConditionalRBMTrainer(reinterpret_cast<ConditionalRBM &>(rbm)); }
    ConditionalRBMTrainer(ConditionalRBM & rbm);
    ~ConditionalRBMTrainer();

    // モーメンタムベクトル初期化
    void initMomentum(ConditionalRBMBase & rbm) { initMomentum(reinterpret_cast<ConditionalRBM &>(rbm)); }
    void initMomentum(ConditionalRBM & rbm);

    // 確保済みのモーメンタムベクトルを0初期化
    void initMomentum();

    // 勾配ベクトル初期化
    void initGradient(ConditionalRBMBase & rbm) { initGradient(reinterpret_cast<ConditionalRBM &>(rbm)); }
    void initGradient(ConditionalRBM & rbm);

    // 確保済みの勾配ベクトルを0初期化
    void initGradient();

    // データ平均ベクトルを初期化
    void initDataMean(ConditionalRBMBase & rbm) { initDataMean(reinterpret_cast<ConditionalRBM &>(rbm)); }
    void initDataMean(ConditionalRBM & rbm);

    // 確保済みデータ平均ベクトルを初期化
    void initDataMean();

    // サンプル平均ベクトルを初期化
    void initRBMExpected(ConditionalRBMBase & rbm) { initRBMExpected(reinterpret_cast<ConditionalRBM &>(rbm)); }
    void initRBMExpected(ConditionalRBM & rbm);

    // 確保済みサンプル平均ベクトルを初期化
    void initRBMExpected();

    // 学習
    void train(ConditionalRBMBase & rbm, std::vector<std::vector<double>> & dataset, std::vector<std::vector<double>> & cond_dataset) { train(reinterpret_cast<ConditionalRBM &>(rbm), dataset, cond_dataset); }
    void train(ConditionalRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<std::vector<double>> & cond_dataset);

    // 1回だけ学習
    void trainOnce(ConditionalRBMBase & rbm, std::vector<std::vector<double>> & dataset, std::vector<std::vector<double>> & cond_dataset) { trainOnce(reinterpret_cast<ConditionalRBM &>(rbm), dataset, cond_dataset); }
    void trainOnce(ConditionalRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<std::vector<double>> & cond_dataset);

    // CD計算
    void calcContrastiveDivergence(ConditionalRBMBase & rbm, std::vector<std::vector<double>> & dataset, std::vector<std::vector<double>> & cond_dataset, std::vector<int> & data_indexes) { calcContrastiveDivergence(reinterpret_cast<ConditionalRBM &>(rbm), dataset, cond_dataset, data_indexes); }
    void calcContrastiveDivergence(ConditionalRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<std::vector<double>> & cond_dataset, std::vector<int> & data_indexes);

    // データ平均の計算
    void calcDataMean(ConditionalRBMBase & rbm, std::vector<std::vector<double>> & dataset, std::vector<std::vector<double>> & cond_dataset, std::vector<int> & data_indexes) { calcDataMean(reinterpret_cast<ConditionalRBM &>(rbm), dataset, cond_dataset, data_indexes); }
    void calcDataMean(ConditionalRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<std::vector<double>> & cond_dataset, std::vector<int> & data_indexes);

    // サンプル平均の計算
    void calcRBMExpected(ConditionalRBMBase & rbm, std::vector<std::vector<double>> & dataset, std::vector<std::vector<double>> & cond_dataset, std::vector<int> & data_indexes) { calcRBMExpected(reinterpret_cast<ConditionalRBM &>(rbm), dataset, cond_dataset, data_indexes); }
    void calcRBMExpected(ConditionalRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<std::vector<double>> & cond_dataset, std::vector<int> & data_indexes);

    // 勾配の計算
    void calcGradient(ConditionalRBMBase & rbm, std::vector<int> & data_indexes) { calcGradient(reinterpret_cast<ConditionalRBM &>(rbm), data_indexes); }
    void calcGradient(ConditionalRBM & rbm, std::vector<int> & data_indexes);

    // モーメンタム更新
    void updateMomentum(ConditionalRBMBase & rbm) { updateMomentum(reinterpret_cast<ConditionalRBM &>(rbm)); }
    void updateMomentum(ConditionalRBM & rbm);

    // 勾配更新
    void updateParams(ConditionalRBMBase & rbm) { updateParams(reinterpret_cast<ConditionalRBM &>(rbm)); }
    void updateParams(ConditionalRBM & rbm);
};

