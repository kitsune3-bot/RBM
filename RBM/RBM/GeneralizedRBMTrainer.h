#pragma once
#include "RBMTrainerBase.h"
#include "Eigen/Core"
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
        Eigen::VectorXd vBias;
        Eigen::VectorXd hBias;
		Eigen::MatrixXd weight;
	};

    struct RBMExpected {
		Eigen::VectorXd vBias;
		Eigen::VectorXd hBias;
		Eigen::MatrixXd weight;
	};

private:
    Momentum momentum;
    Gradient gradient;
    DataMean dataMean;
    RBMExpected rbmexpected;


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
    void initRBMExpected(RBMBase & rbm) { initRBMExpected(reinterpret_cast<GeneralizedRBM &>(rbm)); }
    void initRBMExpected(GeneralizedRBM & rbm);

    // 確保済みサンプル平均ベクトルを初期化
    void initRBMExpected();

    // 学習
    void train(RBMBase & rbm, std::vector<std::vector<double>> & dataset) { train(reinterpret_cast<GeneralizedRBM &>(rbm), dataset); }
    void train(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset);

	void trainCD(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset);
	void trainExact(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset);


    // 1回だけ学習
    void trainOnce(RBMBase & rbm, std::vector<std::vector<double>> & dataset) { trainOnce(reinterpret_cast<GeneralizedRBM &>(rbm), dataset); }
    void trainOnce(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset);

	void trainOnceCD(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset);
	void trainOnceExact(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset);


    // CD計算
    void calcContrastiveDivergence(RBMBase & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) { calcContrastiveDivergence(reinterpret_cast<GeneralizedRBM &>(rbm), dataset, data_indexes); }
    void calcContrastiveDivergence(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes);

	// CD計算
	void calcExact(RBMBase & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) { calcExact(reinterpret_cast<GeneralizedRBM &>(rbm), dataset, data_indexes); }
	void calcExact(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes);

    // データ平均の計算
    void calcDataMean(RBMBase & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) { calcDataMean(reinterpret_cast<GeneralizedRBM &>(rbm), dataset, data_indexes); }
    void calcDataMean(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes);

    // サンプル平均の計算
    void calcRBMExpectedCD(RBMBase & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) { calcRBMExpectedCD(reinterpret_cast<GeneralizedRBM &>(rbm), dataset, data_indexes); }
    void calcRBMExpectedCD(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes);

	// サンプル平均の計算
	void calcRBMExpectedExact(RBMBase & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) { calcRBMExpectedExact(reinterpret_cast<GeneralizedRBM &>(rbm), dataset, data_indexes); }
	void calcRBMExpectedExact(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes);


    // 勾配の計算
    void calcGradient(RBMBase & rbm, std::vector<int> & data_indexes) { calcGradient(reinterpret_cast<GeneralizedRBM &>(rbm), data_indexes); }
    void calcGradient(GeneralizedRBM & rbm, std::vector<int> & data_indexes);

    // モーメンタム更新
    void updateMomentum(RBMBase & rbm) { updateMomentum(reinterpret_cast<GeneralizedRBM &>(rbm)); }
    void updateMomentum(GeneralizedRBM & rbm);

    // 勾配更新
    void updateParams(RBMBase & rbm) { updateParams(reinterpret_cast<GeneralizedRBM &>(rbm)); }
    void updateParams(GeneralizedRBM & rbm);

	// 対数尤度関数
	double logLikeliHood(GeneralizedRBM & rbm, std::vector<std::vector<double>> & dataset);
};

