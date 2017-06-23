#pragma once
#include <Eigen/Core>
#include <vector>

class RBM;

class RBMTrainer{
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
    RBMTrainer(RBM *rbm);
    ~RBMTrainer();

    // モーメンタムベクトル初期化
    void initMomentum(RBM *rbm);

    // 確保済みのモーメンタムベクトルを0初期化
    void initMomentum();

    // 勾配ベクトル初期化
    void initGradient(RBM *rbm);

    // 確保済みの勾配ベクトルを0初期化
    void initGradient();

    // データ平均ベクトルを初期化
    void initDataMean(RBM *rbm);

    // 確保済みデータ平均ベクトルを初期化
    void initDataMean();

    // サンプル平均ベクトルを初期化
    void initSampleMean(RBM *rbm);

    // 確保済みサンプル平均ベクトルを初期化
    void initSampleMean();

    // 学習
    void train(RBM *rbm, std::vector<std::vector<double>> & dataset);

    // 1回だけ学習
    void trainOnce(RBM *rbm, std::vector<std::vector<double>> & dataset);

    // CD計算
    void calcContrastiveDivergence(RBM *rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes);

    // データ平均の計算
    void calcDataMean(RBM *rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes);

    // サンプル平均の計算
    void calcSampleMean(RBM *rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes);

    // 勾配の計算
    void calcGradient(RBM *rbm, std::vector<int> & data_indexes);

    // モーメンタム更新
    void updateMomentum(RBM *rbm);

    // 勾配更新
    void updateParams(RBM *rbm);


};

