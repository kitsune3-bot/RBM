#pragma once
#include "RBMTrainerBase.h"
#include "Eigen/Core"
#include <vector>

class GeneralizedSparseRBM;

class GeneralizedSparseRBMTrainer : RBMTrainerBase {
	struct Momentum {
		Eigen::VectorXd vBias;
		Eigen::VectorXd hBias;
		Eigen::MatrixXd weight;
		Eigen::VectorXd hSparse;
	};

	struct Gradient {
		Eigen::VectorXd vBias;
		Eigen::VectorXd hBias;
		Eigen::MatrixXd weight;
		Eigen::VectorXd hSparse;
	};

	struct DataMean {
		Eigen::VectorXd vBias;
		Eigen::VectorXd hBias;
		Eigen::MatrixXd weight;
		Eigen::VectorXd hSparse;
	};

	struct RBMExpected {
		Eigen::VectorXd vBias;
		Eigen::VectorXd hBias;
		Eigen::MatrixXd weight;
		Eigen::VectorXd hSparse;
	};

private:
	Momentum momentum;
	Gradient gradient;
	DataMean dataMean;
	RBMExpected rbmexpected;
	int _trainCount = 0;


public:
	int epoch = 0;
	int batchSize = 1;
	int cdk = 0;
	double learningRate = 0.01;
	double momentumRate = 0.9;

public:
	GeneralizedSparseRBMTrainer();
	GeneralizedSparseRBMTrainer(RBMBase & rbm) { GeneralizedSparseRBMTrainer(reinterpret_cast<GeneralizedSparseRBM &>(rbm)); }
	GeneralizedSparseRBMTrainer(GeneralizedSparseRBM & rbm);
	~GeneralizedSparseRBMTrainer();

	// モーメンタムベクトル初期化
	void initMomentum(RBMBase & rbm) { initMomentum(reinterpret_cast<GeneralizedSparseRBM &>(rbm)); }
	void initMomentum(GeneralizedSparseRBM & rbm);

	// 確保済みのモーメンタムベクトルを0初期化
	void initMomentum();

	// 勾配ベクトル初期化
	void initGradient(RBMBase & rbm) { initGradient(reinterpret_cast<GeneralizedSparseRBM &>(rbm)); }
	void initGradient(GeneralizedSparseRBM & rbm);

	// 確保済みの勾配ベクトルを0初期化
	void initGradient();

	// データ平均ベクトルを初期化
	void initDataMean(RBMBase & rbm) { initDataMean(reinterpret_cast<GeneralizedSparseRBM &>(rbm)); }
	void initDataMean(GeneralizedSparseRBM & rbm);

	// 確保済みデータ平均ベクトルを初期化
	void initDataMean();

	// サンプル平均ベクトルを初期化
	void initRBMExpected(RBMBase & rbm) { initRBMExpected(reinterpret_cast<GeneralizedSparseRBM &>(rbm)); }
	void initRBMExpected(GeneralizedSparseRBM & rbm);

	// 確保済みサンプル平均ベクトルを初期化
	void initRBMExpected();

	// 学習
	void train(RBMBase & rbm, std::vector<std::vector<double>> & dataset) { train(reinterpret_cast<GeneralizedSparseRBM &>(rbm), dataset); }
	void train(GeneralizedSparseRBM & rbm, std::vector<std::vector<double>> & dataset);

	void trainCD(GeneralizedSparseRBM & rbm, std::vector<std::vector<double>> & dataset);
	void trainExact(GeneralizedSparseRBM & rbm, std::vector<std::vector<double>> & dataset);


	// 1回だけ学習
	void trainOnce(RBMBase & rbm, std::vector<std::vector<double>> & dataset) { trainOnce(reinterpret_cast<GeneralizedSparseRBM &>(rbm), dataset); }
	void trainOnce(GeneralizedSparseRBM & rbm, std::vector<std::vector<double>> & dataset);

	void trainOnceCD(GeneralizedSparseRBM & rbm, std::vector<std::vector<double>> & dataset);
	void trainOnceExact(GeneralizedSparseRBM & rbm, std::vector<std::vector<double>> & dataset);


	// CD計算
	void calcContrastiveDivergence(RBMBase & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) { calcContrastiveDivergence(reinterpret_cast<GeneralizedSparseRBM &>(rbm), dataset, data_indexes); }
	void calcContrastiveDivergence(GeneralizedSparseRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes);

	// CD計算
	void calcExact(RBMBase & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) { calcExact(reinterpret_cast<GeneralizedSparseRBM &>(rbm), dataset, data_indexes); }
	void calcExact(GeneralizedSparseRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes);

	// データ平均の計算
	void calcDataMean(RBMBase & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) { calcDataMean(reinterpret_cast<GeneralizedSparseRBM &>(rbm), dataset, data_indexes); }
	void calcDataMean(GeneralizedSparseRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes);

	// サンプル平均の計算
	void calcRBMExpectedCD(RBMBase & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) { calcRBMExpectedCD(reinterpret_cast<GeneralizedSparseRBM &>(rbm), dataset, data_indexes); }
	void calcRBMExpectedCD(GeneralizedSparseRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes);

	// サンプル平均の計算
	void calcRBMExpectedExact(RBMBase & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) { calcRBMExpectedExact(reinterpret_cast<GeneralizedSparseRBM &>(rbm), dataset, data_indexes); }
	void calcRBMExpectedExact(GeneralizedSparseRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes);


	// 勾配の計算
	void calcGradient(RBMBase & rbm, std::vector<int> & data_indexes) { calcGradient(reinterpret_cast<GeneralizedSparseRBM &>(rbm), data_indexes); }
	void calcGradient(GeneralizedSparseRBM & rbm, std::vector<int> & data_indexes);

	// モーメンタム更新
	void updateMomentum(RBMBase & rbm) { updateMomentum(reinterpret_cast<GeneralizedSparseRBM &>(rbm)); }
	void updateMomentum(GeneralizedSparseRBM & rbm);

	// 勾配更新
	void updateParams(RBMBase & rbm) { updateParams(reinterpret_cast<GeneralizedSparseRBM &>(rbm)); }
	void updateParams(GeneralizedSparseRBM & rbm);

	// 対数尤度関数
	double logLikeliHood(GeneralizedSparseRBM & rbm, std::vector<std::vector<double>> & dataset);
};

