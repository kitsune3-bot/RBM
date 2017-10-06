#pragma once
#include <vector>
#include <string>
class RBMBase;

class RBMTrainerBase {
    // モーメンタムベクトル初期化
    virtual void initMomentum(RBMBase & rbm) = 0;

    // 確保済みのモーメンタムベクトルを0初期化
    virtual void initMomentum() = 0;

    // 勾配ベクトル初期化
    virtual void initGradient(RBMBase & rbm) = 0;

    // 確保済みの勾配ベクトルを0初期化
    virtual void initGradient() = 0;

    // データ平均ベクトルを初期化
    virtual void initDataMean(RBMBase & rbm) = 0;

    // 確保済みデータ平均ベクトルを初期化
    virtual void initDataMean() = 0;

    // サンプル平均ベクトルを初期化
    virtual void initRBMExpected(RBMBase & rbm) = 0;

    // 確保済みサンプル平均ベクトルを初期化
    virtual void initRBMExpected() = 0;

    // 学習
    virtual void train(RBMBase & rbm, std::vector<std::vector<double>> & dataset) = 0;

    // 1回だけ学習
    virtual void trainOnce(RBMBase & rbm, std::vector<std::vector<double>> & dataset) = 0;

    // CD計算
    virtual void calcContrastiveDivergence(RBMBase & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) = 0;

    // データ平均の計算
    virtual void calcDataMean(RBMBase & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) = 0;

    // CD法でのサンプル平均の計算
    virtual void calcRBMExpectedCD(RBMBase & rbm, std::vector<std::vector<double>> & dataset, std::vector<int> & data_indexes) = 0;

    // 勾配の計算
    virtual void calcGradient(RBMBase & rbm, std::vector<int> & data_indexes) = 0;

    // モーメンタム更新
    virtual void updateMomentum(RBMBase & rbm) = 0;

    // 勾配更新
    virtual void updateParams(RBMBase & rbm) = 0;

	// 学習情報出力(JSON)
	virtual std::string trainInfoJson(RBMBase & rbm) = 0;
};