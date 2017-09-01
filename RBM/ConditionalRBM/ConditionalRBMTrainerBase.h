#pragma once
#include <vector>
class ConditionalRBMBase;

class ConditionalRBMTrainerBase {
    // モーメンタムベクトル初期化
    virtual void initMomentum(ConditionalRBMBase & rbm) = 0;

    // 確保済みのモーメンタムベクトルを0初期化
    virtual void initMomentum() = 0;

    // 勾配ベクトル初期化
    virtual void initGradient(ConditionalRBMBase & rbm) = 0;

    // 確保済みの勾配ベクトルを0初期化
    virtual void initGradient() = 0;

    // データ平均ベクトルを初期化
    virtual void initDataMean(ConditionalRBMBase & rbm) = 0;

    // 確保済みデータ平均ベクトルを初期化
    virtual void initDataMean() = 0;

    // サンプル平均ベクトルを初期化
    virtual void initRBMExpected(ConditionalRBMBase & rbm) = 0;

    // 確保済みサンプル平均ベクトルを初期化
    virtual void initRBMExpected() = 0;

    // 学習
    virtual void train(ConditionalRBMBase & rbm, std::vector<std::vector<double>> & dataset, std::vector<std::vector<double>> & cond_dataset) = 0;

    // 1回だけ学習
    virtual void trainOnce(ConditionalRBMBase & rbm, std::vector<std::vector<double>> & dataset, std::vector<std::vector<double>> & cond_dataset) = 0;

    // CD計算
    virtual void calcContrastiveDivergence(ConditionalRBMBase & rbm, std::vector<std::vector<double>> & dataset, std::vector<std::vector<double>> & cond_dataset, std::vector<int> & data_indexes) = 0;

    // データ平均の計算
    virtual void calcDataMean(ConditionalRBMBase & rbm, std::vector<std::vector<double>> & dataset, std::vector<std::vector<double>> & cond_dataset, std::vector<int> & data_indexes) = 0;

    // サンプル平均の計算
    virtual void calcRBMExpected(ConditionalRBMBase & rbm, std::vector<std::vector<double>> & dataset, std::vector<std::vector<double>> & cond_dataset, std::vector<int> & data_indexes) = 0;

    // 勾配の計算
    virtual void calcGradient(ConditionalRBMBase & rbm, std::vector<int> & data_indexes) = 0;

    // モーメンタム更新
    virtual void updateMomentum(ConditionalRBMBase & rbm) = 0;

    // 勾配更新
    virtual void updateParams(ConditionalRBMBase & rbm) = 0;
};