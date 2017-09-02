#include "ConditionalRBMTrainer.h"
#include "ConditionalRBM.h"
#include "ConditionalRBMSampler.h"
#include <vector>
#include <numeric>
#include <random>

ConditionalRBMTrainer::ConditionalRBMTrainer()
{
}


ConditionalRBMTrainer::~ConditionalRBMTrainer()
{
}


ConditionalRBMTrainer::ConditionalRBMTrainer(ConditionalRBM & rbm) {
    initMomentum(rbm);
    initGradient(rbm);
    initDataMean(rbm);
    initRBMExpected(rbm);
}

void ConditionalRBMTrainer::initMomentum(ConditionalRBM & rbm) {
    momentum.vBias.setConstant(rbm.getVisibleSize(), 0.0);
    momentum.hBias.setConstant(rbm.getHiddenSize(), 0.0);
    momentum.hvWeight.setConstant(rbm.getHiddenSize(), rbm.getVisibleSize(), 0.0);
    //momentum.vxWeight.setConstant(rbm.getVisibleSize(), rbm.getCondSize(), 0.0);
    momentum.xhWeight.setConstant(rbm.getCondSize(), rbm.getHiddenSize(), 0.0);
}

void ConditionalRBMTrainer::initMomentum() {
    momentum.vBias.setConstant(0.0);
    momentum.hBias.setConstant(0.0);
    momentum.hvWeight.setConstant(0.0);
    //momentum.vxWeight.setConstant(0.0);
    momentum.xhWeight.setConstant(0.0);
}

void ConditionalRBMTrainer::initGradient(ConditionalRBM & rbm) {
    gradient.vBias.setConstant(rbm.getVisibleSize(), 0.0);
    gradient.hBias.setConstant(rbm.getHiddenSize(), 0.0);
    gradient.hvWeight.setConstant(rbm.getHiddenSize(), rbm.getVisibleSize(), 0.0);
    //gradient.vxWeight.setConstant(rbm.getVisibleSize(), rbm.getCondSize(), 0.0);
    gradient.xhWeight.setConstant(rbm.getCondSize(), rbm.getHiddenSize(), 0.0);
}

void ConditionalRBMTrainer::initGradient() {
    gradient.vBias.setConstant(0.0);
    gradient.hBias.setConstant(0.0);
    gradient.hvWeight.setConstant(0.0);
    //gradient.vxWeight.setConstant(0.0);
    gradient.xhWeight.setConstant(0.0);
}

void ConditionalRBMTrainer::initDataMean(ConditionalRBM & rbm) {
	dataMean.vBias.setConstant(rbm.getVisibleSize(), 0.0);
	dataMean.hBias.setConstant(rbm.getHiddenSize(), 0.0);
	dataMean.hvWeight.setConstant(rbm.getHiddenSize(), rbm.getVisibleSize(), 0.0);
	//dataMean.vxWeight.setConstant(rbm.getVisibleSize(), rbm.getCondSize(), 0.0);
	dataMean.xhWeight.setConstant(rbm.getCondSize(), rbm.getHiddenSize(), 0.0);
}

void ConditionalRBMTrainer::initDataMean() {
	dataMean.vBias.setConstant(0.0);
	dataMean.hBias.setConstant(0.0);
	dataMean.hvWeight.setConstant(0.0);
	//dataMean.vxWeight.setConstant(0.0);
	dataMean.xhWeight.setConstant(0.0);
}

void ConditionalRBMTrainer::initRBMExpected(ConditionalRBM & rbm) {
	rbmExpected.vBias.setConstant(rbm.getVisibleSize(), 0.0);
	rbmExpected.hBias.setConstant(rbm.getHiddenSize(), 0.0);
	rbmExpected.hvWeight.setConstant(rbm.getHiddenSize(), rbm.getVisibleSize(), 0.0);
	//rbmExpected.vxWeight.setConstant(rbm.getVisibleSize(), rbm.getCondSize(), 0.0);
	rbmExpected.xhWeight.setConstant(rbm.getCondSize(), rbm.getHiddenSize(), 0.0);
}

void ConditionalRBMTrainer::initRBMExpected() {
	rbmExpected.vBias.setConstant(0.0);
	rbmExpected.hBias.setConstant(0.0);
	rbmExpected.hvWeight.setConstant(0.0);
	//rbmExpected.vxWeight.setConstant(0.0);
	rbmExpected.xhWeight.setConstant(0.0);
}

void ConditionalRBMTrainer::train(ConditionalRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<std::vector<double>> & cond_dataset) {
    for (int e = 0; e < epoch; e++) {
        trainOnce(rbm, dataset, cond_dataset);
    }
}


// 1回だけ学習
void ConditionalRBMTrainer::trainOnce(ConditionalRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<std::vector<double>> & cond_dataset) {

    // データインデックス集合
    std::vector<int> data_indexes(dataset.size());

    // ミニバッチ学習のためにデータインデックスをシャッフルする
    std::iota(data_indexes.begin(), data_indexes.end(), 0);
    std::shuffle(data_indexes.begin(), data_indexes.end(), std::mt19937());

    // ミニバッチ
    // バッチサイズの確認
    int batch_size = this->batchSize < dataset.size() ? this->batchSize : dataset.size();
    // ミニバッチ学習に使うデータのインデックス集合
    std::vector<int> minibatch_indexes(batch_size);
    std::copy(data_indexes.begin(), data_indexes.begin() + batch_size, minibatch_indexes.begin());

    // Contrastive Divergence
    calcContrastiveDivergence(rbm, dataset, cond_dataset, minibatch_indexes);

    // モーメンタムの更新
    updateMomentum(rbm);

    // 勾配の更新
    updateParams(rbm);

}

void ConditionalRBMTrainer::calcContrastiveDivergence(ConditionalRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<std::vector<double>> & cond_dataset, std::vector<int> & data_indexes) {
    // データ平均の計算
    calcDataMean(rbm, dataset, cond_dataset, data_indexes);

    // サンプル平均の計算(CD)
    calcRBMExpected(rbm, dataset, cond_dataset, data_indexes);

    // 勾配計算
    calcGradient(rbm, data_indexes);
}

void ConditionalRBMTrainer::calcDataMean(ConditionalRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<std::vector<double>> & cond_dataset, std::vector<int> & data_indexes) {
    // 0埋め初期化
    initDataMean();

    for (auto & n : data_indexes) {
        auto & data = dataset[n];
        auto & cond_data = cond_dataset[n];
        Eigen::VectorXd vect = Eigen::Map<Eigen::VectorXd>(data.data(), data.size());
        rbm.nodes.v = vect;
        Eigen::VectorXd cond_vect = Eigen::Map<Eigen::VectorXd>(cond_data.data(), cond_data.size());
        rbm.nodes.x = cond_vect;

        dataMean.vBias += vect;

        for (int h_counter = 0; h_counter < rbm.getHiddenSize(); h_counter++) {
            dataMean.hBias(h_counter) += rbm.actHidJ(h_counter);

			// 使いまわせるものは使いまわせ
			for (int v_counter = 0; v_counter < rbm.getVisibleSize(); v_counter++) {
				dataMean.hvWeight(h_counter, v_counter) += dataMean.hBias(h_counter) *  dataMean.vBias(v_counter);
			}
		
			for (int x_counter = 0; x_counter < rbm.getCondSize(); x_counter++) {
				dataMean.xhWeight(x_counter, h_counter) += dataMean.hBias(h_counter) *  cond_vect(x_counter);
			}
		}

    }

    dataMean.vBias /= static_cast<double>(data_indexes.size());
    dataMean.hBias /= static_cast<double>(data_indexes.size());
	dataMean.xhWeight /= static_cast<double>(data_indexes.size());
	dataMean.hvWeight /= static_cast<double>(data_indexes.size());
}

void ConditionalRBMTrainer::calcRBMExpected(ConditionalRBM & rbm, std::vector<std::vector<double>> & dataset, std::vector<std::vector<double>> & cond_dataset, std::vector<int> & data_indexes) {
    // 0埋め初期化
    initRBMExpected();

    for (auto & n : data_indexes) {
        auto & data = dataset[n];
        auto & cond_data = cond_dataset[n];

        Eigen::VectorXd vect = Eigen::Map<Eigen::VectorXd>(data.data(), data.size());
        Eigen::VectorXd cond_vect = Eigen::Map<Eigen::VectorXd>(cond_data.data(), cond_data.size());

        // ConditionalRBMの初期値設定
        rbm.nodes.v = vect;
        rbm.nodes.x = cond_vect;

        for (int h_counter = 0; h_counter < rbm.getHiddenSize(); h_counter++) {
            rbm.nodes.h(h_counter) = rbm.actHidJ(h_counter);
        }

        // CD-K
        ConditionalRBMSampler sampler;
        for (int x_counter = 0; x_counter < cdk; x_counter++) {
            sampler.updateByBlockedGibbsSamplingVisible(rbm);
            sampler.updateByBlockedGibbsSamplingHidden(rbm);
        }

        // 結果を格納
        rbmExpected.vBias += rbm.nodes.v;
        rbmExpected.hBias += rbm.nodes.h;

		for (int h_counter = 0; h_counter < rbm.getHiddenSize(); h_counter++) {
			// 計算使いまわせるところは使いまわします
			for (int x_counter = 0; x_counter < rbm.getCondSize(); x_counter++) {
				rbmExpected.xhWeight(x_counter, h_counter) = cond_vect(x_counter) * rbmExpected.hBias(h_counter);
			}

			for (int v_counter = 0; v_counter < rbm.getVisibleSize(); v_counter++) {
				rbmExpected.hvWeight(h_counter, v_counter) = rbmExpected.hBias(h_counter) * rbmExpected.vBias(v_counter);
			}
		}
	}

	rbmExpected.vBias /= static_cast<double>(data_indexes.size());
	rbmExpected.hBias /= static_cast<double>(data_indexes.size());
	rbmExpected.hvWeight /= static_cast<double>(data_indexes.size());
	//rbmExpected.vxWeight /= static_cast<double>(data_indexes.size());
	rbmExpected.xhWeight /= static_cast<double>(data_indexes.size());
}

// 勾配の計算
void ConditionalRBMTrainer::calcGradient(ConditionalRBM & rbm, std::vector<int> & data_indexes) {
    // 勾配ベクトルリセット
    initGradient();

    for (int v_counter = 0; v_counter < rbm.getVisibleSize(); v_counter++) {
        gradient.vBias(v_counter) = dataMean.vBias(v_counter) - rbmExpected.vBias(v_counter);

        for (int h_counter = 0; h_counter < rbm.getHiddenSize(); h_counter++) {
            gradient.hvWeight(h_counter, v_counter) = dataMean.hvWeight(h_counter, v_counter) - rbmExpected.hvWeight(h_counter, v_counter);
        }
    }

    for (int h_counter = 0; h_counter < rbm.getHiddenSize(); h_counter++) {
        gradient.hBias(h_counter) = dataMean.hBias(h_counter) - rbmExpected.hBias(h_counter);

        for (int x_counter = 0; x_counter < rbm.getCondSize(); x_counter++) {
			gradient.xhWeight(x_counter, h_counter) = dataMean.xhWeight(x_counter, h_counter) - rbmExpected.xhWeight(x_counter, h_counter);
        }
    }
}

void ConditionalRBMTrainer::updateMomentum(ConditionalRBM & rbm) {
    for (int v_counter = 0; v_counter < rbm.getVisibleSize(); v_counter++) {
        momentum.vBias(v_counter) = momentumRate * momentum.vBias(v_counter) + learningRate * gradient.vBias(v_counter);

        for (int h_counter = 0; h_counter < rbm.getHiddenSize(); h_counter++) {
            momentum.hvWeight(h_counter, v_counter) = momentumRate * momentum.hvWeight(h_counter, v_counter) + learningRate * gradient.hvWeight(h_counter, v_counter);
        }
    }

    for (int h_counter = 0; h_counter < rbm.getHiddenSize(); h_counter++) {
        momentum.hBias(h_counter) = momentumRate * momentum.hBias(h_counter) + learningRate * gradient.hBias(h_counter);
       
        for (int x_counter = 0; x_counter < rbm.getCondSize(); x_counter++) {
            momentum.xhWeight(x_counter, h_counter) = momentumRate * momentum.xhWeight(x_counter, h_counter) + learningRate * gradient.xhWeight(x_counter, h_counter);
        }
    }
}

// パラメータの更新
void ConditionalRBMTrainer::updateParams(ConditionalRBM & rbm) {

    for (int v_counter = 0; v_counter < rbm.getVisibleSize(); v_counter++) {
        rbm.params.b(v_counter) += momentum.vBias(v_counter);

        for (int h_counter = 0; h_counter < rbm.getHiddenSize(); h_counter++) {
            rbm.params.hvW(h_counter, v_counter) += momentum.hvWeight(h_counter, v_counter);
        }
    }

    for (int h_counter = 0; h_counter < rbm.getHiddenSize(); h_counter++) {
        rbm.params.c(h_counter) += momentum.hBias(h_counter);

        for (int x_counter = 0; x_counter < rbm.getCondSize(); x_counter++) {
            rbm.params.xhW(x_counter, h_counter) += momentum.xhWeight(x_counter, h_counter);
        }
    }
}
