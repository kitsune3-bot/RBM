#pragma once
#include <fstream>
#include "../Optimizer.h"
#include "GeneralizedRBM.h"

//
// TODO: 最適化アルゴリズムを使い分けるのはフラグ管理するよりテンプレートで分けたほうがパフォーマンス的にはよいと思うけど…
//       まだまだ未熟ですなぁ…
//
template <>
class Optimizer<GeneralizedRBM> {
	struct Moment{
		Eigen::VectorXd vBias;
		Eigen::VectorXd hBias;
		Eigen::MatrixXd weight;
	};

protected:
	int _OptimizeModeFlag;
	int _iteration = 1;
	// momentum
	double _learningRate = 0.001;
	double _momentumRate = 0.9;

	// AdaGrad
	double _muAdaGrad = 0.001;
	double _epsilonAdaGrad = 1E-08;

	// AdaDelta
	double _rhoAdaGrad = 0.95;
	double _epsilonAdaDelta = 1E-06;

	// Adam
	double _alpha = 0.001;
	double _beta1 = 0.9;
	double _beta2 = 0.999;
	double _epsilonAdam = 1E-08;

	Moment moment1st;
	Moment moment2nd;

	// Default
	void _initDefault(GeneralizedRBM & rbm);
	double _getNewParamVBiasDefault(double gradient, int vindex);
	double _getNewParamHBiasDefault(double gradient, int hindex);
	double _getNewParamWeightDefault(double gradient, int vindex, int hindex);

	// Momentum
	void _initMomentum(GeneralizedRBM & rbm);
	double _getNewParamVBiasMomentum(double gradient, int vindex);
	double _getNewParamHBiasMomentum(double gradient, int hindex);
	double _getNewParamWeightMomentum(double gradient, int vindex, int hindex);

	// AdaGrad
	void _initAdaGrad(GeneralizedRBM & rbm);
	double _getNewParamVBiasAdaGrad(double gradient, int vindex);
	double _getNewParamHBiasAdaGrad(double gradient, int hindex);
	double _getNewParamWeightAdaGrad(double gradient, int vindex, int hindex);

	// AdaDelta
	void _initAdaDelta(GeneralizedRBM & rbm);
	double _getNewParamVBiasAdaDelta(double gradient, int vindex);
	double _getNewParamHBiasAdaDelta(double gradient, int hindex);
	double _getNewParamWeightAdaDelta(double gradient, int vindex, int hindex);

	// Adam
	void _initAdam(GeneralizedRBM & rbm);
	double _getNewParamVBiasAdam(double gradient, int vindex);
	double _getNewParamHBiasAdam(double gradient, int hindex);
	double _getNewParamWeightAdam(double gradient, int vindex, int hindex);

	//// Nadam
	//void _initNadam(GeneralizedRBM & rbm);
	//double _getNewParamVBiasNadam(double gradient, int vindex);
	//double _getNewParamHBiasNadam(double gradient, int hindex);
	//double _getNewParamWeightNadam(double gradient, int vindex, int hindex);
public:
	static const int default = 0;
	static const int momentum = 1;
	static const int adaGrad = 2;
	static const int adaDelta = 3;
	static const int adam = 4;
	static const int nadam = 5;

	Optimizer() = default;
	Optimizer(GeneralizedRBM & rbm, int mode_flag);
	~Optimizer() = default;
	void init(GeneralizedRBM & rbm);
	double getNewParamVBias(double gradient, int vindex);
	double getNewParamHBias(double gradient, int hindex);
	double getNewParamWeight(double gradient, int vindex, int hindex);
	// next timestep
	void updateOptimizer();
};

inline Optimizer<GeneralizedRBM>::Optimizer(GeneralizedRBM & rbm, int mode_flag) {
	this->_OptimizeModeFlag = mode_flag;
	this->init(rbm);
}

inline void Optimizer<GeneralizedRBM>::updateOptimizer() {
	this->_iteration++;
}


inline void Optimizer<GeneralizedRBM>::init(GeneralizedRBM & rbm) {
	// TODO: switch文とフラグ管理は嫌だ…
	switch (this->_OptimizeModeFlag) {
	default:
		_initDefault(rbm);
		break;

	case momentum:
		_initMomentum(rbm);
		break;

	case adaGrad:
		_initAdaGrad(rbm);
		break;

	case adaDelta:
		_initAdaDelta(rbm);
		break;

	case adam:
		_initAdam(rbm);
		break;

	//case nadam:
	//	_initNadam(rbm);
	//	break;
	}
}

inline double Optimizer<GeneralizedRBM>::getNewParamVBias(double gradient, int vindex) {
	// TODO: switch文とフラグ管理は嫌だ…
	switch (this->_OptimizeModeFlag) {
	default:
		return _getNewParamVBiasDefault(gradient, vindex);
		break;

	case momentum:
		return _getNewParamVBiasMomentum(gradient, vindex);
		break;

		case adaGrad:
			return _getNewParamVBiasAdaGrad(gradient, vindex);
			break;

		case adaDelta:
			return _getNewParamVBiasAdaDelta(gradient, vindex);
			break;

	case adam:
		return _getNewParamVBiasAdam(gradient, vindex);
		break;

		//case nadam:
		//	return _getNewParamVBiasNadam(gradient, vindex);
		//	break;
	}
}

inline double Optimizer<GeneralizedRBM>::getNewParamHBias(double gradient, int hindex) {
	// TODO: switch文とフラグ管理は嫌だ…
	switch (this->_OptimizeModeFlag) {
	default:
		return _getNewParamHBiasDefault(gradient, hindex);
		break;

	case momentum:
		return _getNewParamHBiasMomentum(gradient, hindex);
		break;

		case adaGrad:
			return _getNewParamHBiasAdaGrad(gradient, hindex);
			break;

		case adaDelta:
			return _getNewParamHBiasAdaDelta(gradient, hindex);
			break;

	case adam:
		return _getNewParamHBiasAdam(gradient, hindex);
		break;

		//case nadam:
		//	return _getNewParamHBiasNadam(gradient, hindex);
		//	break;
	}
}

inline double Optimizer<GeneralizedRBM>::getNewParamWeight(double gradient, int vindex, int hindex) {
	// TODO: switch文とフラグ管理は嫌だ…
	switch (this->_OptimizeModeFlag) {
	default:
		return _getNewParamWeightDefault(gradient, vindex, hindex);
		break;

	case momentum:
		return _getNewParamWeightMomentum(gradient, vindex, hindex);
		break;

		case adaGrad:
			return _getNewParamWeightAdaGrad(gradient, vindex, hindex);
			break;

		case adaDelta:
			return _getNewParamWeightAdaDelta(gradient, vindex, hindex);
			break;

	case adam:
		return _getNewParamWeightAdam(gradient, vindex, hindex);
		break;

		//case nadam:
		//	return _getNewParamWeightNadam(gradient, vindex, hindex);
		//	break;
	}
}


// default
inline void Optimizer<GeneralizedRBM>::_initDefault(GeneralizedRBM & rbm) {

}

inline double Optimizer<GeneralizedRBM>::_getNewParamVBiasDefault(double gradient, int vindex) {
	return _learningRate * gradient;
}

inline double Optimizer<GeneralizedRBM>::_getNewParamHBiasDefault(double gradient, int hindex) {
	return _learningRate * gradient;
}

inline double Optimizer<GeneralizedRBM>::_getNewParamWeightDefault(double gradient, int vindex, int hindex) {
	return _learningRate * gradient;
}

// Momentum
inline void Optimizer<GeneralizedRBM>::_initMomentum(GeneralizedRBM & rbm) {
	this->moment1st.vBias.setConstant(rbm.getVisibleSize(), 0.0);
	this->moment1st.hBias.setConstant(rbm.getHiddenSize(), 0.0);
	this->moment1st.weight.setConstant(rbm.getVisibleSize(), rbm.getHiddenSize(), 0.0);
}

inline double Optimizer<GeneralizedRBM>::_getNewParamVBiasMomentum(double gradient, int vindex) {
	auto new_gradient = this->moment1st.vBias(vindex) = this->moment1st.vBias(vindex) * _momentumRate + _learningRate * gradient;
	return new_gradient;
}

inline double Optimizer<GeneralizedRBM>::_getNewParamHBiasMomentum(double gradient, int hindex) {
	auto new_gradient = this->moment1st.hBias(hindex) = this->moment1st.hBias(hindex) * _momentumRate + _learningRate * gradient;
	return new_gradient;
}

inline double Optimizer<GeneralizedRBM>::_getNewParamWeightMomentum(double gradient, int vindex, int hindex) {
	auto new_gradient = this->moment1st.weight(vindex, hindex) = this->moment1st.weight(vindex, hindex) * _momentumRate + _learningRate * gradient;
	return new_gradient;
}

// AdaGrad
inline void Optimizer<GeneralizedRBM>::_initAdaGrad(GeneralizedRBM & rbm) {
	this->moment2nd.vBias.setConstant(rbm.getVisibleSize(), 0.0);
	this->moment2nd.hBias.setConstant(rbm.getHiddenSize(), 0.0);
	this->moment2nd.weight.setConstant(rbm.getVisibleSize(), rbm.getHiddenSize(), 0.0);
}

inline double Optimizer<GeneralizedRBM>::_getNewParamVBiasAdaGrad(double gradient, int vindex) {
	this->moment2nd.vBias(vindex) += gradient * gradient;
	auto new_gradient = _muAdaGrad / sqrt(this->moment2nd.vBias(vindex) + _epsilonAdaGrad) * gradient;

	return new_gradient;
}

inline double Optimizer<GeneralizedRBM>::_getNewParamHBiasAdaGrad(double gradient, int hindex) {
	this->moment2nd.hBias(hindex) += gradient * gradient;
	auto new_gradient = _muAdaGrad / sqrt(this->moment2nd.hBias(hindex) + _epsilonAdaGrad) * gradient;

	return new_gradient;
}

inline double Optimizer<GeneralizedRBM>::_getNewParamWeightAdaGrad(double gradient, int vindex, int hindex) {
	this->moment2nd.weight(vindex, hindex) += gradient * gradient;
	auto new_gradient = _muAdaGrad / sqrt(this->moment2nd.weight(vindex, hindex) + _epsilonAdaGrad) * gradient;

	return new_gradient;
}
// AdaDelta
inline void Optimizer<GeneralizedRBM>::_initAdaDelta(GeneralizedRBM & rbm) {
	this->moment1st.vBias.setConstant(rbm.getVisibleSize(), 0.0);
	this->moment1st.hBias.setConstant(rbm.getHiddenSize(), 0.0);
	this->moment1st.weight.setConstant(rbm.getVisibleSize(), rbm.getHiddenSize(), 0.0);

	this->moment2nd.vBias.setConstant(rbm.getVisibleSize(), 0.0);
	this->moment2nd.hBias.setConstant(rbm.getHiddenSize(), 0.0);
	this->moment2nd.weight.setConstant(rbm.getVisibleSize(), rbm.getHiddenSize(), 0.0);
}

inline double Optimizer<GeneralizedRBM>::_getNewParamVBiasAdaDelta(double gradient, int vindex) {
	auto h = this->moment1st.vBias(vindex) = _rhoAdaGrad * this->moment1st.vBias(vindex) + (1 - _rhoAdaGrad) * gradient * gradient;
	auto s = this->moment2nd.vBias(vindex);
	auto v = sqrt(s + _epsilonAdaDelta) / sqrt(h + _epsilonAdaDelta) * gradient;
	s = this->moment2nd.vBias(vindex) = _rhoAdaGrad * s + (1 - _rhoAdaGrad) * v * v;

	return v;
}

inline double Optimizer<GeneralizedRBM>::_getNewParamHBiasAdaDelta(double gradient, int hindex) {
	auto h = this->moment1st.hBias(hindex) = _rhoAdaGrad * this->moment1st.hBias(hindex) + (1 - _rhoAdaGrad) * gradient * gradient;
	auto s = this->moment2nd.hBias(hindex);
	auto v = sqrt(s + _epsilonAdaDelta) / sqrt(h + _epsilonAdaDelta) * gradient;
	s = this->moment2nd.hBias(hindex) = _rhoAdaGrad * s + (1 - _rhoAdaGrad) * v * v;

	return v;
}

inline double Optimizer<GeneralizedRBM>::_getNewParamWeightAdaDelta(double gradient, int vindex, int hindex) {
	auto h = this->moment1st.weight(vindex, hindex) = _rhoAdaGrad * this->moment1st.weight(vindex, hindex) + (1 - _rhoAdaGrad) * gradient * gradient;
	auto s = this->moment2nd.weight(vindex, hindex);
	auto v = sqrt(s + _epsilonAdaDelta) / sqrt(h + _epsilonAdaDelta) * gradient;
	s = this->moment2nd.weight(vindex, hindex) = _rhoAdaGrad * s + (1 - _rhoAdaGrad) * v * v;

	return v;
}

// Adam
inline void Optimizer<GeneralizedRBM>::_initAdam(GeneralizedRBM & rbm) {
	this->moment1st.vBias.setConstant(rbm.getVisibleSize(), 0.0);
	this->moment1st.hBias.setConstant(rbm.getHiddenSize(), 0.0);
	this->moment1st.weight.setConstant(rbm.getVisibleSize(), rbm.getHiddenSize(), 0.0);

	this->moment2nd.vBias.setConstant(rbm.getVisibleSize(), 0.0);
	this->moment2nd.hBias.setConstant(rbm.getHiddenSize(), 0.0);
	this->moment2nd.weight.setConstant(rbm.getVisibleSize(), rbm.getHiddenSize(), 0.0);
}

inline double Optimizer<GeneralizedRBM>::_getNewParamVBiasAdam(double gradient, int vindex) {
	this->moment1st.vBias(vindex) = this->_beta1 * this->moment1st.vBias(vindex) + (1 - _beta1) * gradient;  // m
	this->moment2nd.vBias(vindex) = this->_beta2 * this->moment2nd.vBias(vindex) + (1 - _beta2) * gradient * gradient;  // v

	auto m = this->moment1st.vBias(vindex) / (1 - pow(this->_beta1, this->_iteration));
	auto v = this->moment2nd.vBias(vindex) / (1 - pow(this->_beta2, this->_iteration));

	auto new_gradient = this->_alpha * m / (sqrt(v) + _epsilonAdam);
	return new_gradient;
}

inline double Optimizer<GeneralizedRBM>::_getNewParamHBiasAdam(double gradient, int hindex) {
	this->moment1st.hBias(hindex) = this->_beta1 * this->moment1st.hBias(hindex) + (1 - _beta1) * gradient;  // m
	this->moment2nd.hBias(hindex) = this->_beta2 * this->moment2nd.hBias(hindex) + (1 - _beta2) * gradient * gradient;  // v

	auto m = this->moment1st.hBias(hindex) / (1 - pow(this->_beta1, this->_iteration));
	auto v = this->moment2nd.hBias(hindex) / (1 - pow(this->_beta2, this->_iteration));

	auto new_gradient = this->_alpha * m / (sqrt(v) + _epsilonAdam);
	return new_gradient;
}

inline double Optimizer<GeneralizedRBM>::_getNewParamWeightAdam(double gradient, int vindex, int hindex) {
	this->moment1st.weight(vindex, hindex) = this->_beta1 * this->moment1st.weight(vindex, hindex) + (1 - _beta1) * gradient;  // m
	this->moment2nd.weight(vindex, hindex) = this->_beta2 * this->moment2nd.weight(vindex, hindex) + (1 - _beta2) * gradient * gradient;  // v

	auto m = this->moment1st.weight(vindex, hindex) / (1 - pow(this->_beta1, this->_iteration));
	auto v = this->moment2nd.weight(vindex, hindex) / (1 - pow(this->_beta2, this->_iteration));

	auto new_gradient = this->_alpha * m / (sqrt(v) + _epsilonAdam);
	return new_gradient;
}

// Nadam
