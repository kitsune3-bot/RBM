#pragma once
#include <fstream>
#include "../Optimizer.h"
#include "GeneralizedRBM.h"
#include <cmath>

template <>
class Optimizer<GeneralizedRBM, OptimizerType::Default> {

protected:
	int _iteration = 1;
	// momentum
	double _learningRate = 0.001;

public:

	Optimizer() = default;
	Optimizer(GeneralizedRBM & rbm);
	~Optimizer() = default;
	void init(GeneralizedRBM & rbm);
	double getNewParamVBias(double gradient, int vindex);
	double getNewParamHBias(double gradient, int hindex);
	double getNewParamWeight(double gradient, int vindex, int hindex);
	// next timestep
	void updateOptimizer();
};

inline Optimizer<GeneralizedRBM, OptimizerType::Default>::Optimizer(GeneralizedRBM & rbm) {
	this->init(rbm);
}

inline void Optimizer<GeneralizedRBM, OptimizerType::Default>::updateOptimizer() {
	this->_iteration++;
}

inline void Optimizer<GeneralizedRBM, OptimizerType::Default>::init(GeneralizedRBM & rbm) {

}

inline double Optimizer<GeneralizedRBM, OptimizerType::Default>::getNewParamVBias(double gradient, int vindex) {
	return _learningRate * gradient;
}

inline double Optimizer<GeneralizedRBM, OptimizerType::Default>::getNewParamHBias(double gradient, int hindex) {
	return _learningRate * gradient;
}

inline double Optimizer<GeneralizedRBM, OptimizerType::Default>::getNewParamWeight(double gradient, int vindex, int hindex) {
	return _learningRate * gradient;
}


template <>
class Optimizer<GeneralizedRBM, OptimizerType::Momentum> {
	struct Moment {
		Eigen::VectorXd vBias;
		Eigen::VectorXd hBias;
		Eigen::MatrixXd weight;
	};

protected:
	int _iteration = 1;
	// momentum
	double _learningRate = 0.001;
	double _momentumRate = 0.9;
	Moment moment1st;


public:

	Optimizer() = default;
	Optimizer(GeneralizedRBM & rbm);
	~Optimizer() = default;
	void init(GeneralizedRBM & rbm);
	double getNewParamVBias(double gradient, int vindex);
	double getNewParamHBias(double gradient, int hindex);
	double getNewParamWeight(double gradient, int vindex, int hindex);
	// next timestep
	void updateOptimizer();
};

inline Optimizer<GeneralizedRBM, OptimizerType::Momentum>::Optimizer(GeneralizedRBM & rbm) {
	this->init(rbm);
}

inline void Optimizer<GeneralizedRBM, OptimizerType::Momentum>::updateOptimizer() {
	this->_iteration++;
}

// Momentum
inline void Optimizer<GeneralizedRBM, OptimizerType::Momentum>::init(GeneralizedRBM & rbm) {
	this->moment1st.vBias.setConstant(rbm.getVisibleSize(), 0.0);
	this->moment1st.hBias.setConstant(rbm.getHiddenSize(), 0.0);
	this->moment1st.weight.setConstant(rbm.getVisibleSize(), rbm.getHiddenSize(), 0.0);
}

inline double Optimizer<GeneralizedRBM, OptimizerType::Momentum>::getNewParamVBias(double gradient, int vindex) {
	auto new_gradient = this->moment1st.vBias(vindex) = this->moment1st.vBias(vindex) * _momentumRate + _learningRate * gradient;
	return new_gradient;
}

inline double Optimizer<GeneralizedRBM, OptimizerType::Momentum>::getNewParamHBias(double gradient, int hindex) {
	auto new_gradient = this->moment1st.hBias(hindex) = this->moment1st.hBias(hindex) * _momentumRate + _learningRate * gradient;
	return new_gradient;
}

inline double Optimizer<GeneralizedRBM, OptimizerType::Momentum>::getNewParamWeight(double gradient, int vindex, int hindex) {
	auto new_gradient = this->moment1st.weight(vindex, hindex) = this->moment1st.weight(vindex, hindex) * _momentumRate + _learningRate * gradient;
	return new_gradient;
}

template <>
class Optimizer<GeneralizedRBM, OptimizerType::AdaGrad> {
	struct Moment {
		Eigen::VectorXd vBias;
		Eigen::VectorXd hBias;
		Eigen::MatrixXd weight;
	};

protected:
	int _iteration = 1;

	// AdaGrad
	double _muAdaGrad = 0.001;
	double _epsilonAdaGrad = 1E-08;
	Moment moment2nd;

public:
	Optimizer() = default;
	Optimizer(GeneralizedRBM & rbm);
	~Optimizer() = default;
	void init(GeneralizedRBM & rbm);
	double getNewParamVBias(double gradient, int vindex);
	double getNewParamHBias(double gradient, int hindex);
	double getNewParamWeight(double gradient, int vindex, int hindex);
	// next timestep
	void updateOptimizer();
};

inline Optimizer<GeneralizedRBM, OptimizerType::AdaGrad>::Optimizer(GeneralizedRBM & rbm) {
	this->init(rbm);
}

inline void Optimizer<GeneralizedRBM, OptimizerType::AdaGrad>::updateOptimizer() {
	this->_iteration++;
}

// AdaGrad
inline void Optimizer<GeneralizedRBM, OptimizerType::AdaGrad>::init(GeneralizedRBM & rbm) {
	this->moment2nd.vBias.setConstant(rbm.getVisibleSize(), 0.0);
	this->moment2nd.hBias.setConstant(rbm.getHiddenSize(), 0.0);
	this->moment2nd.weight.setConstant(rbm.getVisibleSize(), rbm.getHiddenSize(), 0.0);
}

inline double Optimizer<GeneralizedRBM, OptimizerType::AdaGrad>::getNewParamVBias(double gradient, int vindex) {
	this->moment2nd.vBias(vindex) += gradient * gradient;
	auto new_gradient = _muAdaGrad / sqrt(this->moment2nd.vBias(vindex) + _epsilonAdaGrad) * gradient;

	return new_gradient;
}

inline double Optimizer<GeneralizedRBM, OptimizerType::AdaGrad>::getNewParamHBias(double gradient, int hindex) {
	this->moment2nd.hBias(hindex) += gradient * gradient;
	auto new_gradient = _muAdaGrad / sqrt(this->moment2nd.hBias(hindex) + _epsilonAdaGrad) * gradient;

	return new_gradient;
}

inline double Optimizer<GeneralizedRBM, OptimizerType::AdaGrad>::getNewParamWeight(double gradient, int vindex, int hindex) {
	this->moment2nd.weight(vindex, hindex) += gradient * gradient;
	auto new_gradient = _muAdaGrad / sqrt(this->moment2nd.weight(vindex, hindex) + _epsilonAdaGrad) * gradient;

	return new_gradient;
}


template <>
class Optimizer<GeneralizedRBM, OptimizerType::AdaDelta> {
	struct Moment {
		Eigen::VectorXd vBias;
		Eigen::VectorXd hBias;
		Eigen::MatrixXd weight;
	};

protected:
	int _iteration = 1;

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

public:
	Optimizer() = default;
	Optimizer(GeneralizedRBM & rbm);
	~Optimizer() = default;
	void init(GeneralizedRBM & rbm);
	double getNewParamVBias(double gradient, int vindex);
	double getNewParamHBias(double gradient, int hindex);
	double getNewParamWeight(double gradient, int vindex, int hindex);
	// next timestep
	void updateOptimizer();
};

inline Optimizer<GeneralizedRBM, OptimizerType::AdaDelta>::Optimizer(GeneralizedRBM & rbm) {
	this->init(rbm);
}

inline void Optimizer<GeneralizedRBM, OptimizerType::AdaDelta>::updateOptimizer() {
	this->_iteration++;
}

// AdaDelta
inline void Optimizer<GeneralizedRBM, OptimizerType::AdaDelta>::init(GeneralizedRBM & rbm) {
	this->moment1st.vBias.setConstant(rbm.getVisibleSize(), 0.0);
	this->moment1st.hBias.setConstant(rbm.getHiddenSize(), 0.0);
	this->moment1st.weight.setConstant(rbm.getVisibleSize(), rbm.getHiddenSize(), 0.0);

	this->moment2nd.vBias.setConstant(rbm.getVisibleSize(), 0.0);
	this->moment2nd.hBias.setConstant(rbm.getHiddenSize(), 0.0);
	this->moment2nd.weight.setConstant(rbm.getVisibleSize(), rbm.getHiddenSize(), 0.0);
}

inline double Optimizer<GeneralizedRBM, OptimizerType::AdaDelta>::getNewParamVBias(double gradient, int vindex) {
	auto h = this->moment1st.vBias(vindex) = _rhoAdaGrad * this->moment1st.vBias(vindex) + (1 - _rhoAdaGrad) * gradient * gradient;
	auto s = this->moment2nd.vBias(vindex);
	auto v = sqrt(s + _epsilonAdaDelta) / sqrt(h + _epsilonAdaDelta) * gradient;
	s = this->moment2nd.vBias(vindex) = _rhoAdaGrad * s + (1 - _rhoAdaGrad) * v * v;

	return v;
}

inline double Optimizer<GeneralizedRBM, OptimizerType::AdaDelta>::getNewParamHBias(double gradient, int hindex) {
	auto h = this->moment1st.hBias(hindex) = _rhoAdaGrad * this->moment1st.hBias(hindex) + (1 - _rhoAdaGrad) * gradient * gradient;
	auto s = this->moment2nd.hBias(hindex);
	auto v = sqrt(s + _epsilonAdaDelta) / sqrt(h + _epsilonAdaDelta) * gradient;
	s = this->moment2nd.hBias(hindex) = _rhoAdaGrad * s + (1 - _rhoAdaGrad) * v * v;

	return v;
}

inline double Optimizer<GeneralizedRBM, OptimizerType::AdaDelta>::getNewParamWeight(double gradient, int vindex, int hindex) {
	auto h = this->moment1st.weight(vindex, hindex) = _rhoAdaGrad * this->moment1st.weight(vindex, hindex) + (1 - _rhoAdaGrad) * gradient * gradient;
	auto s = this->moment2nd.weight(vindex, hindex);
	auto v = sqrt(s + _epsilonAdaDelta) / sqrt(h + _epsilonAdaDelta) * gradient;
	s = this->moment2nd.weight(vindex, hindex) = _rhoAdaGrad * s + (1 - _rhoAdaGrad) * v * v;

	return v;
}

template <>
class Optimizer<GeneralizedRBM, OptimizerType::Adam> {
	struct Moment {
		Eigen::VectorXd vBias;
		Eigen::VectorXd hBias;
		Eigen::MatrixXd weight;
	};

protected:
	int _iteration = 1;

	// Adam
	double _alpha = 0.001;
	double _beta1 = 0.9;
	double _beta2 = 0.999;
	double _epsilonAdam = 1E-08;

	Moment moment1st;
	Moment moment2nd;

public:
	Optimizer() = default;
	Optimizer(GeneralizedRBM & rbm);
	~Optimizer() = default;
	void init(GeneralizedRBM & rbm);
	double getNewParamVBias(double gradient, int vindex);
	double getNewParamHBias(double gradient, int hindex);
	double getNewParamWeight(double gradient, int vindex, int hindex);
	// next timestep
	void updateOptimizer();
};

inline Optimizer<GeneralizedRBM, OptimizerType::Adam>::Optimizer(GeneralizedRBM & rbm) {
	this->init(rbm);
}

inline void Optimizer<GeneralizedRBM, OptimizerType::Adam>::updateOptimizer() {
	this->_iteration++;
}

// Adam
inline void Optimizer<GeneralizedRBM, OptimizerType::Adam>::init(GeneralizedRBM & rbm) {
	this->moment1st.vBias.setConstant(rbm.getVisibleSize(), 0.0);
	this->moment1st.hBias.setConstant(rbm.getHiddenSize(), 0.0);
	this->moment1st.weight.setConstant(rbm.getVisibleSize(), rbm.getHiddenSize(), 0.0);

	this->moment2nd.vBias.setConstant(rbm.getVisibleSize(), 0.0);
	this->moment2nd.hBias.setConstant(rbm.getHiddenSize(), 0.0);
	this->moment2nd.weight.setConstant(rbm.getVisibleSize(), rbm.getHiddenSize(), 0.0);
}

inline double Optimizer<GeneralizedRBM, OptimizerType::Adam>::getNewParamVBias(double gradient, int vindex) {
	this->moment1st.vBias(vindex) = this->_beta1 * this->moment1st.vBias(vindex) + (1 - _beta1) * gradient;  // m
	this->moment2nd.vBias(vindex) = this->_beta2 * this->moment2nd.vBias(vindex) + (1 - _beta2) * gradient * gradient;  // v

	auto m = this->moment1st.vBias(vindex) / (1 - pow(this->_beta1, this->_iteration));
	auto v = this->moment2nd.vBias(vindex) / (1 - pow(this->_beta2, this->_iteration));

	auto new_gradient = this->_alpha * m / (sqrt(v) + _epsilonAdam);
	return new_gradient;
}

inline double Optimizer<GeneralizedRBM, OptimizerType::Adam>::getNewParamHBias(double gradient, int hindex) {
	this->moment1st.hBias(hindex) = this->_beta1 * this->moment1st.hBias(hindex) + (1 - _beta1) * gradient;  // m
	this->moment2nd.hBias(hindex) = this->_beta2 * this->moment2nd.hBias(hindex) + (1 - _beta2) * gradient * gradient;  // v

	auto m = this->moment1st.hBias(hindex) / (1 - pow(this->_beta1, this->_iteration));
	auto v = this->moment2nd.hBias(hindex) / (1 - pow(this->_beta2, this->_iteration));

	auto new_gradient = this->_alpha * m / (sqrt(v) + _epsilonAdam);
	return new_gradient;
}

inline double Optimizer<GeneralizedRBM, OptimizerType::Adam>::getNewParamWeight(double gradient, int vindex, int hindex) {
	this->moment1st.weight(vindex, hindex) = this->_beta1 * this->moment1st.weight(vindex, hindex) + (1 - _beta1) * gradient;  // m
	this->moment2nd.weight(vindex, hindex) = this->_beta2 * this->moment2nd.weight(vindex, hindex) + (1 - _beta2) * gradient * gradient;  // v

	auto m = this->moment1st.weight(vindex, hindex) / (1 - pow(this->_beta1, this->_iteration));
	auto v = this->moment2nd.weight(vindex, hindex) / (1 - pow(this->_beta2, this->_iteration));

	auto new_gradient = this->_alpha * m / (sqrt(v) + _epsilonAdam);
	return new_gradient;
}


template <>
class Optimizer<GeneralizedRBM, OptimizerType::AdaMax> {
	struct Moment {
		Eigen::VectorXd vBias;
		Eigen::VectorXd hBias;
		Eigen::MatrixXd weight;
	};

protected:
	int _iteration = 1;

	// AdaMax
	double _alpha = 0.001;
	double _beta1 = 0.9;
	double _beta2 = 0.999;
	double _epsilonAdaMax = 1E-08;

	Moment moment1st;
	Moment moment2nd;

public:
	Optimizer() = default;
	Optimizer(GeneralizedRBM & rbm);
	~Optimizer() = default;
	void init(GeneralizedRBM & rbm);
	double getNewParamVBias(double gradient, int vindex);
	double getNewParamHBias(double gradient, int hindex);
	double getNewParamWeight(double gradient, int vindex, int hindex);
	// next timestep
	void updateOptimizer();
};

inline Optimizer<GeneralizedRBM, OptimizerType::AdaMax>::Optimizer(GeneralizedRBM & rbm) {
	this->init(rbm);
}

inline void Optimizer<GeneralizedRBM, OptimizerType::AdaMax>::updateOptimizer() {
	this->_iteration++;
}

// AdaMax
inline void Optimizer<GeneralizedRBM, OptimizerType::AdaMax>::init(GeneralizedRBM & rbm) {
	this->moment1st.vBias.setConstant(rbm.getVisibleSize(), 0.0);
	this->moment1st.hBias.setConstant(rbm.getHiddenSize(), 0.0);
	this->moment1st.weight.setConstant(rbm.getVisibleSize(), rbm.getHiddenSize(), 0.0);

	this->moment2nd.vBias.setConstant(rbm.getVisibleSize(), 0.0);
	this->moment2nd.hBias.setConstant(rbm.getHiddenSize(), 0.0);
	this->moment2nd.weight.setConstant(rbm.getVisibleSize(), rbm.getHiddenSize(), 0.0);
}

inline double Optimizer<GeneralizedRBM, OptimizerType::AdaMax>::getNewParamVBias(double gradient, int vindex) {
	auto m = this->moment1st.vBias(vindex) = this->_beta1 * this->moment1st.vBias(vindex) + (1 - _beta1) * gradient;  // m
	auto v = this->moment2nd.vBias(vindex) = std::max(this->_beta2 * this->moment2nd.vBias(vindex), abs(gradient));  // v

	auto new_gradient = this->_alpha  / (1 - pow(this->_beta1, _iteration)) * m / v;
	return new_gradient;
}

inline double Optimizer<GeneralizedRBM, OptimizerType::AdaMax>::getNewParamHBias(double gradient, int hindex) {
	auto m = this->moment1st.hBias(hindex) = this->_beta1 * this->moment1st.hBias(hindex) + (1 - _beta1) * gradient;  // m
	auto v = this->moment2nd.hBias(hindex) = std::max(this->_beta2 * this->moment2nd.hBias(hindex), abs(gradient));  // v

	auto new_gradient = this->_alpha / (1 - pow(this->_beta1, _iteration)) * m / v;
	return new_gradient;
}

inline double Optimizer<GeneralizedRBM, OptimizerType::AdaMax>::getNewParamWeight(double gradient, int vindex, int hindex) {
	auto m = this->moment1st.weight(vindex, hindex) = this->_beta1 * this->moment1st.weight(vindex, hindex) + (1 - _beta1) * gradient;  // m
	auto v = this->moment2nd.weight(vindex, hindex) = std::max(this->_beta2 * this->moment2nd.weight(vindex, hindex), abs(gradient));  // v

	auto new_gradient = this->_alpha / (1 - pow(this->_beta1, _iteration)) * m / v;
	return new_gradient;
}

