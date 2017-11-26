#pragma once
#include <fstream>
#include "../Optimizer.h"
#include "GeneralizedFullSparseRBM.h"
#include <cmath>

template <>
class Optimizer<GeneralizedFullSparseRBM, OptimizerType::Default> {

protected:
	int _iteration = 1;
	double _learningRate = 0.001;

public:

	Optimizer() = default;
	Optimizer(GeneralizedFullSparseRBM & rbm);
	~Optimizer() = default;
	void init(GeneralizedFullSparseRBM & rbm);
	double getNewParamVBias(double gradient, int vindex);
	double getNewParamHBias(double gradient, int hindex);
	double getNewParamWeight(double gradient, int vindex, int hindex);
	double getNewParamHSparse(double gradient, int hindex);
	// next timestep
	void updateOptimizer();
};

inline Optimizer<GeneralizedFullSparseRBM, OptimizerType::Default>::Optimizer(GeneralizedFullSparseRBM & rbm) {
	this->init(rbm);
}

inline void Optimizer<GeneralizedFullSparseRBM, OptimizerType::Default>::updateOptimizer() {
	this->_iteration++;
}

inline void Optimizer<GeneralizedFullSparseRBM, OptimizerType::Default>::init(GeneralizedFullSparseRBM & rbm) {

}

inline double Optimizer<GeneralizedFullSparseRBM, OptimizerType::Default>::getNewParamVBias(double gradient, int vindex) {
	return _learningRate * gradient;
}

inline double Optimizer<GeneralizedFullSparseRBM, OptimizerType::Default>::getNewParamHBias(double gradient, int hindex) {
	return _learningRate * gradient;
}

inline double Optimizer<GeneralizedFullSparseRBM, OptimizerType::Default>::getNewParamWeight(double gradient, int vindex, int hindex) {
	return _learningRate * gradient;
}


template <>
class Optimizer<GeneralizedFullSparseRBM, OptimizerType::Momentum> {
	struct Moment {
		Eigen::VectorXd vBias;
		Eigen::VectorXd hBias;
		Eigen::VectorXd hSparse;
		Eigen::MatrixXd weight;
		Eigen::MatrixXd sparseWeight;
	};

protected:
	int _iteration = 1;
	// momentum
	double _learningRate = 0.001;
	double _momentumRate = 0.9;
	Moment moment1st;


public:

	Optimizer() = default;
	Optimizer(GeneralizedFullSparseRBM & rbm);
	~Optimizer() = default;
	void init(GeneralizedFullSparseRBM & rbm);
	double getNewParamVBias(double gradient, int vindex);
	double getNewParamHBias(double gradient, int hindex);
	double getNewParamHSparse(double gradient, int hindex);
	double getNewParamWeight(double gradient, int vindex, int hindex);
	// next timestep
	void updateOptimizer();
};

inline Optimizer<GeneralizedFullSparseRBM, OptimizerType::Momentum>::Optimizer(GeneralizedFullSparseRBM & rbm) {
	this->init(rbm);
}

inline void Optimizer<GeneralizedFullSparseRBM, OptimizerType::Momentum>::updateOptimizer() {
	this->_iteration++;
}

// Momentum
inline void Optimizer<GeneralizedFullSparseRBM, OptimizerType::Momentum>::init(GeneralizedFullSparseRBM & rbm) {
	this->moment1st.vBias.setConstant(rbm.getVisibleSize(), 0.0);
	this->moment1st.hBias.setConstant(rbm.getHiddenSize(), 0.0);
	this->moment1st.hSparse.setConstant(rbm.getHiddenSize(), 0.0);
	this->moment1st.weight.setConstant(rbm.getVisibleSize(), rbm.getHiddenSize(), 0.0);
}

inline double Optimizer<GeneralizedFullSparseRBM, OptimizerType::Momentum>::getNewParamVBias(double gradient, int vindex) {
	auto new_gradient = this->moment1st.vBias(vindex) = this->moment1st.vBias(vindex) * _momentumRate + _learningRate * gradient;
	return new_gradient;
}

inline double Optimizer<GeneralizedFullSparseRBM, OptimizerType::Momentum>::getNewParamHBias(double gradient, int hindex) {
	auto new_gradient = this->moment1st.hBias(hindex) = this->moment1st.hBias(hindex) * _momentumRate + _learningRate * gradient;
	return new_gradient;
}

inline double Optimizer<GeneralizedFullSparseRBM, OptimizerType::Momentum>::getNewParamHSparse(double gradient, int hindex) {
	auto new_gradient = this->moment1st.hSparse(hindex) = this->moment1st.hSparse(hindex) * _momentumRate + _learningRate * gradient;
	return new_gradient;
}

inline double Optimizer<GeneralizedFullSparseRBM, OptimizerType::Momentum>::getNewParamWeight(double gradient, int vindex, int hindex) {
	auto new_gradient = this->moment1st.weight(vindex, hindex) = this->moment1st.weight(vindex, hindex) * _momentumRate + _learningRate * gradient;
	return new_gradient;
}

template <>
class Optimizer<GeneralizedFullSparseRBM, OptimizerType::AdaGrad> {
	struct Moment {
		Eigen::VectorXd vBias;
		Eigen::VectorXd hBias;
		Eigen::VectorXd hSparse;
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
	Optimizer(GeneralizedFullSparseRBM & rbm);
	~Optimizer() = default;
	void init(GeneralizedFullSparseRBM & rbm);
	double getNewParamVBias(double gradient, int vindex);
	double getNewParamHBias(double gradient, int hindex);
	double getNewParamHSparse(double gradient, int hindex);
	double getNewParamWeight(double gradient, int vindex, int hindex);
	// next timestep
	void updateOptimizer();
};

inline Optimizer<GeneralizedFullSparseRBM, OptimizerType::AdaGrad>::Optimizer(GeneralizedFullSparseRBM & rbm) {
	this->init(rbm);
}

inline void Optimizer<GeneralizedFullSparseRBM, OptimizerType::AdaGrad>::updateOptimizer() {
	this->_iteration++;
}

// AdaGrad
inline void Optimizer<GeneralizedFullSparseRBM, OptimizerType::AdaGrad>::init(GeneralizedFullSparseRBM & rbm) {
	this->moment2nd.vBias.setConstant(rbm.getVisibleSize(), 0.0);
	this->moment2nd.hBias.setConstant(rbm.getHiddenSize(), 0.0);
	this->moment2nd.hSparse.setConstant(rbm.getHiddenSize(), 0.0);
	this->moment2nd.weight.setConstant(rbm.getVisibleSize(), rbm.getHiddenSize(), 0.0);
}

inline double Optimizer<GeneralizedFullSparseRBM, OptimizerType::AdaGrad>::getNewParamVBias(double gradient, int vindex) {
	this->moment2nd.vBias(vindex) += gradient * gradient;
	auto new_gradient = _muAdaGrad / sqrt(this->moment2nd.vBias(vindex) + _epsilonAdaGrad) * gradient;

	return new_gradient;
}

inline double Optimizer<GeneralizedFullSparseRBM, OptimizerType::AdaGrad>::getNewParamHBias(double gradient, int hindex) {
	this->moment2nd.hBias(hindex) += gradient * gradient;
	auto new_gradient = _muAdaGrad / sqrt(this->moment2nd.hBias(hindex) + _epsilonAdaGrad) * gradient;

	return new_gradient;
}

inline double Optimizer<GeneralizedFullSparseRBM, OptimizerType::AdaGrad>::getNewParamHSparse(double gradient, int hindex) {
	this->moment2nd.hBias(hindex) += gradient * gradient;
	auto new_gradient = _muAdaGrad / sqrt(this->moment2nd.hBias(hindex) + _epsilonAdaGrad) * gradient;

	return new_gradient;
}

inline double Optimizer<GeneralizedFullSparseRBM, OptimizerType::AdaGrad>::getNewParamWeight(double gradient, int vindex, int hindex) {
	this->moment2nd.weight(vindex, hindex) += gradient * gradient;
	auto new_gradient = _muAdaGrad / sqrt(this->moment2nd.weight(vindex, hindex) + _epsilonAdaGrad) * gradient;

	return new_gradient;
}


template <>
class Optimizer<GeneralizedFullSparseRBM, OptimizerType::AdaDelta> {
	struct Moment {
		Eigen::VectorXd vBias;
		Eigen::VectorXd hBias;
		Eigen::VectorXd hSparse;
		Eigen::MatrixXd weight;
	};

protected:
	int _iteration = 1;

	// AdaDelta
	double _rhoAdaGrad = 0.95;
	double _epsilonAdaDelta = 1E-06;


	Moment moment1st;
	Moment moment2nd;

public:
	Optimizer() = default;
	Optimizer(GeneralizedFullSparseRBM & rbm);
	~Optimizer() = default;
	void init(GeneralizedFullSparseRBM & rbm);
	double getNewParamVBias(double gradient, int vindex);
	double getNewParamHBias(double gradient, int hindex);
	double getNewParamHSparse(double gradient, int hindex);
	double getNewParamWeight(double gradient, int vindex, int hindex);
	// next timestep
	void updateOptimizer();
};

inline Optimizer<GeneralizedFullSparseRBM, OptimizerType::AdaDelta>::Optimizer(GeneralizedFullSparseRBM & rbm) {
	this->init(rbm);
}

inline void Optimizer<GeneralizedFullSparseRBM, OptimizerType::AdaDelta>::updateOptimizer() {
	this->_iteration++;
}

// AdaDelta
inline void Optimizer<GeneralizedFullSparseRBM, OptimizerType::AdaDelta>::init(GeneralizedFullSparseRBM & rbm) {
	this->moment1st.vBias.setConstant(rbm.getVisibleSize(), 0.0);
	this->moment1st.hBias.setConstant(rbm.getHiddenSize(), 0.0);
	this->moment1st.hSparse.setConstant(rbm.getHiddenSize(), 0.0);
	this->moment1st.weight.setConstant(rbm.getVisibleSize(), rbm.getHiddenSize(), 0.0);

	this->moment2nd.vBias.setConstant(rbm.getVisibleSize(), 0.0);
	this->moment2nd.hBias.setConstant(rbm.getHiddenSize(), 0.0);
	this->moment2nd.hSparse.setConstant(rbm.getHiddenSize(), 0.0);
	this->moment2nd.weight.setConstant(rbm.getVisibleSize(), rbm.getHiddenSize(), 0.0);
}

inline double Optimizer<GeneralizedFullSparseRBM, OptimizerType::AdaDelta>::getNewParamVBias(double gradient, int vindex) {
	auto h = this->moment1st.vBias(vindex) = _rhoAdaGrad * this->moment1st.vBias(vindex) + (1 - _rhoAdaGrad) * gradient * gradient;
	auto s = this->moment2nd.vBias(vindex);
	auto v = sqrt(s + _epsilonAdaDelta) / sqrt(h + _epsilonAdaDelta) * gradient;
	s = this->moment2nd.vBias(vindex) = _rhoAdaGrad * s + (1 - _rhoAdaGrad) * v * v;

	return v;
}

inline double Optimizer<GeneralizedFullSparseRBM, OptimizerType::AdaDelta>::getNewParamHBias(double gradient, int hindex) {
	auto h = this->moment1st.hBias(hindex) = _rhoAdaGrad * this->moment1st.hBias(hindex) + (1 - _rhoAdaGrad) * gradient * gradient;
	auto s = this->moment2nd.hBias(hindex);
	auto v = sqrt(s + _epsilonAdaDelta) / sqrt(h + _epsilonAdaDelta) * gradient;
	s = this->moment2nd.hBias(hindex) = _rhoAdaGrad * s + (1 - _rhoAdaGrad) * v * v;

	return v;
}

inline double Optimizer<GeneralizedFullSparseRBM, OptimizerType::AdaDelta>::getNewParamHSparse(double gradient, int hindex) {
	auto h = this->moment1st.hSparse(hindex) = _rhoAdaGrad * this->moment1st.hSparse(hindex) + (1 - _rhoAdaGrad) * gradient * gradient;
	auto s = this->moment2nd.hSparse(hindex);
	auto v = sqrt(s + _epsilonAdaDelta) / sqrt(h + _epsilonAdaDelta) * gradient;
	s = this->moment2nd.hSparse(hindex) = _rhoAdaGrad * s + (1 - _rhoAdaGrad) * v * v;

	return v;
}

inline double Optimizer<GeneralizedFullSparseRBM, OptimizerType::AdaDelta>::getNewParamWeight(double gradient, int vindex, int hindex) {
	auto h = this->moment1st.weight(vindex, hindex) = _rhoAdaGrad * this->moment1st.weight(vindex, hindex) + (1 - _rhoAdaGrad) * gradient * gradient;
	auto s = this->moment2nd.weight(vindex, hindex);
	auto v = sqrt(s + _epsilonAdaDelta) / sqrt(h + _epsilonAdaDelta) * gradient;
	s = this->moment2nd.weight(vindex, hindex) = _rhoAdaGrad * s + (1 - _rhoAdaGrad) * v * v;

	return v;
}

template <>
class Optimizer<GeneralizedFullSparseRBM, OptimizerType::Adam> {
	struct Moment {
		Eigen::VectorXd vBias;
		Eigen::VectorXd hBias;
		Eigen::VectorXd hSparse;
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
	Optimizer(GeneralizedFullSparseRBM & rbm);
	~Optimizer() = default;
	void init(GeneralizedFullSparseRBM & rbm);
	double getNewParamVBias(double gradient, int vindex);
	double getNewParamHBias(double gradient, int hindex);
	double getNewParamHSparse(double gradient, int hindex);
	double getNewParamWeight(double gradient, int vindex, int hindex);
	// next timestep
	void updateOptimizer();
};

inline Optimizer<GeneralizedFullSparseRBM, OptimizerType::Adam>::Optimizer(GeneralizedFullSparseRBM & rbm) {
	this->init(rbm);
}

inline void Optimizer<GeneralizedFullSparseRBM, OptimizerType::Adam>::updateOptimizer() {
	this->_iteration++;
}

// Adam
inline void Optimizer<GeneralizedFullSparseRBM, OptimizerType::Adam>::init(GeneralizedFullSparseRBM & rbm) {
	this->moment1st.vBias.setConstant(rbm.getVisibleSize(), 0.0);
	this->moment1st.hBias.setConstant(rbm.getHiddenSize(), 0.0);
	this->moment1st.hSparse.setConstant(rbm.getHiddenSize(), 0.0);
	this->moment1st.weight.setConstant(rbm.getVisibleSize(), rbm.getHiddenSize(), 0.0);

	this->moment2nd.vBias.setConstant(rbm.getVisibleSize(), 0.0);
	this->moment2nd.hBias.setConstant(rbm.getHiddenSize(), 0.0);
	this->moment2nd.hSparse.setConstant(rbm.getHiddenSize(), 0.0);
	this->moment2nd.weight.setConstant(rbm.getVisibleSize(), rbm.getHiddenSize(), 0.0);
}

inline double Optimizer<GeneralizedFullSparseRBM, OptimizerType::Adam>::getNewParamVBias(double gradient, int vindex) {
	this->moment1st.vBias(vindex) = this->_beta1 * this->moment1st.vBias(vindex) + (1 - _beta1) * gradient;  // m
	this->moment2nd.vBias(vindex) = this->_beta2 * this->moment2nd.vBias(vindex) + (1 - _beta2) * gradient * gradient;  // v

	auto m = this->moment1st.vBias(vindex) / (1 - pow(this->_beta1, this->_iteration));
	auto v = this->moment2nd.vBias(vindex) / (1 - pow(this->_beta2, this->_iteration));

	auto new_gradient = this->_alpha * m / (sqrt(v) + _epsilonAdam);
	return new_gradient;
}

inline double Optimizer<GeneralizedFullSparseRBM, OptimizerType::Adam>::getNewParamHBias(double gradient, int hindex) {
	this->moment1st.hBias(hindex) = this->_beta1 * this->moment1st.hBias(hindex) + (1 - _beta1) * gradient;  // m
	this->moment2nd.hBias(hindex) = this->_beta2 * this->moment2nd.hBias(hindex) + (1 - _beta2) * gradient * gradient;  // v

	auto m = this->moment1st.hBias(hindex) / (1 - pow(this->_beta1, this->_iteration));
	auto v = this->moment2nd.hBias(hindex) / (1 - pow(this->_beta2, this->_iteration));

	auto new_gradient = this->_alpha * m / (sqrt(v) + _epsilonAdam);
	return new_gradient;
}

inline double Optimizer<GeneralizedFullSparseRBM, OptimizerType::Adam>::getNewParamHSparse(double gradient, int hindex) {
	this->moment1st.hSparse(hindex) = this->_beta1 * this->moment1st.hSparse(hindex) + (1 - _beta1) * gradient;  // m
	this->moment2nd.hSparse(hindex) = this->_beta2 * this->moment2nd.hSparse(hindex) + (1 - _beta2) * gradient * gradient;  // v

	auto m = this->moment1st.hSparse(hindex) / (1 - pow(this->_beta1, this->_iteration));
	auto v = this->moment2nd.hSparse(hindex) / (1 - pow(this->_beta2, this->_iteration));

	auto new_gradient = this->_alpha * m / (sqrt(v) + _epsilonAdam);
	return new_gradient;
}

inline double Optimizer<GeneralizedFullSparseRBM, OptimizerType::Adam>::getNewParamWeight(double gradient, int vindex, int hindex) {
	this->moment1st.weight(vindex, hindex) = this->_beta1 * this->moment1st.weight(vindex, hindex) + (1 - _beta1) * gradient;  // m
	this->moment2nd.weight(vindex, hindex) = this->_beta2 * this->moment2nd.weight(vindex, hindex) + (1 - _beta2) * gradient * gradient;  // v

	auto m = this->moment1st.weight(vindex, hindex) / (1 - pow(this->_beta1, this->_iteration));
	auto v = this->moment2nd.weight(vindex, hindex) / (1 - pow(this->_beta2, this->_iteration));

	auto new_gradient = this->_alpha * m / (sqrt(v) + _epsilonAdam);
	return new_gradient;
}


template <>
class Optimizer<GeneralizedFullSparseRBM, OptimizerType::AdaMax> {
	struct Moment {
		Eigen::VectorXd vBias;
		Eigen::VectorXd hBias;
		Eigen::VectorXd hSparse;
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
	Optimizer(GeneralizedFullSparseRBM & rbm);
	~Optimizer() = default;
	void init(GeneralizedFullSparseRBM & rbm);
	double getNewParamVBias(double gradient, int vindex);
	double getNewParamHBias(double gradient, int hindex);
	double getNewParamHSparse(double gradient, int hindex);
	double getNewParamWeight(double gradient, int vindex, int hindex);
	// next timestep
	void updateOptimizer();
};

inline Optimizer<GeneralizedFullSparseRBM, OptimizerType::AdaMax>::Optimizer(GeneralizedFullSparseRBM & rbm) {
	this->init(rbm);
}

inline void Optimizer<GeneralizedFullSparseRBM, OptimizerType::AdaMax>::updateOptimizer() {
	this->_iteration++;
}

// AdaMax
inline void Optimizer<GeneralizedFullSparseRBM, OptimizerType::AdaMax>::init(GeneralizedFullSparseRBM & rbm) {
	this->moment1st.vBias.setConstant(rbm.getVisibleSize(), 0.0);
	this->moment1st.hBias.setConstant(rbm.getHiddenSize(), 0.0);
	this->moment1st.hSparse.setConstant(rbm.getHiddenSize(), 0.0);
	this->moment1st.weight.setConstant(rbm.getVisibleSize(), rbm.getHiddenSize(), 0.0);

	this->moment2nd.vBias.setConstant(rbm.getVisibleSize(), 0.0);
	this->moment2nd.hBias.setConstant(rbm.getHiddenSize(), 0.0);
	this->moment2nd.hSparse.setConstant(rbm.getHiddenSize(), 0.0);
	this->moment2nd.weight.setConstant(rbm.getVisibleSize(), rbm.getHiddenSize(), 0.0);
}

inline double Optimizer<GeneralizedFullSparseRBM, OptimizerType::AdaMax>::getNewParamVBias(double gradient, int vindex) {
	auto m = this->moment1st.vBias(vindex) = this->_beta1 * this->moment1st.vBias(vindex) + (1 - _beta1) * gradient;  // m
	auto v = this->moment2nd.vBias(vindex) = std::max(this->_beta2 * this->moment2nd.vBias(vindex), abs(gradient));  // v

	auto new_gradient = this->_alpha / (1 - pow(this->_beta1, _iteration)) * m / v;
	return new_gradient;
}

inline double Optimizer<GeneralizedFullSparseRBM, OptimizerType::AdaMax>::getNewParamHBias(double gradient, int hindex) {
	auto m = this->moment1st.hBias(hindex) = this->_beta1 * this->moment1st.hBias(hindex) + (1 - _beta1) * gradient;  // m
	auto v = this->moment2nd.hBias(hindex) = std::max(this->_beta2 * this->moment2nd.hBias(hindex), abs(gradient));  // v

	auto new_gradient = this->_alpha / (1 - pow(this->_beta1, _iteration)) * m / v;
	return new_gradient;
}

inline double Optimizer<GeneralizedFullSparseRBM, OptimizerType::AdaMax>::getNewParamHSparse(double gradient, int hindex) {
	auto m = this->moment1st.hSparse(hindex) = this->_beta1 * this->moment1st.hSparse(hindex) + (1 - _beta1) * gradient;  // m
	auto v = this->moment2nd.hSparse(hindex) = std::max(this->_beta2 * this->moment2nd.hSparse(hindex), abs(gradient));  // v

	auto new_gradient = this->_alpha / (1 - pow(this->_beta1, _iteration)) * m / v;
	return new_gradient;
}


inline double Optimizer<GeneralizedFullSparseRBM, OptimizerType::AdaMax>::getNewParamWeight(double gradient, int vindex, int hindex) {
	auto m = this->moment1st.weight(vindex, hindex) = this->_beta1 * this->moment1st.weight(vindex, hindex) + (1 - _beta1) * gradient;  // m
	auto v = this->moment2nd.weight(vindex, hindex) = std::max(this->_beta2 * this->moment2nd.weight(vindex, hindex), abs(gradient));  // v

	auto new_gradient = this->_alpha / (1 - pow(this->_beta1, _iteration)) * m / v;
	return new_gradient;
}

