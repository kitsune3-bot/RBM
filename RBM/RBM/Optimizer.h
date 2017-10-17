#pragma once

namespace OptimizerType{
	struct Default {};
	struct Momentum {};
	struct AdaGrad {};
	struct AdaDelta {};
	struct Adam {};
	struct Adamax {};
	struct Nadam {};
}

template<class RBMBase, class OPTIMIZERTYPE>
class Optimizer {

};

