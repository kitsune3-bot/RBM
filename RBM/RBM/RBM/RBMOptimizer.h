#pragma once
#include "../Optimizer.h"
#include "RBM.h"
#include "RBMTrainer.h"

template<,>
class Optimizer<RBM, Trainer<RBM>> {
	int iteration;
};