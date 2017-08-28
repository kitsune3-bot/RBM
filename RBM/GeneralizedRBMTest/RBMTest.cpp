
#include "stdafx.h"

TEST(AddTest, Test2) {
	GeneralizedRBM general_rbm(10, 100);
	general_rbm.setHiddenMin(-1.0);
	general_rbm.setHiddenMax(1.0);

	ASSERT_LT(0, general_rbm.getNormalConstant());
}

