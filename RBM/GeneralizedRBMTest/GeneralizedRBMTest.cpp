// GeneralizedRBMTest.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "stdafx.h"

TEST(RBMTest, NormalConstantTestBeforeTrain) {
	GeneralizedRBM general_rbm(10, 100);
	general_rbm.setHiddenMin(-1.0);
	general_rbm.setHiddenMax(1.0);

	// 0 < Z
	auto z = general_rbm.getNormalConstant();
	ASSERT_LT(0, z);

	// chk NaN
	ASSERT_FALSE(isnan(z));

	// chk inf, -inf
	ASSERT_FALSE(isinf(z));
}


/*
TEST(RBMTest, NormalConstantAfterTrainWithCDTest) {
	GeneralizedRBM general_rbm(10, 100);
	general_rbm.setHiddenMin(-1.0);
	general_rbm.setHiddenMax(1.0);

	// 0 < Z
	auto z = general_rbm.getNormalConstant();
	ASSERT_LT(0, z);

	// chk NaN
	ASSERT_FALSE(isnan(z));

	// chk inf, -inf
	ASSERT_FALSE(isinf(z));
}
*/

/*
TEST(RBMTest, NormalConstantAfterTrainWithExactTest) {
	throw;
	GeneralizedRBM general_rbm(10, 100);
	general_rbm.setHiddenMin(-1.0);
	general_rbm.setHiddenMax(1.0);

	// 0 < Z
	auto z = general_rbm.getNormalConstant();
	ASSERT_LT(0, z);

	// chk NaN
	ASSERT_FALSE(isnan(z));

	// chk inf, -inf
	ASSERT_FALSE(isinf(z));
}
*/

