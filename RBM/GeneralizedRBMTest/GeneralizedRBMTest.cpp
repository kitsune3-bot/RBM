// GeneralizedRBMTest.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "stdafx.h"

TEST(RBMTest, NormalConstantTestBeforeTrain) {
	GeneralizedRBM general_rbm(10, 10);
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

TEST(RBMTest, ParamsTest) {
	GeneralizedRBM general_rbm(1, 1);
	auto reset_params = [&] {
		general_rbm.params.b.setConstant(0);
		general_rbm.params.c.setConstant(0);
		general_rbm.params.w.setConstant(0);
		general_rbm.nodes.v.setConstant(0.0);
		general_rbm.nodes.h.setConstant(0.0);
	};

	// b << 0: 0.5 < p(v = 0)
	reset_params();
	general_rbm.params.b.setConstant(-5);
	auto p = general_rbm.condProbVis(0, 0.0);
	ASSERT_LT(0.5, p);

	// 0 << b: p(v = 0) < 0.5
	reset_params();
	general_rbm.params.b.setConstant(5);
	p = general_rbm.condProbVis(0, 0.0);
	ASSERT_LT(p, 0.5);

	// c << 0: 0.5 < p(h = 0)
	reset_params();
	general_rbm.params.c.setConstant(-5);
	p = general_rbm.condProbHid(0, 0.0);
	ASSERT_LT(0.5, p);

	// 0 << c: p(h = 0) < 0.5
	reset_params();
	general_rbm.params.c.setConstant(5);
	p = general_rbm.condProbHid(0, 0.0);
	ASSERT_LT(p, 0.5);

	//　TODO: w: 同符号, 位符号のとりやすさをチェック
	reset_params();
	general_rbm.params.w.setConstant(10);
	general_rbm.nodes.v.setConstant(0.0);
	general_rbm.nodes.h.setConstant(1.0);
	volatile auto p_v = general_rbm.condProbVis(0, 0.0);
	volatile auto p_h = general_rbm.condProbHid(0, 0.0);
	p = general_rbm.condProbHid(0, 0.0);

}
