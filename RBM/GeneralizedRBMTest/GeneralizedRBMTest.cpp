// GeneralizedRBMTest.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "stdafx.h"
#include <rbmutil.h>
#include <StateCounter.h>

int main()
{
	GeneralizedRBM general_rbm(10, 10);
	GeneralizedRBM general_rbm2(10, 10);
	general_rbm.setHiddenMin(-1.0);
	general_rbm.setHiddenMax(1.0);
	std::vector<double> dat = { 1,1,1,1,1,1,1,1,1,1 };
	StateCounter<std::vector<int>> sc(std::vector<int>(general_rbm.getVisibleSize(), 2));  // 可視変数Vの状態カウンター

	std::cout << rbmutil::kld(general_rbm, general_rbm2, std::vector<int>{0, 1}) << std::endl;

	return 0;
}

