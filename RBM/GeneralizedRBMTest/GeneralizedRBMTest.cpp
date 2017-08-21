// GeneralizedRBMTest.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "stdafx.h"


int main()
{
	GeneralizedRBM general_rbm(10, 10);
	general_rbm.setHiddenMin(-1.0);
	general_rbm.setHiddenMax(1.0);
	std::vector<double> dat = { 1,1,1,1,1,1,1,1,1,1 };

	std::cout << general_rbm.probVis(dat) << std::endl;
	return 0;
}

