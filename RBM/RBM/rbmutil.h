#pragma once
#include <iostream>

namespace rbmutil{
	template <class T, class STL>
	STL data_gen(T & rbm, int update_count) {
		for (int c = 0; c < update_count; c++) {
			rbm.sampler.updateByBlockedGibbsSamplingVisible(rbm);
			rbm.sampler.updateByBlockedGibbsSamplingHidden(rbm);
		}

		std::vector<double> dat(rbm.getVisibleSize());

		for (int i = 0; i < dat.size(); i++) {
			dat[i] = rbm.nodes.v(i);
		}

		return dat;
	}

	template <class STL>
	void print_stl(STL & stl) {
		for (int i = 0; i < stl.size() - 1; i++) {
			std::cout << stl[i] << ", "
		}

		std::cout << stl[stl.size() - 1] << std::endl;
	}
}
