#pragma once
#include <iostream>
#include <cmath>
#include "StateCounter.h"

namespace rbmutil{

	// generate data from rbm
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

	// output stl value to stdout
	template <class STL>
	void print_stl(STL & stl) {
		for (int i = 0; i < stl.size() - 1; i++) {
			std::cout << stl[i] << ", ";
		}

		std::cout << stl[stl.size() - 1] << std::endl;
	}

	// Kullback–Leibler divergence
	// v_val: EX: v_i \in {0, 1}
	template <class RBM1, class RBM2, class STL>
	double kld(RBM1 & rbm1, RBM2 & rbm2, STL & v_val) {
		StateCounter<std::vector<int>> sc(std::vector<int>(rbm1.getVisibleSize(), v_val.size()));
		std::vector<double> dat(rbm1.getVisibleSize());
		auto setting_data_from_state = [&] {
			auto state = sc.getState();

			for (int i = 0; i < rbm1.getVisibleSize(); i++) {
				dat[i] = v_val[state[i]];
			}
		};



		int max_count = sc.getMaxCount();
		double value = 0.0;
		for (int c = 0; c < max_count; c++, sc++) {
			double prob[2];
			prob[0] = rbm1.probVis(dat);
			prob[1] = rbm2.probVis(dat);

			value += prob[0] * log(prob[0] / prob[1]);
		}

		return value;
	}
}
