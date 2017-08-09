#pragma once
#include <Eigen/Core>
#include <iostream>
#include <numeric>

template <typename T>
class StateCounter
{
protected:
	size_t _innerCounter = 0;  // 内部状態数カウンター
	size_t _maxCount = 0;  // 状態数カウンターの最大値
	size_t _elemNum = 0;  // 桁数
	T _stateSet;  // 各状態数
	T _stateCounterSet;

	//
	// 状態計算
	//
	void _calcState() {
		for (int n = 0; n < _stateSet.size(); n++) {
			auto & s = _stateSet;
			auto & sc = _stateCounterSet;

			sc[n] = _innerCounter % std::accumulate(s.begin(), s.begin() + 1 + n, 1, [](int init, int i) {
				return init * i;
			}) / std::accumulate(s.begin(), s.begin() + n, 1, [](int init, int i) {
				return init * i;
			});
		}
	}

public:
	StateCounter(T state_set = {}) {
		_stateSet = state_set;
		_stateCounterSet = state_set;
		std::fill(_stateCounterSet.begin(), _stateCounterSet.end(), 0);
		_maxCount = std::accumulate(state_set.begin(), state_set.end(), 1, [](int init, int i) { return init * i; });
		_elemNum = state_set.size();
	}

	~StateCounter() = default;

	void operator++(int value) {
		_innerCounter++;
	}

	T operator()(int value) {
		return getState();
	}

	T getState() {
		_calcState();
		return _stateCounterSet;
	}

	size_t getMaxCount() {
		return _maxCount;
	}

};

