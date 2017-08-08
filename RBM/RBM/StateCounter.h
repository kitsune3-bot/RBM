#pragma once
#include <vector>
#include <iostream>
#include <numeric>

template <typename T>
class StateCounter
{
protected:
	size_t _innerCounter = 0;  // 内部状態数カウンター
	size_t _maxCount = 0;  // 状態数カウンターの最大値
	T _steteSet;  // 各状態数
	T _stateCounterSet;

	//
	// 状態計算
	//
	void _calcState() {
		for (int n = 0; n < _stateSet.size(); n++) {
			auto & s = _stateCounterSet;
			s[n] = _innerCounter % std::accumulate(s.begin(), s.begin() + 1 + n, 1, [](int init, int i) {
				return init * i; 
			}) / std::accumulate(s.begin(), s.begin() + n, 1, [](int init, int i) {
				return init * i; 
			});
		}
	}

public: 
	StateCounter(T state_set = {}) {
		_stateSet = stateSet;
		_stateCounterSet(stateSet);
		std::fill(_stateCounterSet.begin(), _stateCounterSet.end(), 0);
		_maxCount = std::accumulate(stateSet.begin(), stateSet.end(), 1, [](int init, int i) { return init * i; });
	}

	~StateCounter() = default;

	void operator++(int value) {
		_innerCounter++;
	}

	T operator()(int value) {
		return getState();
	}

	T getState() {
		return _stateCounterSet;
	}

};
