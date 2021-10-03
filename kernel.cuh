#pragma once
#include <iostream>
#include <vector>

template <typename FUNC>
std::vector<float> f(std::vector<float> const& a, FUNC func);
