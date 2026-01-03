#pragma once

#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace shekhirev_v_cg_method {

struct InputData {
  std::vector<double> A;
  std::vector<double> b;
  int n;

  InputData() : n(0) {}
  InputData(const std::vector<double> &matrix, const std::vector<double> &rhs, int size) : A(matrix), b(rhs), n(size) {}
};

inline std::ostream &operator<<(std::ostream &os, const InputData &data) {
  os << "n=" << data.n;
  return os;
}

using InType = InputData;
using OutType = std::vector<double>;
using TestType = std::tuple<InputData, std::vector<double>, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace shekhirev_v_cg_method
