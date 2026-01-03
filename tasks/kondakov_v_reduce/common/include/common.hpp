#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace kondakov_v_reduce {

enum class ReduceOp : std::uint8_t { kSum, kProd, kMin, kMax };

struct ReduceInput {
  std::vector<int> values;
  ReduceOp op{ReduceOp::kSum};
};

using InType = ReduceInput;
using OutType = int64_t;
using TestType = std::tuple<std::vector<int>, std::string>;

using BaseTask = ppc::task::Task<InType, OutType>;

inline OutType ApplyReduceOp(OutType a, OutType b, ReduceOp op) {
  switch (op) {
    case ReduceOp::kSum:
      return a + b;
    case ReduceOp::kProd:
      return a * b;
    case ReduceOp::kMin:
      return std::min(a, b);
    case ReduceOp::kMax:
      return std::max(a, b);
    default:
      return a;
  }
}

inline OutType GetNeutralElement(ReduceOp op) {
  switch (op) {
    case ReduceOp::kSum:
      return 0;
    case ReduceOp::kProd:
      return 1;
    case ReduceOp::kMin:
      return std::numeric_limits<int>::max();
    case ReduceOp::kMax:
      return std::numeric_limits<int>::min();
    default:
      return 0;
  }
}

}  // namespace kondakov_v_reduce
