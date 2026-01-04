#include "kondakov_v_reduce/seq/include/ops_seq.hpp"

#include <cstddef>

#include "kondakov_v_reduce/common/include/common.hpp"

namespace kondakov_v_reduce {

KondakovVReduceTaskSEQ::KondakovVReduceTaskSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool KondakovVReduceTaskSEQ::ValidationImpl() {
  return !GetInput().values.empty();
}

bool KondakovVReduceTaskSEQ::PreProcessingImpl() {
  GetOutput() = GetInput().values[0];
  return true;
}

bool KondakovVReduceTaskSEQ::RunImpl() {
  const auto &values = GetInput().values;
  const ReduceOp op = GetInput().op;

  OutType result = values[0];
  for (std::size_t i = 1; i < values.size(); ++i) {
    result = ApplyReduceOp(result, values[i], op);
  }

  GetOutput() = result;
  return true;
}

bool KondakovVReduceTaskSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kondakov_v_reduce
