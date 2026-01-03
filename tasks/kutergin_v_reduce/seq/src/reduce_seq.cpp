#include "../include/reduce_seq.hpp"

#include <numeric>  // для std::accumulate

#include "../../common/include/common.hpp"

namespace kutergin_v_reduce {

ReduceSequential::ReduceSequential(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());  // установка типа задачи
  GetInput() = in;                       // сохранение входных данных
  GetOutput() = 0;                       // инициализация выходных данных
}

bool ReduceSequential::ValidationImpl() {
  return true;
}

bool ReduceSequential::PreProcessingImpl() {
  return true;
}

bool ReduceSequential::RunImpl() {
  const auto &input_vec = GetInput().data;
  if (input_vec.empty()) {
    GetOutput() = 0;
  } else {
    GetOutput() = std::accumulate(input_vec.begin(), input_vec.end(), 0);  // эмулирование операции MPI_SUM
  }
  return true;
}

bool ReduceSequential::PostProcessingImpl() {
  return true;
}

}  // namespace kutergin_v_reduce
