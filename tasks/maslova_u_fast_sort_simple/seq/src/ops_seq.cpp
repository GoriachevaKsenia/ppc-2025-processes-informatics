#include "maslova_u_fast_sort_simple/seq/include/ops_seq.hpp"

#include "maslova_u_fast_sort_simple/common/include/common.hpp"
#include "task/include/task.hpp"

namespace maslova_u_fast_sort_simple {

MaslovaUFastSortSimpleSEQ::MaslovaUFastSortSimpleSEQ(const InType &in) {
  SetTypeOfTask(ppc::task::TypeOfTask::kMPI);
  GetInput() = in;
}

bool MaslovaUFastSortSimpleSEQ::ValidationImpl() {
  return true;
}

bool MaslovaUFastSortSimpleSEQ::PreProcessingImpl() {
  return true;
}

bool MaslovaUFastSortSimpleSEQ::RunImpl() {
  GetOutput() = GetInput();

  // Сортируем
  GetOutput() = GetInput();
  if (!GetOutput().empty()) {
    QuickSort(GetOutput().data(), 0, static_cast<int>(GetOutput().size()) - 1);
  }
  return true;
}

bool MaslovaUFastSortSimpleSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace maslova_u_fast_sort_simple
