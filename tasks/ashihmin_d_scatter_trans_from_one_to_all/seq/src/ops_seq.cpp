#include "ashihmin_d_scatter_trans_from_one_to_all/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace ashihmin_d_scatter_trans_from_one_to_all {

template <typename T>
AshihminDScatterTransFromOneToAllSEQ<T>::AshihminDScatterTransFromOneToAllSEQ(const InType &in) {
  this->SetTypeOfTask(GetStaticTypeOfTask());
  this->GetInput() = in;
  this->GetOutput().resize(0);
}

template <typename T>
bool AshihminDScatterTransFromOneToAllSEQ<T>::ValidationImpl() {
  const auto &params = this->GetInput();
  return (params.elements_per_process > 0) && (params.root >= 0) && (this->GetOutput().empty());
}

template <typename T>
bool AshihminDScatterTransFromOneToAllSEQ<T>::PreProcessingImpl() {
  return true;
}

template <typename T>
bool AshihminDScatterTransFromOneToAllSEQ<T>::RunImpl() {
  const auto &params = this->GetInput();

  int elements_per_proc = params.elements_per_process;
  int root = params.root;

  std::vector<T> local_data(elements_per_proc);

  if (root == 0 && params.data.size() >= static_cast<size_t>(elements_per_proc)) {
    std::copy(params.data.begin(), params.data.begin() + elements_per_proc, local_data.begin());
  } else {
    std::fill(local_data.begin(), local_data.end(), T{});
  }

  this->GetOutput() = local_data;
  return true;
}

template <typename T>
bool AshihminDScatterTransFromOneToAllSEQ<T>::PostProcessingImpl() {
  return !this->GetOutput().empty();
}

template class AshihminDScatterTransFromOneToAllSEQ<int>;
template class AshihminDScatterTransFromOneToAllSEQ<float>;
template class AshihminDScatterTransFromOneToAllSEQ<double>;

}  // namespace ashihmin_d_scatter_trans_from_one_to_all
