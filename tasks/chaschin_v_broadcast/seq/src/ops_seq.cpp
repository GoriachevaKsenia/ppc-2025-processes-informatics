#include "chaschin_v_broadcast/seq/include/ops_seq.hpp"

namespace chaschin_v_broadcast {

template <typename T>
ChaschinVBroadcastSEQ<T>::ChaschinVBroadcastSEQ(const InType &in) {
  this->SetTypeOfTask(GetStaticTypeOfTask());
  this->GetInput() = in;
  this->GetOutput().resize(0);
}

template <typename T>
bool ChaschinVBroadcastSEQ<T>::ValidationImpl() {
  return (!this->GetInput().empty()) && (this->GetOutput().empty());
}

template <typename T>
bool ChaschinVBroadcastSEQ<T>::PreProcessingImpl() {
  return true;
}

template <typename T>
bool ChaschinVBroadcastSEQ<T>::RunImpl() {
  this->GetOutput() = this->GetInput();
  return true;
}

template <typename T>
bool ChaschinVBroadcastSEQ<T>::PostProcessingImpl() {
  return true;
}

template class ChaschinVBroadcastSEQ<int>;
template class ChaschinVBroadcastSEQ<float>;
template class ChaschinVBroadcastSEQ<double>;

}  // namespace chaschin_v_broadcast
