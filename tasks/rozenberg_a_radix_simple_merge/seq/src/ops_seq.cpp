#include "rozenberg_a_radix_simple_merge/seq/include/ops_seq.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "rozenberg_a_radix_simple_merge/common/include/common.hpp"

namespace rozenberg_a_radix_simple_merge {

RozenbergARadixSimpleMergeSEQ::RozenbergARadixSimpleMergeSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  InType empty;
  GetInput().swap(empty);

  for (const auto &elem : in) {
    GetInput().push_back(elem);
  }

  GetOutput().clear();
}

bool RozenbergARadixSimpleMergeSEQ::ValidationImpl() {
  return (!(GetInput().empty())) && (GetOutput().empty());
}

bool RozenbergARadixSimpleMergeSEQ::PreProcessingImpl() {
  GetOutput().resize(GetInput().size());
  return GetOutput().size() == GetInput().size();
}

bool RozenbergARadixSimpleMergeSEQ::RunImpl() {
  InType data = GetInput();
  size_t n = data.size();
  InType buffer(n);

  for (int shift = 0; shift < 64; shift += 8) {
    std::array<size_t, 256> count = {0};

    for (double val : data) {
      uint64_t u = 0;
      std::memcpy(&u, &val, sizeof(double));

      u = (u >> 63 != 0U) ? ~u : (u ^ 0x8000000000000000);

      auto byte = static_cast<uint8_t>((u >> shift) & 0xFF);
      count.at(byte)++;
    }

    for (int i = 1; i < 256; ++i) {
      count.at(i) += count.at(i - 1);
    }

    for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
      uint64_t u = 0;
      std::memcpy(&u, &data[i], 8);

      u = (u >> 63 != 0U) ? ~u : (u ^ 0x8000000000000000);

      auto byte = static_cast<uint8_t>((u >> shift) & 0xFF);
      buffer[--count.at(byte)] = data[i];
    }
    data = buffer;
  }

  GetOutput() = data;

  return true;
}

bool RozenbergARadixSimpleMergeSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace rozenberg_a_radix_simple_merge
