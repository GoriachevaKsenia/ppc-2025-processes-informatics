#include "gusev_d_sentence_count/seq/include/ops_seq.hpp"

#include <cstddef>
#include <string>

#include "gusev_d_sentence_count/common/include/common.hpp"

namespace gusev_d_sentence_count {

namespace {

bool IsTerminator(char c) {
  return (c == '.' || c == '!' || c == '?');
}

}  // namespace

GusevDSentenceCountSEQ::GusevDSentenceCountSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool GusevDSentenceCountSEQ::ValidationImpl() {
  return true;
}

bool GusevDSentenceCountSEQ::PreProcessingImpl() {
  return true;
}

bool GusevDSentenceCountSEQ::RunImpl() {
  const std::string &data = GetInput();

  if (data.empty()) {
    GetOutput() = 0;
    return true;
  }

  size_t sentences = 0;
  size_t len = data.length();

  for (size_t i = 0; i < len; ++i) {
    if (IsTerminator(data[i])) {
      bool is_next_not_term = (i + 1 == len) || !IsTerminator(data[i + 1]);
      if (is_next_not_term) {
        sentences++;
      }
    }
  }

  GetOutput() = sentences;
  return true;
}

bool GusevDSentenceCountSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace gusev_d_sentence_count
