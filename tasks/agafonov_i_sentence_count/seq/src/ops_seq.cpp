#include "agafonov_i_sentence_count/seq/include/ops_seq.hpp"

#include <cctype>
#include <string>

#include "agafonov_i_sentence_count/common/include/common.hpp"

namespace agafonov_i_sentence_count {

SentenceCountSEQ::SentenceCountSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool SentenceCountSEQ::ValidationImpl() {
  return true;
}

bool SentenceCountSEQ::PreProcessingImpl() {
  return true;
}

bool SentenceCountSEQ::RunImpl() {
  const std::string &text = GetInput();
  if (text.empty()) {
    GetOutput() = 0;
    return true;
  }

  int count = 0;
  bool in_word = false;
  int len = static_cast<int>(text.length());

  for (int i = 0; i < len; ++i) {
    auto c = static_cast<unsigned char>(text[i]);
    if (std::isalnum(c) != 0) {
      in_word = true;
    } else if ((c == '.' || c == '!' || c == '?') && in_word) {
      if (c == '.' && i + 1 < len && text[i + 1] == '.') {
        continue;
      }
      count++;
      in_word = false;
    }
  }

  if (in_word) {
    count++;
  }

  GetOutput() = count;
  return true;
}

bool SentenceCountSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace agafonov_i_sentence_count
