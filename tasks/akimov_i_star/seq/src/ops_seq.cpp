#include "akimov_i_star/seq/include/ops_seq.hpp"

#include <cctype>
#include <cstddef>
#include <string_view>
#include <vector>

#include "akimov_i_star/common/include/common.hpp"

namespace akimov_i_star {

namespace {

constexpr std::size_t kPrefixLen = 5;
constexpr std::string_view kPrefix = "send:";

bool CheckPrefix(const char *data, std::size_t n, std::size_t i) {
  if (i + kPrefixLen > n) {
    return false;
  }
  return std::string_view(data + i, kPrefixLen) == kPrefix;
}

std::size_t SkipWhitespace(const char *data, std::size_t n, std::size_t pos) {
  while (pos < n && (data[pos] == ' ' || data[pos] == '\t' || data[pos] == '\r')) {
    ++pos;
  }
  return pos;
}

bool ProcessLineForZeroDst(const char *data, std::size_t n, std::size_t &i) {
  if (!CheckPrefix(data, n, i)) {
    return false;
  }

  std::size_t j = i + kPrefixLen;

  while (j < n && data[j] != '\n' && data[j] != ':') {
    ++j;
  }

  if (j >= n || data[j] != ':') {
    return false;
  }

  std::size_t dst_start = j + 1;
  dst_start = SkipWhitespace(data, n, dst_start);

  if (dst_start >= n) {
    return false;
  }

  if (data[dst_start] != '0') {
    return false;
  }

  std::size_t after = dst_start + 1;
  after = SkipWhitespace(data, n, after);

  return after < n && data[after] == ':';
}

void SkipToNextLine(const char *data, std::size_t n, std::size_t &i) {
  while (i < n && data[i] != '\n') {
    ++i;
  }
  if (i < n && data[i] == '\n') {
    ++i;
  }
}

int CountDstZeroFromBuffer(const InType &buf) {
  const char *data = buf.empty() ? nullptr : buf.data();
  std::size_t n = buf.size();
  if (n == 0 || data == nullptr) {
    return 0;
  }

  int count = 0;
  std::size_t i = 0;

  while (i < n) {
    if (data[i] == 's') {
      std::size_t current_pos = i;
      if (ProcessLineForZeroDst(data, n, current_pos)) {
        ++count;
      }
      SkipToNextLine(data, n, i);
      continue;
    }

    if (data[i] == '\n') {
      ++i;
      continue;
    }

    SkipToNextLine(data, n, i);
  }

  return count;
}

}  // namespace

AkimovIStarSEQ::AkimovIStarSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool AkimovIStarSEQ::ValidationImpl() {
  return !GetInput().empty();
}

bool AkimovIStarSEQ::PreProcessingImpl() {
  input_buffer_ = GetInput();
  received_count_ = CountDstZeroFromBuffer(input_buffer_);
  return true;
}

bool AkimovIStarSEQ::RunImpl() {
  GetOutput() = received_count_;
  return true;
}

bool AkimovIStarSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace akimov_i_star
