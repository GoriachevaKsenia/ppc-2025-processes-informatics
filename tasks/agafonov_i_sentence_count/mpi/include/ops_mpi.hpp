#pragma once

#include <string>

#include "agafonov_i_sentence_count/common/include/common.hpp"
#include "task/include/task.hpp"

namespace agafonov_i_sentence_count {

class SentenceCountMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit SentenceCountMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static int CountSentences(const std::string &text, int start, int end, int total_length);
  static int CalculateStart(int rank, int chunk, int rem);
  static int CalculateEnd(int rank, int chunk, int rem);
};

}  // namespace agafonov_i_sentence_count
