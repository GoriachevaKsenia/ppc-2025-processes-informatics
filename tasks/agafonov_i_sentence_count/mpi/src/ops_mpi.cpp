#include "agafonov_i_sentence_count/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cctype>
#include <string>

#include "agafonov_i_sentence_count/common/include/common.hpp"

namespace agafonov_i_sentence_count {

SentenceCountMPI::SentenceCountMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool SentenceCountMPI::ValidationImpl() {
  return true;
}

bool SentenceCountMPI::PreProcessingImpl() {
  return true;
}

int SentenceCountMPI::CalculateStart(int rank, int chunk, int rem) {
  return (rank * chunk) + std::min(rank, rem);
}

int SentenceCountMPI::CalculateEnd(int rank, int chunk, int rem) {
  int start = (rank * chunk) + std::min(rank, rem);
  return start + chunk + (rank < rem ? 1 : 0);
}

int SentenceCountMPI::CountSentences(const std::string &text, int start, int end, int total_length) {
  int count = 0;
  bool in_word = (start > 0 && std::isalnum(static_cast<unsigned char>(text[start - 1])) != 0);

  for (int i = start; i < end; ++i) {
    auto c = static_cast<unsigned char>(text[i]);
    if (std::isalnum(c) != 0) {
      in_word = true;
      continue;
    }
    if (in_word && (c == '.' || c == '!' || c == '?')) {
      if (c == '.' && i + 1 < total_length && text[i + 1] == '.') {
        continue;
      }
      count++;
      in_word = false;
    }
  }
  return count;
}

bool SentenceCountMPI::RunImpl() {
  int world_size = 0;
  int world_rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int total_length = (world_rank == 0) ? static_cast<int>(GetInput().length()) : 0;
  MPI_Bcast(&total_length, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (total_length == 0) {
    GetOutput() = 0;
    return true;
  }

  std::string text = (world_rank == 0) ? GetInput() : std::string(total_length, ' ');
  MPI_Bcast(const_cast<char *>(text.data()), total_length, MPI_CHAR, 0, MPI_COMM_WORLD);

  int start = CalculateStart(world_rank, world_size, total_length);
  int end = CalculateEnd(world_rank, world_size, total_length);

  int local_count = CountSentences(text, start, end, total_length);
  int global_count = 0;
  MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  if (std::isalnum(static_cast<unsigned char>(text[total_length - 1])) != 0) {
    global_count++;
  }

  GetOutput() = global_count;
  return true;
}

bool SentenceCountMPI::PostProcessingImpl() {
  return true;
}

}  // namespace agafonov_i_sentence_count
