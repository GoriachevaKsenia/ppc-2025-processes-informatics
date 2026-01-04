#include "gusev_d_sentence_count/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <string>
#include <vector>

#include "gusev_d_sentence_count/common/include/common.hpp"

namespace gusev_d_sentence_count {

namespace {

bool IsTerminator(char c) {
  return c == '.' || c == '!' || c == '?';
}

size_t CountSentencesInChunk(const std::vector<char> &local_chunk, int chunk_size) {
  size_t sentence_count = 0;

  for (int i = 0; i < chunk_size; ++i) {
    if (IsTerminator(local_chunk[i])) {
      if (!IsTerminator(local_chunk[i + 1])) {
        sentence_count++;
      }
    }
  }
  return sentence_count;
}

}  // namespace

GusevDSentenceCountMPI::GusevDSentenceCountMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool GusevDSentenceCountMPI::ValidationImpl() {
  return true;
}

bool GusevDSentenceCountMPI::PreProcessingImpl() {
  return true;
}

bool GusevDSentenceCountMPI::RunImpl() {
  int rank = 0;
  int comm_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  std::string &text_data = GetInput();

  if (text_data.empty()) {
    GetOutput() = 0;
    return true;
  }

  int chunk_size = 0;
  if (rank == 0) {
    size_t length = text_data.size();

    if (length % comm_size != 0) {
      size_t pad = comm_size - (length % comm_size);
      text_data.resize(length + pad, ' ');
      length = text_data.size();
    }

    text_data.push_back(' ');

    chunk_size = static_cast<int>(length / comm_size);
  }

  MPI_Bcast(&chunk_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<char> local_buffer(chunk_size + 1);
  std::vector<int> counts(comm_size, chunk_size + 1);
  std::vector<int> offsets(comm_size);

  if (rank == 0) {
    for (int i = 0; i < comm_size; ++i) {
      offsets[i] = i * chunk_size;
    }
  }

  MPI_Scatterv(text_data.data(), counts.data(), offsets.data(), MPI_CHAR, local_buffer.data(), chunk_size + 1, MPI_CHAR,
               0, MPI_COMM_WORLD);

  size_t local_res = CountSentencesInChunk(local_buffer, chunk_size);

  size_t total_res = 0;
  MPI_Reduce(&local_res, &total_res, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    GetOutput() = total_res;
  }

  return true;
}

bool GusevDSentenceCountMPI::PostProcessingImpl() {
  return true;
}

}  // namespace gusev_d_sentence_count
