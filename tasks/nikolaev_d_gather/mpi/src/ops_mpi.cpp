#include "nikolaev_d_gather/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <utility>
#include <vector>

#include "nikolaev_d_gather/common/include/common.hpp"

namespace nikolaev_d_gather {

namespace {
int GetTypeSize(MPI_Datatype datatype) {
  if (datatype == MPI_INT) {
    return sizeof(int);
  }
  if (datatype == MPI_FLOAT) {
    return sizeof(float);
  }
  if (datatype == MPI_DOUBLE) {
    return sizeof(double);
  }
  return 0;
}
}  // namespace

NikolaevDGatherMPI::NikolaevDGatherMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool NikolaevDGatherMPI::ValidationImpl() {
  const auto &input = GetInput();

  if (input.count <= 0) {
    return false;
  }

  if (input.datatype != MPI_INT && input.datatype != MPI_FLOAT && input.datatype != MPI_DOUBLE) {
    return false;
  }

  if (input.root < 0) {
    return false;
  }

  const int type_size = GetTypeSize(input.datatype);
  if (type_size <= 0) {
    return false;
  }

  const size_t expected_size = static_cast<size_t>(input.count) * static_cast<size_t>(type_size);

  if (input.data.size() != expected_size) {
    return false;
  }

  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  return input.root < size;
}

bool NikolaevDGatherMPI::PreProcessingImpl() {
  return true;
}

namespace {

bool CheckGatherArgs(int sendcount, int recvcount, MPI_Datatype sendtype, MPI_Datatype recvtype, int rank, int root,
                     const void *recvbuf) {
  if (sendcount != recvcount) {
    return false;
  }
  if (sendtype != recvtype) {
    return false;
  }
  if (rank == root && recvbuf == nullptr) {
    return false;
  }
  return true;
}

int GetBlockSize(int sendcount, MPI_Datatype datatype) {
  int type_size = 0;
  MPI_Type_size(datatype, &type_size);
  return sendcount * type_size;
}

void InitLocalData(const void *sendbuf, int block_sz, int rank, std::vector<char> &data, std::vector<int> &ranks) {
  data.resize(static_cast<size_t>(block_sz));
  ranks.clear();
  ranks.push_back(rank);

  const auto *send_ptr = static_cast<const char *>(sendbuf);
  std::copy(send_ptr, send_ptr + block_sz, data.begin());
}

void ReceiveFromChild(int sender_rank, int block_sz, MPI_Comm comm, std::vector<char> &data, std::vector<int> &ranks) {
  int num_ranks = 0;
  MPI_Recv(&num_ranks, 1, MPI_INT, sender_rank, 100, comm, MPI_STATUS_IGNORE);

  const size_t recv_size = static_cast<size_t>(num_ranks) * static_cast<size_t>(block_sz);

  std::vector<char> recv_data(recv_size);
  MPI_Recv(recv_data.data(), static_cast<int>(recv_size), MPI_BYTE, sender_rank, 101, comm, MPI_STATUS_IGNORE);

  std::vector<int> recv_ranks(static_cast<size_t>(num_ranks));
  MPI_Recv(recv_ranks.data(), num_ranks, MPI_INT, sender_rank, 102, comm, MPI_STATUS_IGNORE);

  data.insert(data.end(), recv_data.begin(), recv_data.end());
  ranks.insert(ranks.end(), recv_ranks.begin(), recv_ranks.end());
}

void SendToParent(int receiver_rank, int block_sz, MPI_Comm comm, const std::vector<char> &data,
                  const std::vector<int> &ranks) {
  const int num_ranks = static_cast<int>(ranks.size());

  MPI_Send(&num_ranks, 1, MPI_INT, receiver_rank, 100, comm);
  MPI_Send(data.data(), num_ranks * block_sz, MPI_BYTE, receiver_rank, 101, comm);
  MPI_Send(ranks.data(), num_ranks, MPI_INT, receiver_rank, 102, comm);
}

void AssembleAtRoot(const std::vector<char> &data, const std::vector<int> &ranks, int size, int block_sz,
                    void *recvbuf) {
  auto *out = static_cast<char *>(recvbuf);
  std::vector<char> full_data(static_cast<size_t>(size) * static_cast<size_t>(block_sz), 0);

  for (size_t i = 0; i < ranks.size(); ++i) {
    const int r = ranks[i];
    if (r >= 0 && r < size) {
      const size_t src_offset = i * static_cast<size_t>(block_sz);
      const size_t dest_offset = static_cast<size_t>(r) * static_cast<size_t>(block_sz);
      const auto len = static_cast<size_t>(block_sz);

      std::copy(data.begin() + static_cast<std::ptrdiff_t>(src_offset),
                data.begin() + static_cast<std::ptrdiff_t>(src_offset + len),
                full_data.begin() + static_cast<std::ptrdiff_t>(dest_offset));
    }
  }

  std::ranges::copy(full_data, out);
}

int TreeGatherImpl(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount,
                   MPI_Datatype recvtype, int root, MPI_Comm comm) {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  if (!CheckGatherArgs(sendcount, recvcount, sendtype, recvtype, rank, root, recvbuf)) {
    return MPI_ERR_ARG;
  }

  const int block_sz = GetBlockSize(sendcount, sendtype);
  const int rel_rank = (rank - root + size) % size;

  std::vector<char> current_data;
  std::vector<int> current_ranks;
  InitLocalData(sendbuf, block_sz, rank, current_data, current_ranks);

  int step = 1;
  while (step < size) {
    if (rel_rank % (2 * step) == 0) {
      const int sender_rel = rel_rank + step;
      if (sender_rel < size) {
        const int sender_rank = (sender_rel + root) % size;
        ReceiveFromChild(sender_rank, block_sz, comm, current_data, current_ranks);
      }
    } else {
      const int receiver_rel = rel_rank - step;
      const int receiver_rank = (receiver_rel + root) % size;
      SendToParent(receiver_rank, block_sz, comm, current_data, current_ranks);
      break;
    }
    step *= 2;
  }

  if (rank == root) {
    AssembleAtRoot(current_data, current_ranks, size, block_sz, recvbuf);
  }

  return MPI_SUCCESS;
}
}  // namespace

bool NikolaevDGatherMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &input = GetInput();

  const int type_size = GetTypeSize(input.datatype);

  std::vector<char> recv_buffer;
  if (rank == input.root) {
    recv_buffer.resize(static_cast<size_t>(input.count) * static_cast<size_t>(size) * static_cast<size_t>(type_size));
  }

  int result =
      TreeGatherImpl(input.data.data(), input.count, input.datatype, rank == input.root ? recv_buffer.data() : nullptr,
                     input.count, input.datatype, input.root, MPI_COMM_WORLD);

  if (result != MPI_SUCCESS) {
    return false;
  }

  if (rank == input.root) {
    GetOutput() = std::move(recv_buffer);
  } else {
    GetOutput() = std::vector<char>();
  }

  return true;
}

bool NikolaevDGatherMPI::PostProcessingImpl() {
  return true;
}

}  // namespace nikolaev_d_gather
