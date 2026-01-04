#include "akimov_i_star/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include "akimov_i_star/common/include/common.hpp"

namespace akimov_i_star {

namespace {

std::string ToString(const InType &buf) {
  return {buf.begin(), buf.end()};
}

inline void Trim(std::string &s) {
  const char *ws = " \t\n\r\f\v";
  auto first = s.find_first_not_of(ws);
  if (first == std::string::npos) {
    s.clear();
    return;
  }
  auto last = s.find_last_not_of(ws);
  s = s.substr(first, last - first + 1);
}

std::vector<AkimovIStarMPI::Op> ParseOpsFromString(const std::string &s) {
  std::vector<AkimovIStarMPI::Op> res;
  std::istringstream ss(s);
  std::string line;
  while (std::getline(ss, line)) {
    Trim(line);
    if (line.empty()) {
      continue;
    }
    const std::string prefix = "send:";
    if (!line.starts_with(prefix)) {
      continue;
    }
    std::string rest = line.substr(prefix.size());
    size_t p1 = rest.find(':');
    if (p1 == std::string::npos) {
      continue;
    }
    size_t p2 = rest.find(':', p1 + 1);
    if (p2 == std::string::npos) {
      continue;
    }
    std::string srcs = rest.substr(0, p1);
    std::string dsts = rest.substr(p1 + 1, p2 - (p1 + 1));
    std::string msg = rest.substr(p2 + 1);
    Trim(srcs);
    Trim(dsts);
    Trim(msg);
    try {
      int src = std::stoi(srcs);
      int dst = std::stoi(dsts);
      res.push_back(AkimovIStarMPI::Op{.src = src, .dst = dst, .msg = msg});
    } catch (...) {
      continue;
    }
  }
  return res;
}

int CountDstZero(const std::vector<AkimovIStarMPI::Op> &ops) {
  int cnt = 0;
  for (const auto &op : ops) {
    if (op.dst == 0) {
      ++cnt;
    }
  }
  return cnt;
}

void SendOutgoingToCenter(int myrank, const std::vector<AkimovIStarMPI::Op> &ops) {
  const int center = 0;
  for (const auto &op : ops) {
    if (op.src != myrank) {
      continue;
    }
    std::array<int, 2> header{op.dst, static_cast<int>(op.msg.size())};
    MPI_Send(header.data(), static_cast<int>(header.size()), MPI_INT, center, 0, MPI_COMM_WORLD);
    if (header[1] > 0) {
      MPI_Send(op.msg.data(), header[1], MPI_CHAR, center, 0, MPI_COMM_WORLD);
    }
  }
}

int ReceiveForwardedFromCenter(int expected) {
  int recvd = 0;
  const int center = 0;
  for (int i = 0; i < expected; ++i) {
    std::array<int, 2> header{0, 0};
    MPI_Recv(header.data(), static_cast<int>(header.size()), MPI_INT, center, MPI_ANY_TAG, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    int payload_len = header[1];
    std::string payload;
    payload.resize(payload_len);
    if (payload_len > 0) {
      MPI_Recv(payload.data(), payload_len, MPI_CHAR, center, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    ++recvd;
  }
  return recvd;
}

void CenterProcessLocalOutgoing(const std::vector<AkimovIStarMPI::Op> &ops, int &received_count) {
  const int center = 0;
  for (const auto &op : ops) {
    if (op.src != center) {
      continue;
    }
    if (op.dst == center) {
      ++received_count;
    } else {
      std::array<int, 2> header{op.src, static_cast<int>(op.msg.size())};
      MPI_Send(header.data(), static_cast<int>(header.size()), MPI_INT, op.dst, 0, MPI_COMM_WORLD);
      if (header[1] > 0) {
        MPI_Send(op.msg.data(), header[1], MPI_CHAR, op.dst, 0, MPI_COMM_WORLD);
      }
    }
  }
}

void CenterReceiveAndForward(int recv_from_others, int &received_count) {
  const int center = 0;
  for (int i = 0; i < recv_from_others; ++i) {
    std::array<int, 2> header{0, 0};
    MPI_Recv(header.data(), static_cast<int>(header.size()), MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    int dst = header[0];
    int payload_len = header[1];
    std::string payload;
    payload.resize(payload_len);
    if (payload_len > 0) {
      MPI_Recv(payload.data(), payload_len, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (dst == center) {
      ++received_count;
    } else {
      std::array<int, 2> fwd_header{0, payload_len};
      MPI_Send(fwd_header.data(), static_cast<int>(fwd_header.size()), MPI_INT, dst, 0, MPI_COMM_WORLD);
      if (payload_len > 0) {
        MPI_Send(payload.data(), payload_len, MPI_CHAR, dst, 0, MPI_COMM_WORLD);
      }
    }
  }
}

int CountMessagesForCenterFromInput(const InType &input) {
  int total_expected_for_center = 0;
  std::string s(input.begin(), input.end());
  std::istringstream ss(s);
  std::string line;
  const std::string prefix = "send:";

  while (std::getline(ss, line)) {
    size_t start = line.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
      continue;
    }
    size_t end = line.find_last_not_of(" \t\r\n");
    std::string t = line.substr(start, end - start + 1);
    if (!t.starts_with(prefix)) {
      continue;
    }
    std::string rest = t.substr(prefix.size());
    size_t p1 = rest.find(':');
    if (p1 == std::string::npos) {
      continue;
    }
    size_t p2 = rest.find(':', p1 + 1);
    if (p2 == std::string::npos) {
      continue;
    }
    std::string dsts = rest.substr(p1 + 1, p2 - (p1 + 1));
    size_t sd = dsts.find_first_not_of(" \t\r\n");
    if (sd == std::string::npos) {
      continue;
    }
    size_t ed = dsts.find_last_not_of(" \t\r\n");
    std::string dsttrim = dsts.substr(sd, ed - sd + 1);
    try {
      int dst = std::stoi(dsttrim);
      if (dst == 0) {
        ++total_expected_for_center;
      }
    } catch (...) {
      continue;
    }
  }

  return total_expected_for_center;
}

void ProcessMultiProcMode(int rank, const std::vector<AkimovIStarMPI::Op> &ops, int &received_count) {
  int local_send_count = 0;
  int local_expected_recv = 0;

  for (const auto &op : ops) {
    if (op.src == rank) {
      ++local_send_count;
    }
    if (op.dst == rank) {
      ++local_expected_recv;
    }
  }

  int total_sends = 0;
  MPI_Allreduce(&local_send_count, &total_sends, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  const int center = 0;

  if (rank != center) {
    SendOutgoingToCenter(rank, ops);
    received_count = ReceiveForwardedFromCenter(local_expected_recv);
  } else {
    CenterProcessLocalOutgoing(ops, received_count);
    int center_local_sends = 0;
    for (const auto &op : ops) {
      if (op.src == center) {
        ++center_local_sends;
      }
    }
    int recv_from_others = total_sends - center_local_sends;
    CenterReceiveAndForward(recv_from_others, received_count);
  }
}

}  // namespace

AkimovIStarMPI::AkimovIStarMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool AkimovIStarMPI::ValidationImpl() {
  int rank = 0;
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized == 0) {
    return true;
  }
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    return !GetInput().empty();
  }
  return true;
}

bool AkimovIStarMPI::PreProcessingImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  input_buffer_.clear();
  ops_.clear();
  received_count_ = 0;

  std::string raw;
  if (rank == 0) {
    raw = ToString(GetInput());
  }

  int len = static_cast<int>(raw.size());
  MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
  len = std::max(len, 0);
  std::string buf;
  buf.resize(len);
  MPI_Bcast((len != 0) ? buf.data() : nullptr, len, MPI_CHAR, 0, MPI_COMM_WORLD);

  input_buffer_.clear();
  if (len > 0) {
    input_buffer_.assign(buf.begin(), buf.end());
  }

  std::string parsed = ToString(input_buffer_);
  ops_ = ParseOpsFromString(parsed);

  return true;
}

bool AkimovIStarMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size == 1) {
    received_count_ = CountDstZero(ops_);
  } else {
    ProcessMultiProcMode(rank, ops_, received_count_);
  }

  int total_expected_for_center = 0;
  if (rank == 0) {
    total_expected_for_center = CountMessagesForCenterFromInput(GetInput());
  }

  MPI_Bcast(&total_expected_for_center, 1, MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput() = total_expected_for_center;

  return true;
}

bool AkimovIStarMPI::PostProcessingImpl() {
  return true;
}

}  // namespace akimov_i_star
