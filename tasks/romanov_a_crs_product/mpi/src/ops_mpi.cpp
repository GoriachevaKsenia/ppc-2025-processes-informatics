#include "romanov_a_crs_product/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <utility>
#include <vector>

#include "romanov_a_crs_product/common/include/common.hpp"

namespace romanov_a_crs_product {

RomanovACRSProductMPI::RomanovACRSProductMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = CRS(static_cast<uint64_t>(0));
}

bool RomanovACRSProductMPI::ValidationImpl() {
  return (std::get<0>(GetInput()).GetCols() == std::get<1>(GetInput()).GetRows());
}

bool RomanovACRSProductMPI::PreProcessingImpl() {
  return true;
}

namespace {

void BroadcastCRS(CRS &m, int root, MPI_Comm comm) {
  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  std::array<uint64_t, 2> dims{m.n, m.m};
  MPI_Bcast(dims.data(), dims.size(), MPI_UINT64_T, root, comm);

  if (rank != root) {
    m.n = dims[0];
    m.m = dims[1];
    m.row_index.resize(m.n + 1);
  }

  uint64_t nnz = m.Nnz();
  MPI_Bcast(&nnz, 1, MPI_UINT64_T, root, comm);

  if (rank != root) {
    m.value.resize(nnz);
    m.column.resize(nnz);
  }

  if (nnz > 0) {
    MPI_Bcast(m.value.data(), static_cast<int>(nnz), MPI_DOUBLE, root, comm);
    if (nnz > 0) {
      MPI_Bcast(reinterpret_cast<uint64_t *>(m.column.data()), static_cast<int>(nnz), MPI_UINT64_T, root, comm);
    }
  }

  if (m.n + 1 > 0) {
    MPI_Bcast(reinterpret_cast<uint64_t *>(m.row_index.data()), static_cast<int>(m.n + 1), MPI_UINT64_T, root, comm);
  }
}

void SendCRS(const CRS &m, int dest, int tag, MPI_Comm comm) {
  std::array<uint64_t, 2> dims{m.n, m.m};
  MPI_Send(dims.data(), dims.size(), MPI_UINT64_T, dest, tag, comm);

  uint64_t nnz = m.Nnz();
  MPI_Send(&nnz, 1, MPI_UINT64_T, dest, tag + 1, comm);

  if (nnz > 0) {
    MPI_Send(m.value.data(), static_cast<int>(nnz), MPI_DOUBLE, dest, tag + 2, comm);
    if (nnz > 0) {
      MPI_Send(reinterpret_cast<const uint64_t *>(m.column.data()), static_cast<int>(nnz), MPI_UINT64_T, dest, tag + 3,
               comm);
    }
  }

  if (m.n + 1 > 0) {
    MPI_Send(reinterpret_cast<const uint64_t *>(m.row_index.data()), static_cast<int>(m.n + 1), MPI_UINT64_T, dest,
             tag + 4, comm);
  }
}

void RecvCRS(CRS &m, int src, int tag, MPI_Comm comm) {
  std::array<uint64_t, 2> dims{};
  MPI_Recv(dims.data(), dims.size(), MPI_UINT64_T, src, tag, comm, MPI_STATUS_IGNORE);
  m.n = dims[0];
  m.m = dims[1];

  uint64_t nnz = 0;
  MPI_Recv(&nnz, 1, MPI_UINT64_T, src, tag + 1, comm, MPI_STATUS_IGNORE);

  m.value.resize(nnz);
  m.column.resize(nnz);
  m.row_index.resize(m.n + 1);

  if (nnz > 0) {
    MPI_Recv(m.value.data(), static_cast<int>(nnz), MPI_DOUBLE, src, tag + 2, comm, MPI_STATUS_IGNORE);
    if (nnz > 0) {
      MPI_Recv(reinterpret_cast<uint64_t *>(m.column.data()), static_cast<int>(nnz), MPI_UINT64_T, src, tag + 3, comm,
               MPI_STATUS_IGNORE);
    }
  }

  if (m.n + 1 > 0) {
    MPI_Recv(reinterpret_cast<uint64_t *>(m.row_index.data()), static_cast<int>(m.n + 1), MPI_UINT64_T, src, tag + 4,
             comm, MPI_STATUS_IGNORE);
  }
}

}  // namespace

bool RomanovACRSProductMPI::RunImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int num_processes = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

  CRS a;
  CRS b;

  if (rank == 0) {
    a = std::get<0>(GetInput());
    b = std::get<1>(GetInput());
  }

  BroadcastCRS(b, 0, MPI_COMM_WORLD);

  CRS a_local;

  if (rank == 0) {
    uint64_t n = a.n;
    uint64_t rows_per_proc = (n + num_processes - 1) / num_processes;

    for (int pn = 1; pn < num_processes; pn++) {
      uint64_t start = pn * rows_per_proc;
      if (start >= n) {
        CRS empty(0, a.m);
        SendCRS(empty, pn, 100 + pn, MPI_COMM_WORLD);
        continue;
      }
      uint64_t end = std::min(n, start + rows_per_proc);
      CRS part = a.ExtractRows(start, end);
      SendCRS(part, pn, 100 + pn, MPI_COMM_WORLD);
    }

    uint64_t end0 = std::min(a.n, rows_per_proc);
    a_local = a.ExtractRows(0, end0);

  } else {
    RecvCRS(a_local, 0, 100 + rank, MPI_COMM_WORLD);
  }

  CRS c_local;
  if (a_local.n > 0 && b.n > 0) {
    c_local = a_local * b;
  } else {
    c_local = CRS(0, b.m);
  }

  CRS c_total;

  if (rank == 0) {
    std::vector<CRS> parts;
    parts.reserve(static_cast<uint64_t>(num_processes));

    parts.push_back(std::move(c_local));

    for (int pn = 1; pn < num_processes; pn++) {
      CRS temp;
      RecvCRS(temp, pn, 200 + pn, MPI_COMM_WORLD);
      parts.push_back(std::move(temp));
    }

    c_total = CRS::ConcatRows(parts);

  } else {
    SendCRS(c_local, 0, 200 + rank, MPI_COMM_WORLD);
  }

  BroadcastCRS(c_total, 0, MPI_COMM_WORLD);

  GetOutput() = c_total;

  return true;
}

bool RomanovACRSProductMPI::PostProcessingImpl() {
  return true;
}

}  // namespace romanov_a_crs_product
