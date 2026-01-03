#pragma once

#include <mpi.h>

#include <cmath>
#include <memory>
#include <stdexcept>
namespace guseva_a_hypercube {

class Hypercube {
 private:
  int rank_ = 0;
  int size_ = 0;
  int ndims_ = 0;
  MPI_Comm comm_{};

  explicit Hypercube(MPI_Comm comm) : comm_(comm) {
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &size_);
    // NOLINTNEXTLINE // to supress "ndims_ should be initialized in a member initializer of the constructor"
    ndims_ = floor(log2(size_));
  }

 public:
  static std::unique_ptr<Hypercube> Create() {
    int world_size = 0;
    int world_rank = 0;

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int ndims = floor(log2(world_size));
    int real_size = static_cast<int>(pow(2, ndims));

    int color = (world_rank < real_size) ? 0 : MPI_UNDEFINED;
    MPI_Comm hypercube_comm{};
    MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &hypercube_comm);

    if (hypercube_comm == MPI_COMM_NULL) {
      return nullptr;
    }

    return std::unique_ptr<Hypercube>(new Hypercube(hypercube_comm));
  }

  [[nodiscard]] int GetRank() const {
    return rank_;
  }

  [[nodiscard]] int GetSize() const {
    return size_;
  }

  template <typename _datatype>
  void ReduceSum(_datatype local_value, MPI_Datatype mpi_datatype, _datatype &result_buf, int root = 0) const {
    if (comm_ == MPI_COMM_NULL) {
      return;
    }

    _datatype current_value = local_value;
    for (int dim = 0; dim < ndims_; ++dim) {
      int neighbor = rank_ ^ (1 << dim);

      _datatype received_value = 0;
      MPI_Status status;
      if ((rank_ >> dim) & 1) {
        MPI_Send(&current_value, 1, mpi_datatype, neighbor, dim, comm_);
        MPI_Recv(&received_value, 1, mpi_datatype, neighbor, dim, comm_, &status);
      } else {
        MPI_Recv(&received_value, 1, mpi_datatype, neighbor, dim, comm_, &status);
        MPI_Send(&current_value, 1, mpi_datatype, neighbor, dim, comm_);
      }

      current_value += received_value;
    }
    if (rank_ == root) {
      result_buf = current_value;
    }
  }

  template <typename _datatype>
  void SendRecv(int sender, int receiver, _datatype message, MPI_Datatype mpi_datatype, _datatype &dst_buff) const {
    if (sender >= size_ || receiver >= size_) {
      throw std::invalid_argument("Sender or receiver is out of hypercube.");
    }
    int current = sender;
    _datatype current_msg = message;

    while (current != receiver) {
      int route = current ^ receiver;
      int dim = 0;
      while ((route & (1 << dim)) == 0) {
        dim++;
      }

      int next = current ^ (1 << dim);

      _datatype recv_msg = 0;
      MPI_Status status;

      if (rank_ == current) {
        MPI_Send(&current_msg, 1, mpi_datatype, next, 0, comm_);
      }
      if (rank_ == next) {
        MPI_Recv(&recv_msg, 1, mpi_datatype, current, 0, comm_, &status);
      }

      current = next;
      current_msg = recv_msg;
    }

    if (rank_ == receiver) {
      dst_buff = current_msg;
    }
  }

  ~Hypercube() {
    if (comm_ != MPI_COMM_NULL) {
      MPI_Comm_free(&comm_);
    }
  }
};
}  // namespace guseva_a_hypercube
