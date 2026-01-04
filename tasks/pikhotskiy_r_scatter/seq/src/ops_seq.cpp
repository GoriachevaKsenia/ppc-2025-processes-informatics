#include "pikhotskiy_r_scatter/seq/include/ops_seq.hpp"

#include <mpi.h>

#include <cstddef>
#include <cstring>

#include "pikhotskiy_r_scatter/common/include/common.hpp"

namespace pikhotskiy_r_scatter {

PikhotskiyRScatterSEQ::PikhotskiyRScatterSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = nullptr;
}

size_t PikhotskiyRScatterSEQ::GetTypeSize(MPI_Datatype datatype) {
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

bool PikhotskiyRScatterSEQ::ValidationImpl() {
  const auto &input_data = GetInput();
  int send_element_count = std::get<1>(input_data);
  MPI_Datatype send_datatype = std::get<2>(input_data);
  void *receive_data_ptr = std::get<3>(input_data);
  int receive_element_count = std::get<4>(input_data);
  MPI_Datatype receive_datatype = std::get<5>(input_data);
  int root_process = std::get<6>(input_data);
  MPI_Comm comm = std::get<7>(input_data);

  (void)receive_datatype;
  (void)root_process;
  (void)comm;

  if (send_element_count < 0 || receive_element_count < 0) {
    return false;
  }

  if (send_element_count != receive_element_count) {
    return false;
  }

  if (send_datatype != receive_datatype) {
    return false;
  }

  if (receive_element_count > 0 && receive_data_ptr == nullptr) {
    return false;
  }

  return true;
}

bool PikhotskiyRScatterSEQ::PreProcessingImpl() {
  return true;
}

bool PikhotskiyRScatterSEQ::RunImpl() {
  const auto &input_data = GetInput();
  const void *send_data_ptr = std::get<0>(input_data);
  int send_element_count = std::get<1>(input_data);
  MPI_Datatype send_datatype = std::get<2>(input_data);
  void *receive_data_ptr = std::get<3>(input_data);
  int receive_element_count = std::get<4>(input_data);
  MPI_Datatype receive_datatype = std::get<5>(input_data);
  int root_process = std::get<6>(input_data);
  MPI_Comm comm = std::get<7>(input_data);

  (void)send_element_count;
  (void)receive_datatype;
  (void)root_process;
  (void)comm;

  size_t element_size = GetTypeSize(send_datatype);

  size_t total_copy_size = static_cast<size_t>(receive_element_count) * element_size;

  if (total_copy_size > 0) {
    if (send_data_ptr != nullptr) {
      std::memcpy(receive_data_ptr, send_data_ptr, total_copy_size);
    } else {
      std::memset(receive_data_ptr, 0, total_copy_size);
    }
  }

  GetOutput() = receive_data_ptr;

  return true;
}

bool PikhotskiyRScatterSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace pikhotskiy_r_scatter
