#include "luzan_e_matrix_horis_rib_mult_sheme/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <tuple>
#include <vector>

#include "luzan_e_matrix_horis_rib_mult_sheme/common/include/common.hpp"

namespace luzan_e_matrix_horis_rib_mult_sheme {

LuzanEMatrixHorisRibMultShemeMPI::LuzanEMatrixHorisRibMultShemeMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetOutput() = {};

  // saving matrix only if it's rank=0
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    GetInput() = in;
  } else {
    GetInput() = {};
  }
}

bool LuzanEMatrixHorisRibMultShemeMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) {
    return true;
  }
  bool res = true;
  int height = std::get<1>(GetInput());
  int width = std::get<2>(GetInput());
  int vec_height = std::get<4>(GetInput());

  // matrix check
  res = std::get<0>(GetInput()).size() == static_cast<size_t>(height) * static_cast<size_t>(width) && height > 0 &&
        width > 0;

  // vec check
  res = res && std::get<3>(GetInput()).size() == static_cast<size_t>(vec_height);

  // matrix & vec sizes cmp.
  res = res && (width == vec_height);
  return res;
}

bool LuzanEMatrixHorisRibMultShemeMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank != 0) {
    return true;
  }

  int height = std::get<1>(GetInput());

  GetOutput().resize(height);
  for (int cell = 0; cell < height; cell++) {
    GetOutput()[cell] = 0;
  }

  return true;
}

bool LuzanEMatrixHorisRibMultShemeMPI::RunImpl() {
  int height = 0;
  int width = 0;
  int vec_len = 0;
  std::vector<int> mat(0);
  std::vector<int> vec(0);

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // sharing matrix sizes
  if (rank == 0) {
    // mat = std::get<0>(GetInput());
    height = std::get<1>(GetInput());
    width = std::get<2>(GetInput());
    vec = std::get<3>(GetInput());
    vec_len = std::get<4>(GetInput());
  }
  MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&vec_len, 1, MPI_INT, 0, MPI_COMM_WORLD);

  vec.resize(vec_len);
  MPI_Bcast(vec.data(), vec_len, MPI_INT, 0, MPI_COMM_WORLD);

  // calculating shifts & rows_per_proc (only about rows rigth now)
  int rest = height % size;
  std::vector<int> shift(size);
  std::vector<int> per_proc(size, height / size);  // rows per proc

  shift[0] = 0;
  int accumulator = 0;
  for (int i = 0; i < size; i++) {
    if (rest != 0) {
      per_proc[i]++;
      rest--;
    }
    shift[i] = accumulator;
    accumulator = per_proc[i] + shift[i];
  }

  // preparing to recieve data
  std::vector<int> recv(static_cast<size_t>(per_proc[rank] * width));
  std::vector<int> shift_elem(size);
  std::vector<int> per_proc_elem(size, height / size);  // rows per proc

  for (int i = 0; i < size; i++) {
    per_proc_elem[i] = per_proc[i] * width;
    shift_elem[i] = shift[i] * width;
  }

  MPI_Scatterv(std::get<0>(GetInput()).data(), per_proc_elem.data(), shift_elem.data(), MPI_INT, recv.data(),
               per_proc_elem[rank], MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> prod(static_cast<size_t>(per_proc[rank]), 0);  // sums
  int rows_to_calc = static_cast<int>(per_proc[rank]);
  for (int row = 0; row < rows_to_calc; row++) {
    int row_step = row * width;
    int sum = 0;
    for (int col = 0; col < width; col++) {
      sum += recv[row_step + col] * vec[col];
    }
    prod[row] = sum;
  }

  std::vector<int> fin_prod(height);

  MPI_Gatherv(prod.data(), rows_to_calc, MPI_INT, fin_prod.data(), per_proc.data(), shift.data(), MPI_INT, 0,
              MPI_COMM_WORLD);
  MPI_Bcast(fin_prod.data(), height, MPI_INT, 0, MPI_COMM_WORLD);
  GetOutput() = fin_prod;
  return true;
}

bool LuzanEMatrixHorisRibMultShemeMPI::PostProcessingImpl() {
  return true;
}

}  // namespace luzan_e_matrix_horis_rib_mult_sheme
