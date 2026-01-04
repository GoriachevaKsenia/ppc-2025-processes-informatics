#include "../include/linear_contrast_stretching_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "../../common/include/common.hpp"

namespace kutergin_v_linear_contrast_stretching {

LinearContrastStretchingMPI::LinearContrastStretchingMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());  // установка типа задачи
  GetInput() = in;                       // сохранение входных данных
  if (!in.data.empty()) {
    GetOutput().resize(in.data.size());  // инициализация выходных данных
  }
}

bool LinearContrastStretchingMPI::ValidationImpl() {
  return !GetInput().data.empty();  // пустое изображение невалидно
}

bool LinearContrastStretchingMPI::PreProcessingImpl() {
  return true;
}

bool LinearContrastStretchingMPI::RunImpl() {
  int process_rank = 0;
  int process_count = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &process_count);

  const int root = 0;
  size_t total_pixel_count = (process_rank == root) ? GetInput().data.size() : 0;

  const int base_chunk =
      static_cast<int>(total_pixel_count) / process_count;  // целое часть от деления числа пикселей на число процессов
  const int remainder =
      static_cast<int>(total_pixel_count) % process_count;  // остаток от деления числа пикселей на число процессов

  std::vector<int> send_counts(process_count);  // вектор чисел пикселей на каждый процесс
  std::vector<int> displs(process_count, 0);    // вектор смещений начал "кусков" на каждый процесс

  if (process_rank == root)  // root-процесс вычисляет кому сколько пикселей дать и какое будет смещение
  {
    for (int i = 0; i < process_count; ++i) {
      send_counts[i] = base_chunk + (i < remainder ? 1 : 0);
      if (i > 0) {
        displs[i] = displs[i - 1] + send_counts[i - 1];
      }
    }
  }

  int local_chunk_size = 0;
  MPI_Scatter(send_counts.data(), 1, MPI_INT, &local_chunk_size, 1, MPI_INT, root, MPI_COMM_WORLD);

  std::vector<unsigned char> local_chunk(
      local_chunk_size);  // локальный буфер для приема части вектора на каждом процессе

  const auto &image_in_full = GetInput().data;
  auto &image_out_full = GetOutput();

  MPI_Scatterv(image_in_full.data(), send_counts.data(), displs.data(), MPI_UNSIGNED_CHAR, local_chunk.data(),
               local_chunk_size, MPI_UNSIGNED_CHAR, root, MPI_COMM_WORLD);

  unsigned char local_min = 255;
  unsigned char local_max = 0;
  if (!local_chunk.empty()) {
    // поиск локальных минимумов и максимумов (на каждом куске вектора)
    const auto &minmax_it = std::ranges::minmax_element(local_chunk);
    local_min = *minmax_it.min;
    local_max = *minmax_it.max;
  }

  // получение глобального минимума и максимума (для всего вектора)
  unsigned char global_min = 0;
  unsigned char global_max = 0;
  MPI_Allreduce(&local_min, &global_min, 1, MPI_UNSIGNED_CHAR, MPI_MIN,
                MPI_COMM_WORLD);  // сбор минимума из всех local_min
  MPI_Allreduce(&local_max, &global_max, 1, MPI_UNSIGNED_CHAR, MPI_MAX,
                MPI_COMM_WORLD);  // сбор максимума из всех local_max

  if (global_min != global_max) {
    const double scale = 255.0 / (global_max - global_min);
    for (unsigned char &pixel : local_chunk) {
      double p_out = (pixel - global_min) * scale;
      pixel = static_cast<unsigned char>(std::round(p_out));
    }
  }

  MPI_Gatherv(local_chunk.data(), local_chunk_size, MPI_UNSIGNED_CHAR, image_out_full.data(), send_counts.data(),
              displs.data(), MPI_UNSIGNED_CHAR, root, MPI_COMM_WORLD);

  return true;
}

bool LinearContrastStretchingMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kutergin_v_linear_contrast_stretching
