#include "votincev_d_qsort_batcher/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <vector>

#include "votincev_d_qsort_batcher/common/include/common.hpp"

namespace votincev_d_qsort_batcher {

VotincevDQsortBatcherMPI::VotincevDQsortBatcherMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

// убеждаемся, что массив не пустой
bool VotincevDQsortBatcherMPI::ValidationImpl() {
  const auto &vec = GetInput();
  return !vec.empty();
}

// препроцессинг (не нужен)
bool VotincevDQsortBatcherMPI::PreProcessingImpl() {
  return true;
}

// главный MPI метод
bool VotincevDQsortBatcherMPI::RunImpl() {
  // получаю кол-во процессов
  int proc_n = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &proc_n);

  // получаю ранг процесса
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // если процесс 1 - то просто как в SEQ
  if (proc_n == 1) {
    auto out = GetInput();
    if (!out.empty()) {
      QuickSort(out.data(), 0, static_cast<int>(out.size()) - 1);
    }
    GetOutput() = out;
    return true;
  }

  // размер массива знает только 0-й процесс
  int total_size = 0;
  if (rank == 0) {
    total_size = static_cast<int>(GetInput().size());
  }

  // рассылаем размер массива всем процессам
  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (total_size == 0) {
    return true;
  }

  // вычисление размеров и смещений
  std::vector<int> sizes;
  std::vector<int> offsets;
  ComputeDistribution(proc_n, total_size, sizes, offsets);

  // распределение данных (Scatter)
  std::vector<double> local(sizes[rank]);
  ScatterData(rank, sizes, offsets, local);

  // локальная сортировка
  // каждый процесс сортирует свой кусок
  if (!local.empty()) {
    QuickSort(local.data(), 0, static_cast<int>(local.size()) - 1);
  }

  // чет-нечет слияние Бэтчера
  BatcherMergeSort(rank, proc_n, sizes, local);

  // 0й процесс собирает результыт от других процессов
  GatherResult(rank, total_size, sizes, offsets, local);

  return true;
}

void VotincevDQsortBatcherMPI::ComputeDistribution(int proc_n, int total_size, std::vector<int> &sizes,
                                                   std::vector<int> &offsets) {
  int base = total_size / proc_n;   // минимальный размер блока
  int extra = total_size % proc_n;  // оставшиеся элементы

  sizes.resize(proc_n);
  offsets.resize(proc_n);

  // распределяем остаток по первым процессам (процессам с меньшим рангом)
  for (int i = 0; i < proc_n; i++) {
    sizes[i] = base + (i < extra ? 1 : 0);
  }

  //  смещения (начальный индекс для каждого блока)
  offsets[0] = 0;
  for (int i = 1; i < proc_n; i++) {
    offsets[i] = offsets[i - 1] + sizes[i - 1];
  }
}

void VotincevDQsortBatcherMPI::ScatterData(int rank, const std::vector<int> &sizes, const std::vector<int> &offsets,
                                           std::vector<double> &local) {
  //  GetInput().data() только для процесса с rank == 0
  if (rank == 0) {
    MPI_Scatterv(GetInput().data(), sizes.data(), offsets.data(), MPI_DOUBLE, local.data(), sizes[rank], MPI_DOUBLE, 0,
                 MPI_COMM_WORLD);
  } else {
    MPI_Scatterv(nullptr, sizes.data(), offsets.data(), MPI_DOUBLE, local.data(), sizes[rank], MPI_DOUBLE, 0,
                 MPI_COMM_WORLD);
  }
}

int VotincevDQsortBatcherMPI::GetPartnerRank(int rank, int proc_n, int phase) {
  int partner = -1;
  // Классическая схема Odd-Even Transposition (Batcher-like для P процессов)
  if (phase % 2 == 0) {
    // Чётная фаза: (0,1), (2,3), (4,5)...
    if (rank % 2 == 0) {
      partner = rank + 1;
    } else {
      partner = rank - 1;
    }
  } else {
    // Нечётная фаза: (1,2), (3,4), (5,6)...
    if (rank % 2 == 1) {
      partner = rank + 1;
    } else {
      partner = rank - 1;
    }
  }

  if (partner < 0 || partner >= proc_n) {
    return -1;
  }
  return partner;
}

void VotincevDQsortBatcherMPI::PerformMergePhase(int rank, int partner, const std::vector<int> &sizes,
                                                 std::vector<double> &local, std::vector<double> &recv_buf,
                                                 std::vector<double> &merge_buf) {
  MPI_Sendrecv(local.data(), sizes[rank], MPI_DOUBLE, partner, 0, recv_buf.data(), sizes[partner], MPI_DOUBLE, partner,
               0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  std::merge(local.begin(), local.end(), recv_buf.begin(), recv_buf.begin() + sizes[partner], merge_buf.begin());

  // малые влево, большие вправо
  if (rank < partner) {
    // младший процесс забирает начало (наименьшие элементы)
    std::copy(merge_buf.begin(), merge_buf.begin() + sizes[rank], local.begin());
  } else {
    // старший процесс забирает конец (наибольшие элементы)
    std::copy(merge_buf.begin() + sizes[partner], merge_buf.begin() + sizes[partner] + sizes[rank], local.begin());
  }
}

void VotincevDQsortBatcherMPI::BatcherMergeSort(int rank, int proc_n, const std::vector<int> &sizes,
                                                std::vector<double> &local) {
  if (proc_n <= 1) {
    return;
  }

  int max_block = *std::ranges::max_element(sizes);
  std::vector<double> recv_buf(max_block);
  std::vector<double> merge_buf(sizes[rank] + max_block);

  // для полной сортировки в схеме Odd-Even требуется P фаз
  for (int phase = 0; phase < proc_n; phase++) {
    int partner = GetPartnerRank(rank, proc_n, phase);

    if (partner != -1) {
      PerformMergePhase(rank, partner, sizes, local, recv_buf, merge_buf);
    }
  }
}

// 0й процесс собирает данные со всех других процессов
void VotincevDQsortBatcherMPI::GatherResult(int rank, int total_size, const std::vector<int> &sizes,
                                            const std::vector<int> &offsets, const std::vector<double> &local) {
  if (rank == 0) {
    std::vector<double> result(total_size);

    // 0-й процесс собирает данные
    MPI_Gatherv(local.data(), sizes[rank], MPI_DOUBLE, result.data(), sizes.data(), offsets.data(), MPI_DOUBLE, 0,
                MPI_COMM_WORLD);

    GetOutput() = result;
  } else {
    // остальные процессы отправляют свои данные 0-му
    MPI_Gatherv(local.data(), sizes[rank], MPI_DOUBLE, nullptr, nullptr, nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
}

// partition (для qsort)
int VotincevDQsortBatcherMPI::Partition(double *arr, int l, int h) {
  int i = l;
  int j = h;
  double pivot = arr[(l + h) / 2];

  while (i <= j) {
    while (arr[i] < pivot) {
      i++;
    }
    while (arr[j] > pivot) {
      j--;
    }

    if (i <= j) {
      std::swap(arr[i], arr[j]);
      i++;
      j--;
    }
  }
  // i — это граница следующего правого подмассива
  return i;
}

// итеративная qsort
void VotincevDQsortBatcherMPI::QuickSort(double *arr, int left, int right) {
  std::vector<int> stack;

  stack.push_back(left);
  stack.push_back(right);

  while (!stack.empty()) {
    int h = stack.back();
    stack.pop_back();
    int l = stack.back();
    stack.pop_back();

    if (l >= h) {
      continue;
    }

    // вызываю Partition для разделения массива
    int p = Partition(arr, l, h);
    // p - это i после Partition. j находится на p-1 или p-2.

    // p - начало правого подмассива (i)
    // j - конец левого подмассива (j после Partition)

    // пересчитываю l и h для стека, используя внутренние границы Partition
    int l_end = p - 1;  // конец левого подмассива
    int r_start = p;    // начало правого подмассива

    // если левый подмассив существует
    if (l < l_end) {
      stack.push_back(l);
      stack.push_back(l_end);
    }

    // если правый подмассив существует
    if (r_start < h) {
      stack.push_back(r_start);
      stack.push_back(h);
    }
  }
}

// здесь ничего не надо делать
bool VotincevDQsortBatcherMPI::PostProcessingImpl() {
  return true;
}

}  // namespace votincev_d_qsort_batcher
