#include "../include/reduce_mpi.hpp"

#include <mpi.h>

#include <cstdint>
#include <cstring>
#include <numeric>

#include "../../common/include/common.hpp"

namespace kutergin_v_reduce {

namespace {

// вспомогательная функция для применения операции op к двум буферам a и b
void ApplyOp(void *a, const void *b, int count, MPI_Datatype datatype, MPI_Op op) {
  if (op == MPI_SUM && datatype == MPI_INT) {
    for (int i = 0; i < count; ++i) {
      reinterpret_cast<int *>(a)[i] += reinterpret_cast<const int *>(b)[i];
    }
  }
}

}  // namespace

ReduceMPI::ReduceMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());  // установка типа задачи
  GetInput() = in;                       // сохранение входных данных
  GetOutput() = 0;                       // инициализация выходных данных
}

int Reduce(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm) {
  int process_rank = 0;
  int process_count = 0;
  MPI_Comm_rank(comm, &process_rank);
  MPI_Comm_size(comm, &process_count);

  if (root < 0 || root >= process_count)  // root-процесс не существует
  {
    return MPI_ERR_ROOT;  // возвращение стандартного кода ошибки MPI
  }

  int type_size = 0;
  MPI_Type_size(datatype, &type_size);

  // Древовидный сбор
  for (int mask = 1; mask < process_count;
       mask <<= 1)  // удвоение битовой маски на каждой итерации посредством битового сдвига
  {
    if ((process_rank & mask) != 0)  // процессы-отправители
    {
      MPI_Send(sendbuf, count, datatype, process_rank - mask, 0, comm);
      break;
    }

    if (process_rank + mask < process_count)  // процессы-получатели
    {
      auto *recv_temp = new uint8_t[static_cast<size_t>(count) * type_size];
      MPI_Recv(recv_temp, count, datatype, process_rank + mask, 0, comm, MPI_STATUS_IGNORE);

      ApplyOp(sendbuf, recv_temp, count, datatype, op);  // выполнение op (MPI_SUM) для datatype (MPI_INT)

      delete[] recv_temp;
    }
  }

  // Результат с процесса 0 отправляется на процесс 'root'
  if (process_rank == 0 && root != 0) {
    MPI_Send(sendbuf, count, datatype, root, 0, comm);
  }

  // Корневой процесс 'root' получает финальный результат
  if (process_rank == root) {
    if (root == 0) {
      std::memcpy(recvbuf, sendbuf, static_cast<size_t>(count) * type_size);
    } else {
      MPI_Recv(recvbuf, count, datatype, 0, 0, comm, MPI_STATUS_IGNORE);
    }
  }

  return MPI_SUCCESS;
}

bool ReduceMPI::ValidationImpl() {
  return true;
}

bool ReduceMPI::PreProcessingImpl() {
  return true;
}

bool ReduceMPI::RunImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const auto &input = GetInput();
  int root_process = input.root;
  const auto &input_vec = input.data;

  /*
  int send_data = input_vec.empty() ? 0 : input_vec[0];  // у каждого процесса - свое число
  int recv_data = 0;                                     // буфер для результат
  */

  int local_sum =
      std::accumulate(input_vec.begin(), input_vec.end(), 0);  // вычисление локальной суммы всего входного вектора
  int global_sum = 0;

  // Вызов своей реализации Reduce()
  Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, root_process, MPI_COMM_WORLD);

  // Только корневой процесс записывает результат в Output
  if (rank == root_process) {
    GetOutput() = global_sum;
  }

  return true;
}

bool ReduceMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kutergin_v_reduce
