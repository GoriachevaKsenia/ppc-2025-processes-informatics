#pragma once

#include <mpi.h>

#include "../../common/include/common.hpp"
#include "task/include/task.hpp"

namespace kutergin_v_reduce {

int Reduce(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm);

class ReduceMPI : public BaseTask  // класс-обертка для тестирования
{
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  };

  explicit ReduceMPI(const InType &in);

 protected:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace kutergin_v_reduce
