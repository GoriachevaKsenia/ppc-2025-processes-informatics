#pragma once

#include <mpi.h>

#include "romanov_a_scatter/common/include/common.hpp"
#include "task/include/task.hpp"

namespace romanov_a_scatter {

class RomanovAScatterMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit RomanovAScatterMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

int MyMPIScatter(void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount,
                 MPI_Datatype recvtype, int root, MPI_Comm comm);

}  // namespace romanov_a_scatter
