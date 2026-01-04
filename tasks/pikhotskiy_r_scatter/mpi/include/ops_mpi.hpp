#pragma once

#include <mpi.h>

#include "pikhotskiy_r_scatter/common/include/common.hpp"
#include "task/include/task.hpp"

namespace pikhotskiy_r_scatter {

class PikhotskiyRScatterMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit PikhotskiyRScatterMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static int CustomScatterInt(const void *sendbuf, int sendcount, void *recvbuf, int recvcount, int root,
                              MPI_Comm comm);

  static int CustomScatterFloat(const void *sendbuf, int sendcount, void *recvbuf, int recvcount, int root,
                                MPI_Comm comm);

  static int CustomScatterDouble(const void *sendbuf, int sendcount, void *recvbuf, int recvcount, int root,
                                 MPI_Comm comm);

  static int CustomScatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount,
                           MPI_Datatype recvtype, int root, MPI_Comm comm);
};

}  // namespace pikhotskiy_r_scatter
