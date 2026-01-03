#pragma once

#include <mpi.h>

#include "sakharov_a_transmission_from_one_to_all/common/include/common.hpp"
#include "task/include/task.hpp"

namespace sakharov_a_transmission_from_one_to_all {

int MyBcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm);

class SakharovATransmissionFromOneToAllMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit SakharovATransmissionFromOneToAllMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace sakharov_a_transmission_from_one_to_all
