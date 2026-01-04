#pragma once

#include "task/include/task.hpp"
#include "zenin_a_topology_star/common/include/common.hpp"

namespace zenin_a_topology_star {

class ZeninATopologyStarSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit ZeninATopologyStarSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace zenin_a_topology_star
