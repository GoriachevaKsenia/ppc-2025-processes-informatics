#pragma once
#include <vector>

#include "chaschin_v_broadcast/common/include/common.hpp"
#include "task/include/task.hpp"

namespace chaschin_v_broadcast {

template <typename T>
class ChaschinVBroadcastSEQ : public BaseTask<T> {
 public:
  using InType = std::vector<T>;
  using OutType = std::vector<T>;

  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit ChaschinVBroadcastSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace chaschin_v_broadcast
