#pragma once

#include <vector>

#include "task/include/task.hpp"
#include "titaev_m_metod_pryamougolnikov/common/include/common.hpp"

namespace titaev_m_metod_pryamougolnikov {

class TitaevMMetodPryamougolnikovSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit TitaevMMetodPryamougolnikovSEQ(const InType &input);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static double IntegrandFunction(const std::vector<double> &coords);
};

}  // namespace titaev_m_metod_pryamougolnikov
