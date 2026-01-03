#pragma once

#include <vector>

#include "eremin_v_strongin_algorithm/common/include/common.hpp"
#include "task/include/task.hpp"

namespace eremin_v_strongin_algorithm {

class EreminVStronginAlgorithmSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit EreminVStronginAlgorithmSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  struct IntervalCharacteristic {
    double value;
    int index;
  };

  static double CalculateLipschitzEstimate(const std::vector<double> &search_points,
                                           const std::vector<double> &function_values);
  static IntervalCharacteristic FindBestInterval(const std::vector<double> &search_points,
                                                 const std::vector<double> &function_values, double m_parameter);
};

}  // namespace eremin_v_strongin_algorithm
