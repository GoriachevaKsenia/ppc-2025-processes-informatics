#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <string>
#include <tuple>
#include <vector>

#include "titaev_m_metod_pryamougolnikov/common/include/common.hpp"
#include "titaev_m_metod_pryamougolnikov/mpi/include/ops_mpi.hpp"
#include "titaev_m_metod_pryamougolnikov/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace titaev_m_metod_pryamougolnikov {

class TitaevMMetodPryamougolnikovFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    test_input_.left_bounds = {0.0, 0.0};
    test_input_.right_bounds = {1.0, 1.0};
    test_input_.partitions = 10;
  }

  bool CheckTestOutputData(OutType &output_value) override {
    const double exact_value = 1.0;
    return std::abs(output_value - exact_value) < 1e-4;
  }

  InType GetTestInputData() override {
    return test_input_;
  }

 private:
  InType test_input_;
};

namespace {

TEST_P(TitaevMMetodPryamougolnikovFuncTests, Correctness) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(1, "small"), std::make_tuple(2, "medium"),
                                            std::make_tuple(3, "large")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<TitaevMMetodPryamougolnikovMPI, InType>(
                                               kTestParam, PPC_SETTINGS_titaev_m_metod_pryamougolnikov),
                                           ppc::util::AddFuncTask<TitaevMMetodPryamougolnikovSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_titaev_m_metod_pryamougolnikov));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    TitaevMMetodPryamougolnikovFuncTests::PrintFuncTestName<TitaevMMetodPryamougolnikovFuncTests>;

INSTANTIATE_TEST_SUITE_P(Correctness, TitaevMMetodPryamougolnikovFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace titaev_m_metod_pryamougolnikov
