#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "goriacheva_k_strassen_algorithm/common/include/common.hpp"
#include "goriacheva_k_strassen_algorithm/mpi/include/ops_mpi.hpp"
#include "goriacheva_k_strassen_algorithm/seq/include/ops_seq.hpp"
#include "task/include/task.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace goriacheva_k_strassen_algorithm {

using FuncParam = ppc::util::FuncTestParam<InType, OutType, TestType>;

class GoriachevaKStrassenAlgorithmFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const testing::TestParamInfo<FuncParam> &info) {
    return std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kNameTest)>(info.param);
  }

 protected:
  void SetUp() override {
    const auto &params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_ = std::get<0>(params);
    expected_ = std::get<1>(params);
  }

  InType GetTestInputData() final {
    return input_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_.size()) {
      return false;
    }

    for (std::size_t i = 0; i < expected_.size(); ++i) {
      if (output_data[i].size() != expected_[i].size()) {
        return false;
      }
      for (std::size_t j = 0; j < expected_[i].size(); ++j) {
        if (std::fabs(output_data[i][j] - expected_[i][j]) > 1e-6) {
          return false;
        }
      }
    }
    return true;
  }

 private:
  InType input_;
  OutType expected_;
};

namespace {

std::vector<FuncParam> LoadTestParams() {
  const std::string path = ppc::util::GetAbsoluteTaskPath(PPC_ID_goriacheva_k_strassen_algorithm, "tests.json");

  std::ifstream fin(path);
  if (!fin.is_open()) {
    throw std::runtime_error("Cannot open tests.json");
  }

  nlohmann::json j;
  fin >> j;

  std::vector<FuncParam> cases;
  cases.reserve(j.size() * 2);

  const std::string settings_path = PPC_SETTINGS_goriacheva_k_strassen_algorithm;

  const std::string mpi_suffix =
      ppc::task::GetStringTaskType(GoriachevaKStrassenAlgorithmMPI::GetStaticTypeOfTask(), settings_path);

  const std::string seq_suffix =
      ppc::task::GetStringTaskType(GoriachevaKStrassenAlgorithmSEQ::GetStaticTypeOfTask(), settings_path);

  for (const auto &item : j) {
    InType input;
    input.a = item.at("input").at("A").get<std::vector<std::vector<double>>>();
    input.b = item.at("input").at("B").get<std::vector<std::vector<double>>>();

    OutType expected = item.at("result").get<OutType>();

    TestType tc{input, expected, item.at("name").get<std::string>()};

    std::string mpi_name = std::get<2>(tc) + "_" + mpi_suffix;
    cases.emplace_back(ppc::task::TaskGetter<GoriachevaKStrassenAlgorithmMPI, InType>, mpi_name, tc);

    std::string seq_name = std::get<2>(tc) + "_" + seq_suffix;
    cases.emplace_back(ppc::task::TaskGetter<GoriachevaKStrassenAlgorithmSEQ, InType>, seq_name, tc);
  }

  return cases;
}

const std::vector<FuncParam> kFuncParams = LoadTestParams();

TEST_P(GoriachevaKStrassenAlgorithmFuncTests, MatrixMultiplication) {
  ExecuteTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(GoriachevaKStrassenAlgorithmFunctionalTests, GoriachevaKStrassenAlgorithmFuncTests,
                         testing::ValuesIn(kFuncParams), GoriachevaKStrassenAlgorithmFuncTests::PrintTestParam);

}  // namespace
}  // namespace goriacheva_k_strassen_algorithm
