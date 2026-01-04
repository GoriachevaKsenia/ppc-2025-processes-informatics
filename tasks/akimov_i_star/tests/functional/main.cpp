#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cstddef>
#include <fstream>
#include <ios>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>

#include "akimov_i_star/common/include/common.hpp"
#include "akimov_i_star/mpi/include/ops_mpi.hpp"
#include "akimov_i_star/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace akimov_i_star {

class AkimovIStarFromFileFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_akimov_i_star, "message.txt");

    std::ifstream file(abs_path, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
      throw std::runtime_error("Cannot open message.txt at path: " + abs_path);
    }

    std::string content;
    {
      std::ostringstream ss;
      ss << file.rdbuf();
      content = ss.str();
    }
    file.close();

    input_data_ = InType(content.begin(), content.end());

    expected_result_ = 0;
    {
      std::istringstream ss(content);
      std::string line;
      while (std::getline(ss, line)) {
        const std::string prefix = "send:";
        if (!line.starts_with(prefix)) {
          continue;
        }
        std::string rest = line.substr(prefix.size());
        size_t p1 = rest.find(':');
        if (p1 == std::string::npos) {
          continue;
        }
        size_t p2 = rest.find(':', p1 + 1);
        if (p2 == std::string::npos) {
          continue;
        }
        std::string dsts = rest.substr(p1 + 1, p2 - (p1 + 1));
        try {
          int dst = std::stoi(dsts);
          if (dst == 0) {
            ++expected_result_;
          }
        } catch (...) {
          continue;
        }
      }
    }

    is_valid_ = !content.empty();

    (void)std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
  }

  InType GetTestInputData() final {
    return input_data_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank = 0;
    int mpi_initialized = 0;

    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized != 0) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    } else {
      rank = 0;
    }

    if (!is_valid_ || rank != 0) {
      return true;
    }
    return output_data == expected_result_;
  }

 private:
  InType input_data_;
  OutType expected_result_ = 0;
  bool is_valid_ = true;
};

namespace {

TEST_P(AkimovIStarFromFileFuncTests, StarTopologyFromFile) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 1> kTestParam = {std::make_tuple(0, std::string("default"))};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<AkimovIStarMPI, InType>(kTestParam, PPC_SETTINGS_akimov_i_star),
                   ppc::util::AddFuncTask<AkimovIStarSEQ, InType>(kTestParam, PPC_SETTINGS_akimov_i_star));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = AkimovIStarFromFileFuncTests::PrintFuncTestName<AkimovIStarFromFileFuncTests>;

INSTANTIATE_TEST_SUITE_P(StarFuncTests, AkimovIStarFromFileFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace akimov_i_star
