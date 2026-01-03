#include <gtest/gtest.h>
#include <mpi.h>

#include <cstddef>
#include <numeric>
#include <random>
#include <string>

#include "../../common/include/common.hpp"
#include "../../mpi/include/ops_mpi.hpp"
#include "../../seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace kutergin_a_allreduce {

class KuterginAllreducePerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    int rank = 0;
    int world_size = 1;
    if (ppc::util::IsUnderMpirun()) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    }

    size_t total_elements = 100'000'000;
    size_t elements_per_process = total_elements / world_size;

    input_data_.elements.resize(elements_per_process);
    input_data_.root = 0;

    std::mt19937 gen(rank);
    std::uniform_int_distribution<> dis(1, 10);
    for (int &v : input_data_.elements) {
      v = dis(gen);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const auto &input = GetTestInputData();
    const auto &test_name = std::get<1>(GetParam());

    if (test_name.find("_seq_") != std::string::npos) {
      int expected_local_sum = std::accumulate(input.elements.begin(), input.elements.end(), 0);
      return output_data == expected_local_sum;
    }

    if (test_name.find("_mpi_") != std::string::npos) {
      int world_size = 1;
      if (ppc::util::IsUnderMpirun()) {
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
      }

      int expected_global_sum = 0;
      for (int rank_i = 0; rank_i < world_size; ++rank_i) {
        std::mt19937 gen_r(rank_i);
        std::uniform_int_distribution<> dis_r(1, 10);
        for (size_t i = 0; i < input.elements.size(); ++i) {
          expected_global_sum += dis_r(gen_r);
        }
      }
      return output_data == expected_global_sum;
    }

    return false;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{};
};

namespace {

TEST_P(KuterginAllreducePerfTests, PerfTest) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, AllreduceMPI, AllreduceSequential>(PPC_SETTINGS_kutergin_a_allreduce);

INSTANTIATE_TEST_SUITE_P(KuterginAllreducePerf, KuterginAllreducePerfTests,
                         ppc::util::TupleToGTestValues(kAllPerfTasks), KuterginAllreducePerfTests::CustomPerfTestName);

}  // namespace
}  // namespace kutergin_a_allreduce
