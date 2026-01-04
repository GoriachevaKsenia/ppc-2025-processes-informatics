#include <gtest/gtest.h>
#include <mpi.h>

#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "pikhotskiy_r_scatter/common/include/common.hpp"
#include "pikhotskiy_r_scatter/mpi/include/ops_mpi.hpp"
#include "pikhotskiy_r_scatter/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace pikhotskiy_r_scatter {

class PikhotskiyRScatterPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;
  std::vector<int> send_vec_;
  std::vector<int> recv_vec_;

  const int count_per_proc_ = 3000000;

  bool static IsSeqTest() {
    const auto *test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    if (test_info == nullptr) {
      return false;
    }
    std::string test_name = test_info->name();
    return (test_name.find("seq") != std::string::npos) || (test_name.find("SEQ") != std::string::npos);
  }

 protected:
  void SetUp() override {
    bool is_seq = IsSeqTest();

    int rank = 0;
    int size = 1;

    if (!is_seq) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &size);
    }

    int root = 0;
    recv_vec_.resize(count_per_proc_);
    bool i_am_root = is_seq || (rank == root);

    if (i_am_root) {
      size_t total_send_count = is_seq ? count_per_proc_ : static_cast<size_t>(count_per_proc_) * size;

      send_vec_.resize(total_send_count);
      for (size_t i = 0; i < total_send_count; i++) {
        send_vec_[i] = static_cast<int>(i);
      }
    }

    const void *sendbuf_ptr = i_am_root ? send_vec_.data() : nullptr;

    input_data_ = std::make_tuple(sendbuf_ptr,       // sendbuf
                                  count_per_proc_,   // sendcount
                                  MPI_INT,           // sendtype
                                  recv_vec_.data(),  // recvbuf
                                  count_per_proc_,   // recvcount
                                  MPI_INT,           // recvtype
                                  root,              // root
                                  MPI_COMM_WORLD     // comm
    );
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data == nullptr && count_per_proc_ > 0) {
      return false;
    }

    bool is_seq = IsSeqTest();
    int rank = 0;

    if (!is_seq) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

    const int *actual_data = reinterpret_cast<const int *>(output_data);

    for (int i = 0; i < count_per_proc_; i++) {
      int expected_value = 0;
      if (is_seq) {
        expected_value = i;
      } else {
        expected_value = (rank * count_per_proc_) + i;
      }

      if (actual_data[i] != expected_value) {
        return false;
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(PikhotskiyRScatterPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, PikhotskiyRScatterMPI, PikhotskiyRScatterSEQ>(
    PPC_SETTINGS_pikhotskiy_r_scatter);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = PikhotskiyRScatterPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, PikhotskiyRScatterPerfTests, kGtestValues, kPerfTestName);

}  // namespace pikhotskiy_r_scatter
