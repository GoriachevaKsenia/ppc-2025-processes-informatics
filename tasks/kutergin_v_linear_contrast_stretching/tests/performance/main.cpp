#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include "../../common/include/common.hpp"
#include "../../mpi/include/linear_contrast_stretching_mpi.hpp"
#include "../../seq/include/linear_contrast_stretching_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace kutergin_v_linear_contrast_stretching {

class LinearContrastStretchingPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  // подготовка входных данных перед каждым тестом
  void SetUp() override {
    int rank = 0;
    if (ppc::util::IsUnderMpirun()) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

    const int width = 4000;
    const int height = 4000;
    const size_t image_size = static_cast<size_t>(width) * height;

    std::mt19937 gen(rank);
    std::uniform_int_distribution<> dis(100, 150);

    input_data_.width = width;
    input_data_.height = height;
    input_data_.data.resize(image_size);

    for (size_t i = 0; i < image_size; ++i) {
      input_data_.data[i] = static_cast<unsigned char>(dis(gen));
    }
  }

  // проверка результата после выполнения задачи
  bool CheckTestOutputData([[maybe_unused]] OutType &output_data) final {
    return true;
  }

  // возврат подготовленных входных данных для задачи
  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{};

  /*
  bool CheckOutput(const OutType &output_data) {
    const auto &input_image = GetTestInputData().data;
    if (input_image.empty()) {
      return false;
    }

    OutType expected_output(input_image.size());

    const auto minmax_it = std::minmax_element(input_image.begin(), input_image.end());
    unsigned char min_in = *minmax_it.first;
    unsigned char max_in = *minmax_it.second;

    if (min_in == max_in) {
      expected_output = input_image;
    } else {
      const double scale = 255.0 / (max_in - min_in);
      for (size_t i = 0; i < input_image.size(); ++i) {
        double p_out = (input_image[i] - min_in) * scale;
        expected_output[i] = static_cast<unsigned char>(std::round(p_out));
      }
    }
    return output_data == expected_output;
  }
  */
};

namespace {

TEST_P(LinearContrastStretchingPerfTests, PerfTest) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, LinearContrastStretchingMPI, LinearContrastStretchingSequential>(
        PPC_SETTINGS_kutergin_v_linear_contrast_stretching);

// NOLINTNEXTLINE(modernize-type-traits, cppcoreguidelines-avoid-non-const-global-variables)
INSTANTIATE_TEST_SUITE_P(LinearContrastStretching, LinearContrastStretchingPerfTests,
                         ppc::util::TupleToGTestValues(kAllPerfTasks),
                         LinearContrastStretchingPerfTests::CustomPerfTestName);

}  // namespace

}  // namespace kutergin_v_linear_contrast_stretching
