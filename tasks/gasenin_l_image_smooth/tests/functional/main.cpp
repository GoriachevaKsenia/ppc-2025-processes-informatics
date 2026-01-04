#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "gasenin_l_image_smooth/common/include/common.hpp"
#include "gasenin_l_image_smooth/mpi/include/ops_mpi.hpp"
#include "gasenin_l_image_smooth/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace gasenin_l_image_smooth {

class GaseninLRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    auto test_params = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int kernel_size = std::get<0>(test_params);
    std::string test_name = std::get<1>(test_params);

    if (test_name == "small_image") {
      input_data_.width = 10;
      input_data_.height = 2;
      input_data_.data.resize(static_cast<size_t>(input_data_.width) * static_cast<size_t>(input_data_.height));

      for (size_t i = 0; i < input_data_.data.size(); ++i) {
        input_data_.data[i] = static_cast<uint8_t>(i % 256);
      }
    } else if (test_name == "mpi_coverage_zero_rows") {
      input_data_.width = 5;
      input_data_.height = 1;
      input_data_.data.assign(5, 255);
    } else if (test_name == "kernel1") {
      input_data_.width = 4;
      input_data_.height = 4;
      input_data_.data.assign(16, 100);
    } else {
      int width = -1;
      int height = -1;
      int channels = -1;

      std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_gasenin_l_image_smooth, "pic.jpg");
      auto *data = stbi_load(abs_path.c_str(), &width, &height, &channels, 1);

      if (data == nullptr) {
        width = 64;
        height = 64;
        input_data_.width = width;
        input_data_.height = height;
        input_data_.data.resize(static_cast<size_t>(width) * static_cast<size_t>(height));
        for (int i = 0; i < width * height; ++i) {
          input_data_.data[i] = (i % 2) * 255;
        }
      } else {
        input_data_.width = width;
        input_data_.height = height;
        input_data_.data =
            std::vector<uint8_t>(data, data + (static_cast<ptrdiff_t>(width) * static_cast<ptrdiff_t>(height)));
        stbi_image_free(data);
      }
    }

    input_data_.kernel_size = kernel_size;

    ref_output_ = input_data_;
    ref_output_.data.assign(input_data_.data.size(), 0);

    GaseninLImageSmoothSEQ task(input_data_);

    task.Validation();
    task.PreProcessing();
    task.Run();
    task.PostProcessing();

    auto result_task = task.GetOutput();
    ref_output_.data = result_task.data;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.data.empty()) {
      return true;
    }

    return output_data == ref_output_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType ref_output_;
};

namespace {

TEST_P(GaseninLRunFuncTestsProcesses, ImageSmoothing) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 6> kTestParam = {std::make_tuple(3, "kernel3"),
                                            std::make_tuple(5, "kernel5"),
                                            std::make_tuple(7, "kernel7"),
                                            std::make_tuple(3, "small_image"),
                                            std::make_tuple(3, "mpi_coverage_zero_rows"),
                                            std::make_tuple(1, "kernel1")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<GaseninLImageSmoothMPI, InType>(kTestParam, PPC_SETTINGS_gasenin_l_image_smooth),
    ppc::util::AddFuncTask<GaseninLImageSmoothSEQ, InType>(kTestParam, PPC_SETTINGS_gasenin_l_image_smooth));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = GaseninLRunFuncTestsProcesses::PrintFuncTestName<GaseninLRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(ImageSmoothTests, GaseninLRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace gasenin_l_image_smooth
