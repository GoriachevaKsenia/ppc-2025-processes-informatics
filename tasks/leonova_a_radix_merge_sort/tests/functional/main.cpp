#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "leonova_a_radix_merge_sort/common/include/common.hpp"
#include "leonova_a_radix_merge_sort/mpi/include/ops_mpi.hpp"
#include "leonova_a_radix_merge_sort/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace leonova_a_radix_merge_sort {

class LeonovaARadixMergeSortRunFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    const auto &input = std::get<0>(test_param);

    std::string name = "size_" + std::to_string(input.size());

    static int test_counter = 0;
    name += "_test" + std::to_string(++test_counter);

    if (input.size() == 1) {
      name += "_single";
    } else if (input.size() <= 32) {
      name += "_small";
    } else if (input.size() <= 100) {
      name += "_medium";
    } else {
      name += "_large";
    }

    bool has_negative = false;
    bool all_same = true;
    bool is_sorted = true;
    bool is_reverse = true;

    for (size_t i = 0; i < input.size(); ++i) {
      if (i > 0) {
        if (input[i] != input[i - 1]) {
          all_same = false;
        }
        if (input[i] < input[i - 1]) {
          is_sorted = false;
        }
        if (input[i] > input[i - 1]) {
          is_reverse = false;
        }
      }
      if (input[i] < 0.0) {
        has_negative = true;
      }
    }

    if (all_same && input.size() > 1) {
      name += "_allsame";
    }
    if (is_sorted && input.size() > 1) {
      name += "_sorted";
    }
    if (is_reverse && input.size() > 1) {
      name += "_reverse";
    }
    if (has_negative) {
      name += "_withneg";
    }
    if (input.size() == 32) {
      name += "_threshold";
    }
    if (input.size() == 33) {
      name += "_abovethreshold";
    }

    return name;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    input_data_ = std::get<0>(params);
    expected_output_ = std::get<1>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_output_.size()) {
      return false;
    }
    return output_data == expected_output_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

namespace {

TEST_P(LeonovaARadixMergeSortRunFuncTests, RadixMergeSort) {
  ExecuteTest(GetParam());
}

// 1. Малые массивы (меньше порога 32)
const std::array<TestType, 20> kTestParam = {
    // 1 элемент
    std::make_tuple(std::vector<double>{42.0}, 
                    std::vector<double>{42.0}),
    
    // 2 элемента
    std::make_tuple(std::vector<double>{3.14, 2.71}, 
                    std::vector<double>{2.71, 3.14}),
    
    // 5 элементов, случайные
    std::make_tuple(std::vector<double>{5.0, 1.0, 4.0, 2.0, 3.0}, 
                    std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0}),
    
    // С отрицательными
    std::make_tuple(std::vector<double>{-5.0, 0.0, 5.0, -10.0, 10.0}, 
                    std::vector<double>{-10.0, -5.0, 0.0, 5.0, 10.0}),
    
    // Дубликаты
    std::make_tuple(std::vector<double>{2.0, 2.0, 1.0, 1.0, 3.0, 3.0}, 
                    std::vector<double>{1.0, 1.0, 2.0, 2.0, 3.0, 3.0}),
    
    // Уже отсортированный
    std::make_tuple(std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0}, 
                    std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0}),
    
    // Обратно отсортированный
    std::make_tuple(std::vector<double>{5.0, 4.0, 3.0, 2.0, 1.0}, 
                    std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0}),
    
    // Ровно 32 элемента
    std::make_tuple(std::vector<double>{
        32.0, 31.0, 30.0, 29.0, 28.0, 27.0, 26.0, 25.0,
        24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0,
        16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0,
        8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0
    }, 
    std::vector<double>{
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
        25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0
    }),
    
    // 33 элемента
    std::make_tuple(std::vector<double>{
        33.0, 32.0, 31.0, 30.0, 29.0, 28.0, 27.0, 26.0, 25.0,
        24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0,
        15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0,
        6.0, 5.0, 4.0, 3.0, 2.0, 1.0
    }, 
    std::vector<double>{
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
        10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
        19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0,
        28.0, 29.0, 30.0, 31.0, 32.0, 33.0
    }),
    
    // 50 элементов
    std::make_tuple(std::vector<double>{
        50.0, 49.0, 48.0, 47.0, 46.0, 45.0, 44.0, 43.0, 42.0, 41.0,
        40.0, 39.0, 38.0, 37.0, 36.0, 35.0, 34.0, 33.0, 32.0, 31.0,
        30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0,
        20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0,
        10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0
    }, 
    std::vector<double>{
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
        11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
        21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
        31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
        41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0
    }),
    
    // 100 элементов
    std::make_tuple(
        []() {
            std::vector<double> v(100);
            for (int i = 0; i < 100; ++i){ v[i] = 100.0 - i;
}
            return v;
        }(),
        []() {
            std::vector<double> v(100);
            for (int i = 0; i < 100; ++i){ v[i] = i + 1.0;
}
            return v;
        }()
    ),
    
    // Смешанные положительные и отрицательные
    std::make_tuple(std::vector<double>{-10.5, 3.14, 0.0, -3.14, 10.5, -1.0, 1.0}, 
                    std::vector<double>{-10.5, -3.14, -1.0, 0.0, 1.0, 3.14, 10.5}),
    
    // Дробные числа
    std::make_tuple(std::vector<double>{0.1, 0.01, 0.001, 1.0, 10.0, 100.0}, 
                    std::vector<double>{0.001, 0.01, 0.1, 1.0, 10.0, 100.0}),
    
    // Большие и маленькие числа
    std::make_tuple(std::vector<double>{1e10, 1e-10, 1e5, 1e-5, 0.0}, 
                    std::vector<double>{0.0, 1e-10, 1e-5, 1e5, 1e10}),
    
    // Все одинаковые
    std::make_tuple(std::vector<double>(25, 7.7), 
                    std::vector<double>(25, 7.7)),
    
    // Чередование
    std::make_tuple(std::vector<double>{1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0}, 
                    std::vector<double>{-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0}),
    
    // Специальные значения: ноль и -ноль
    std::make_tuple(std::vector<double>{0.0, -0.0, 1.0, -1.0}, 
                    std::vector<double>{-1.0, 0.0, 0.0, 1.0}),
    
    // 64 элемента (степень двойки)
    std::make_tuple(
        []() {
            std::vector<double> v(64);
            for (int i = 0; i < 64; ++i){ v[i] = 64.0 - i;
}
            return v;
        }(),
        []() {
            std::vector<double> v(64);
            for (int i = 0; i < 64; ++i){ v[i] = i + 1.0;
}
            return v;
        }()
    ),
    
    // 127 элементов (не степень двойки)
    std::make_tuple(
        []() {
            std::vector<double> v(127);
            for (int i = 0; i < 127; ++i){ v[i] = 127.0 - i;
}
            return v;
        }(),
        []() {
            std::vector<double> v(127);
            for (int i = 0; i < 127; ++i){ v[i] = i + 1.0;
}
            return v;
        }()
    ),
    
    // 200 элементов
    std::make_tuple(
        []() {
            std::vector<double> v(200);
            for (int i = 0; i < 200; ++i){ v[i] = 200.0 - i;
}
            return v;
        }(),
        []() {
            std::vector<double> v(200);
            for (int i = 0; i < 200; ++i){ v[i] = i + 1.0;
}
            return v;
        }()
    )
};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<LeonovaARadixMergeSortMPI, InType>(kTestParam, PPC_SETTINGS_leonova_a_radix_merge_sort),
    ppc::util::AddFuncTask<LeonovaARadixMergeSortSEQ, InType>(kTestParam, PPC_SETTINGS_leonova_a_radix_merge_sort));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = LeonovaARadixMergeSortRunFuncTests::PrintFuncTestName<LeonovaARadixMergeSortRunFuncTests>;

INSTANTIATE_TEST_SUITE_P(RadixMergeSortTests, LeonovaARadixMergeSortRunFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace leonova_a_radix_merge_sort
