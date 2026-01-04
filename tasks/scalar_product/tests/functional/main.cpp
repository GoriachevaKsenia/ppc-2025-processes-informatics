#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "scalar_product/common/include/common.hpp"
#include "scalar_product/mpi/include/ops_mpi.hpp"
#include "scalar_product/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace scalar_product {

class ScalarProductRunFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    const auto &[vectors, description] = test_param;
    return {description};
    ;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);  // Получаем пару векторов
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const auto &[vector_a, vector_b] = input_data_;

    int expected_result = 0;
    for (size_t i = 0; i < vector_a.size(); ++i) {
      expected_result += vector_a[i] * vector_b[i];
    }

    return (output_data == expected_result);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(ScalarProductRunFuncTests, DotProductTest) {
  ExecuteTest(GetParam());
}

// Тестовые параметры: ((вектор1, вектор2), описание)
const std::array<TestType, 17> kTestParam = {
    // 1. Модифицированный базовый тест
    std::make_tuple(std::make_pair(std::vector<int>{2, 3, 4}, 
                                   std::vector<int>{5, 6, 7}), 
                    "modified_basic_vectors_3"),

    // 2. Отрицательные с разными множителями
    std::make_tuple(std::make_pair(std::vector<int>{-2, -3, -4},
                                   std::vector<int>{-3, -4, -5}),
                    "all_negative_different"),

    // 3. Сложные чередующиеся знаки
    std::make_tuple(std::make_pair(std::vector<int>{2, -3, 5, -7},
                                   std::vector<int>{-3, 5, -7, 11}),
                    "complex_alternating_signs"),

    // 4. Очень большие числа
    std::make_tuple(std::make_pair(std::vector<int>{15000, 25000, 35000},
                                   std::vector<int>{25000, 35000, 15000}),
                    "very_large_numbers"),
    
    // 5. Постоянные значения с разными размерами
    std::make_tuple(std::make_pair(std::vector<int>(150, 3),
                                   std::vector<int>(150, 7)),
                    "constant_vectors_150"),
    
    // 6. Нули в разных позициях
    std::make_tuple(std::make_pair(std::vector<int>{0, 5, 0, 3}, 
                                   std::vector<int>{2, 0, 4, 0}), 
                    "zero_patterns_mixed"),

    std::make_tuple(std::make_pair(std::vector<int>{8, 9, 10}, 
                                   std::vector<int>{0, 0, 0}), 
                    "all_zero_second_vector"),
    // 7. Маленький вектор с отрицательным значением
    std::make_tuple(std::make_pair(std::vector<int>{-5},
                                   std::vector<int>{4}),
                    "single_negative_element"),

    // 8. Вектор длиной 1500
    std::make_tuple(std::make_pair(std::vector<int>(1500, 2),
                                   std::vector<int>(1500, -3)),
                    "size_1500_opposite_signs"),

    // 9. Геометрическая прогрессия
    std::make_tuple(std::make_pair(
                        []{
                            std::vector<int> v(40);
                            for (int i = 0; i < 40; i++){ v[i] = i * 2;
}
                            return v;
                        }(),
                        []{
                            std::vector<int> v(40);
                            for (int i = 0; i < 40; i++){ v[i] = 39 - i;
}
                            return v;
                        }()),
                    "geometric_sequence"),

    // 10. Квадратичная зависимость
    std::make_tuple(std::make_pair(
                        []{
                            std::vector<int> v(70);
                            for (int i = 0; i < 70; i++){ v[i] = ((i * i) % 17) - 8;
}
                            return v;
                        }(),
                        []{
                            std::vector<int> v(70);
                            for (int i = 0; i < 70; i++){ v[i] = ((i * 5) % 19) - 9;
}
                            return v;
                        }()),
                    "quadratic_pattern"),

    // 11. Специальный паттерн (шахматная доска)
    std::make_tuple(std::make_pair(std::vector<int>{1,0,1,0,1,0,1,0},
                                   std::vector<int>{0,2,0,2,0,2,0,2}),
                    "chessboard_pattern"),

    // 12. Симметричные векторы
    std::make_tuple(std::make_pair(
                        []{
                            std::vector<int> v(80);
                            for (int i = 0; i < 80; i++){ v[i] = i * 3;
}
                            return v;
                        }(),
                        []{
                            std::vector<int> v(80);
                            for (int i = 0; i < 80; i++){ v[i] = (79 - i) * 2;
}
                            return v;
                        }()),
                    "symmetric_vectors"),

    // 13. Средний тест (3000 элементов)
    std::make_tuple(std::make_pair(std::vector<int>(3000, 5),
                                   std::vector<int>(3000, -2)),
                    "size_3000_mixed_sign"),

    // 14. Экстремальные значения с нулем
    std::make_tuple(std::make_pair(std::vector<int>{999999, 0, -999999},
                                   std::vector<int>{-333333, 777777, 333333}),
                    "extreme_with_zero"),

    // 15. Векторы с простыми числами
    std::make_tuple(std::make_pair(std::vector<int>{2, 3, 5, 7, 11},
                                   std::vector<int>{13, 17, 19, 23, 29}),
                    "prime_numbers"),

    // 16. Минимальный непустой вектор
    std::make_tuple(std::make_pair(std::vector<int>{1}, 
                                   std::vector<int>{1}),
                    "minimum_nonempty")
};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<ScalarProductMPI, InType>(kTestParam, PPC_SETTINGS_scalar_product),
                   ppc::util::AddFuncTask<ScalarProductSEQ, InType>(kTestParam, PPC_SETTINGS_scalar_product));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = ScalarProductRunFuncTests::PrintFuncTestName<ScalarProductRunFuncTests>;

INSTANTIATE_TEST_SUITE_P(ScalarProductTests, ScalarProductRunFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace scalar_product
