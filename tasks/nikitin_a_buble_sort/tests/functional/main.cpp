#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "nikitin_a_buble_sort/common/include/common.hpp"
#include "nikitin_a_buble_sort/mpi/include/ops_mpi.hpp"
#include "nikitin_a_buble_sort/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace nikitin_a_buble_sort {

class NikitinABubleSortFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);  // возвращаем строковое описание теста
  }

 protected:
  void SetUp() override {
    auto params = GetParam();
    TestType test_params = std::get<2>(params);

    test_case_id_ = static_cast<int>(std::get<0>(test_params));
    test_description_ = std::get<1>(test_params);

    // Генерируем тестовые данные в зависимости от ID теста
    switch (test_case_id_) {
      case 0:
        GTEST_SKIP() << "Test case 0 is not implemented";
        break;
      case 1:  // Простой случай - уже отсортированный массив
        input_data_ = {1.0, 2.0, 3.0, 4.0, 5.0};
        expected_output_ = {1.0, 2.0, 3.0, 4.0, 5.0};
        break;

      case 2:  // Обратный порядок
        input_data_ = {5.0, 4.0, 3.0, 2.0, 1.0};
        expected_output_ = {1.0, 2.0, 3.0, 4.0, 5.0};
        break;

      case 3:  // Случайные числа
        input_data_ = {3.5, 1.2, 4.8, 2.1, 5.3, 0.5};
        expected_output_ = {0.5, 1.2, 2.1, 3.5, 4.8, 5.3};
        break;

      case 4:  // Дубликаты
        input_data_ = {2.0, 1.0, 2.0, 3.0, 2.0, 1.0};
        expected_output_ = {1.0, 1.0, 2.0, 2.0, 2.0, 3.0};
        break;

      case 5:  // Один элемент
        input_data_ = {42.0};
        expected_output_ = {42.0};
        break;

      case 6:  // Два элемента
        input_data_ = {2.0, 1.0};
        expected_output_ = {1.0, 2.0};
        break;

      case 7:  // Отрицательные числа
        input_data_ = {-5.0, -1.0, -3.0, -2.0, -4.0};
        expected_output_ = {-5.0, -4.0, -3.0, -2.0, -1.0};
        break;

      case 8:  // Смешанные положительные и отрицательные
        input_data_ = {-2.0, 3.0, -1.0, 0.0, 2.0, -3.0};
        expected_output_ = {-3.0, -2.0, -1.0, 0.0, 2.0, 3.0};
        break;

      case 9:  // Большой массив (100 элементов)
        input_data_ = GenerateRandomVector(100, 12345);
        expected_output_ = input_data_;
        std::ranges::sort(expected_output_.begin(), expected_output_.end());
        break;

      case 10:  // Очень большой массив (1000 элементов)
        input_data_ = GenerateRandomVector(1000, 54321);
        expected_output_ = input_data_;
        std::ranges::sort(expected_output_.begin(), expected_output_.end());
        break;

      case 11:  // Граничные значения double
        input_data_ = {std::numeric_limits<double>::min(), std::numeric_limits<double>::max(),
                       -std::numeric_limits<double>::max(), -std::numeric_limits<double>::min(), 0.0};
        expected_output_ = {-std::numeric_limits<double>::max(), -std::numeric_limits<double>::min(), 0.0,
                            std::numeric_limits<double>::min(), std::numeric_limits<double>::max()};
        break;

      case 12:  // Дробные числа с высокой точностью
        input_data_ = {1.111111, 1.111112, 1.111110, 1.111113, 1.111109};
        expected_output_ = {1.111109, 1.111110, 1.111111, 1.111112, 1.111113};
        break;

      case 13:  // Почти отсортированный массив
        input_data_ = {1.0, 2.0, 3.0, 5.0, 4.0, 6.0, 8.0, 7.0, 9.0};
        expected_output_ = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
        break;

      case 14:  // Все одинаковые элементы
        input_data_ = std::vector<double>(10, 10);
        expected_output_ = input_data_;
        break;

      case 15:  // Чередование больших и маленьких значений
        input_data_ = {1000.0, 0.001, 999.0, 0.002, 998.0, 0.003};
        expected_output_ = {0.001, 0.002, 0.003, 998.0, 999.0, 1000.0};
        break;

      case 16:  // Пустой массив (особый случай)
        input_data_ = {};
        expected_output_ = {};
        break;

      case 17:  // Массив с бесконечностями
        input_data_ = {std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(), 1.0, 0.0,
                       -1.0};
        expected_output_ = {-std::numeric_limits<double>::infinity(), -1.0, 0.0, 1.0,
                            std::numeric_limits<double>::infinity()};
        break;

      case 18:  // Алгоритмически сложный случай для пузырьковой сортировки
        input_data_ = {9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0};
        expected_output_ = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
        break;

      case 19:  // Большой случай для тестирования производительности и стабильности
        input_data_ = GenerateRandomVector(500, 99999);
        expected_output_ = input_data_;
        std::ranges::sort(expected_output_.begin(), expected_output_.end());
        break;

      case 20:  // Уже отсортированный массив
        input_data_ = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
        expected_output_ = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
        break;

      default:
        throw std::runtime_error("Unknown test case ID: " + std::to_string(test_case_id_));
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // Проверяем размер
    if (output_data.size() != expected_output_.size()) {
      return false;
    }

    // Проверяем каждый элемент с учетом погрешности для чисел с плавающей точкой
    for (size_t i = 0; i < output_data.size(); ++i) {
      // Для обычных чисел используем относительную погрешность
      if (std::abs(expected_output_[i]) > 1e-10) {
        if (std::abs(output_data[i] - expected_output_[i]) / std::abs(expected_output_[i]) > 1e-10) {
          return false;
        }
      } else {
        // Для очень маленьких чисел используем абсолютную погрешность
        if (std::abs(output_data[i] - expected_output_[i]) > 1e-10) {
          return false;
        }
      }
    }

    // Дополнительная проверка: массив должен быть отсортирован
    for (size_t i = 1; i < output_data.size(); ++i) {
      if (output_data[i - 1] > output_data[i]) {
        return false;
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  int test_case_id_ = 0;
  std::string test_description_;
  std::vector<double> input_data_;
  std::vector<double> expected_output_;

  // Генератор случайного вектора
  static std::vector<double> GenerateRandomVector(size_t size, int seed) {
    std::vector<double> result(size);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);

    for (size_t i = 0; i < size; ++i) {
      result[i] = dist(gen);
    }

    return result;
  }
};

namespace {

TEST_P(NikitinABubleSortFuncTests, BubbleSortTest) {
  ExecuteTest(GetParam());
}

// Определяем тестовые случаи
const std::array<TestType, 20> kTestParam = {
    std::make_tuple(1, "already_sorted"),     // Уже отсортированный
    std::make_tuple(2, "reverse_order"),      // Обратный порядок
    std::make_tuple(3, "random_numbers"),     // Случайные числа
    std::make_tuple(4, "duplicates"),         // Дубликаты
    std::make_tuple(5, "single_element"),     // Один элемент
    std::make_tuple(6, "two_elements"),       // Два элемента
    std::make_tuple(7, "negative_only"),      // Только отрицательные
    std::make_tuple(8, "mixed_signs"),        // Смешанные знаки
    std::make_tuple(9, "medium_array"),       // Средний массив
    std::make_tuple(10, "large_array"),       // Большой массив
    std::make_tuple(11, "boundary_values"),   // Граничные значения
    std::make_tuple(12, "high_precision"),    // Высокая точность
    std::make_tuple(13, "almost_sorted"),     // Почти отсортированный
    std::make_tuple(14, "all_equal"),         // Все одинаковые
    std::make_tuple(15, "wide_range"),        // Широкий диапазон
    std::make_tuple(16, "empty_array"),       // Пустой массив
    std::make_tuple(17, "infinity_values"),   // Бесконечности
    std::make_tuple(18, "worst_case"),        // Худший случай для пузырька
    std::make_tuple(19, "performance_test"),  // Тест производительности
    std::make_tuple(20, "sorted_array")       // Уже отсортированный массив

};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<NikitinABubleSortMPI, InType>(kTestParam, PPC_SETTINGS_nikitin_a_buble_sort),
                   ppc::util::AddFuncTask<NikitinABubleSortSEQ, InType>(kTestParam, PPC_SETTINGS_nikitin_a_buble_sort));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = NikitinABubleSortFuncTests::PrintFuncTestName<NikitinABubleSortFuncTests>;

INSTANTIATE_TEST_SUITE_P(BubbleSortTests, NikitinABubleSortFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace nikitin_a_buble_sort
