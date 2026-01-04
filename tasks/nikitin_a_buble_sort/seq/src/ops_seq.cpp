#include "nikitin_a_buble_sort/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "nikitin_a_buble_sort/common/include/common.hpp"

namespace nikitin_a_buble_sort {

NikitinABubleSortSEQ::NikitinABubleSortSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;   // Копируем входной вектор
  GetOutput() = {};  // Инициализируем выходной вектор как пустой
}

bool NikitinABubleSortSEQ::ValidationImpl() {
  return true;
}

bool NikitinABubleSortSEQ::PreProcessingImpl() {
  return true;
}

bool NikitinABubleSortSEQ::RunImpl() {
  const std::vector<double> &input_arr = GetInput();
  std::vector<double> arr = input_arr;  // Копируем входные данные во временный вектор
  size_t n = arr.size();

  // Обработка пустого массива
  if (input_arr.empty()) {
    GetOutput() = {};
    return true;
  }

  // Стандартная пузырьковая сортировка
  for (size_t i = 0; i < n - 1; ++i) {
    for (size_t j = 0; j < n - i - 1; ++j) {
      if (arr[j] > arr[j + 1]) {
        // Обмен элементов
        std::swap(arr[j], arr[j + 1]);
      }
    }
  }

  // Записываем отсортированный результат в GetOutput
  GetOutput() = arr;

  return true;
}

bool NikitinABubleSortSEQ::PostProcessingImpl() {
  // Проверяем, что массив отсортирован по возрастанию
  const std::vector<double> &arr = GetOutput();
  for (size_t i = 1; i < arr.size(); ++i) {
    if (arr[i - 1] > arr[i]) {
      return false;  // Массив не отсортирован
    }
  }
  return true;
}

}  // namespace nikitin_a_buble_sort
