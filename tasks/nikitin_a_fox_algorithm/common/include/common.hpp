#pragma once

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "task/include/task.hpp"

namespace nikitin_a_fox_algorithm {

// Входные данные: пара матриц
using InType = std::pair<std::vector<std::vector<double>>,  // matrix_a
                         std::vector<std::vector<double>>   // matrix_b
                         >;

// Выходные данные: результирующая матрица
using OutType = std::vector<std::vector<double>>;

// Тип для тестов: размер матрицы и описание теста
using TestType = std::tuple<int, std::string>;

// Базовый класс задачи
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace nikitin_a_fox_algorithm
