#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace khruev_a_global_opt {

struct SearchData {
  int func_id;
  double ax, bx;
  double ay, by;
  double epsilon;
  int max_iter;
  double r;
};

struct SearchResult {
  double x;
  double y;
  double value;
  int iter_count;
};

using InType = SearchData;
using OutType = SearchResult;
using BaseTask = ppc::task::Task<InType, OutType>;
using TestType = std::tuple<std::string, int, double, double, double, double, double>;

// 1. Кривая Гильберта: переводит t [0,1] -> (x, y) [0,1]x[0,1]
// Вспомогательная функция для поворота/отражения квадранта
// n: размер текущего квадрата (степень двойки)
// x, y: координаты
// rx, ry: биты, определяющие квадрант
inline void Rotate(uint64_t n, uint64_t &x, uint64_t &y, uint64_t rx, uint64_t ry) {
  if (ry == 0) {
    if (rx == 1) {
      x = n - 1 - x;
      y = n - 1 - y;
    }
    // Swap x и y
    uint64_t t = x;
    x = y;
    y = t;
  }
}

// 1. Кривая Гильберта: переводит t [0,1] -> (x, y) [0,1]x[0,1]
// Используем 32-битный порядок кривой (сетка 2^32 x 2^32), что покрывает точность double
inline void D2xy(double t, double &x, double &y) {
  // Ограничиваем t диапазоном [0, 1]
  t = std::max(t, 0.0);
  t = std::min(t, 1.0);

  // Порядок кривой N = 32. Всего точек 2^64 (влезает в uint64_t).
  // Масштабируем t [0, 1] в целое число s [0, 2^64 - 1]
  // Используем (2^64 - 1) как множитель.
  const auto max_dist = static_cast<uint64_t>(-1);
  const double two_to_64 = ldexp(1.0, 64);
  double val = t * two_to_64;
  uint64_t s = 0;
  if (val >= two_to_64) {
    s = max_dist;
  } else {
    s = static_cast<uint64_t>(val);
  }

  uint64_t ix = 0;
  uint64_t iy = 0;

  // Итеративный алгоритм от младших битов к старшим (bottom-up)
  // n - это размер под-квадрата на текущем уровне
  for (uint64_t nn = 1; nn < (1ULL << 32); nn <<= 1) {
    uint64_t rx = 1 & (s / 2);
    uint64_t ry = 1 & (s ^ rx);

    Rotate(nn, ix, iy, rx, ry);

    ix += nn * rx;
    iy += nn * ry;

    s /= 4;
  }

  // Нормализуем обратно в [0, 1]
  // Делим на 2^32 (размер сетки)
  const double scale = 1.0 / 4294967296.0;  // 1.0 / 2^32
  x = static_cast<double>(ix) * scale;
  y = static_cast<double>(iy) * scale;
}

inline double TargetFunction(int id, double x, double y) {
  if (id == 1) {
    // квадратичная
    return ((x - 0.5) * (x - 0.5)) + ((y - 0.5) * (y - 0.5));
  }
  if (id == 2) {
    // Rastrigin-like
    constexpr double kA = 10.0;
    constexpr double kTwoPi = 6.2831853071795864769;
    return (2.0 * kA) + ((x * x) - (kA * std::cos(kTwoPi * x))) + ((y * y) - (kA * std::cos(kTwoPi * y)));
  }
  if (id == 3) {
    // BoothFunc
    const double t1 = x + (2.0 * y) - 7.0;
    const double t2 = (2.0 * x) + y - 5.0;
    return (t1 * t1) + (t2 * t2);
  }
  if (id == 4) {
    // MatyasFunc
    return (0.26 * ((x * x) + (y * y))) - (0.48 * x * y);
  }
  if (id == 5) {  // Himme
    const double t1 = ((x * x) + y - 11.0);
    const double t2 = (x + (y * y) - 7.0);
    return (t1 * t1) + (t2 * t2);
  }

  return 0.0;
}

// Структура для хранения испытаний
struct Trial {
  double x;  // координата на отрезке [0, 1]
  double z;  // значение функции
  int id;    // исходный индекс

  bool operator<(const Trial &other) const {
    return x < other.x;
  }
};

}  // namespace khruev_a_global_opt
