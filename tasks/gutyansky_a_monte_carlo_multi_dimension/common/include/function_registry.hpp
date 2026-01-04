#pragma once

#include <cmath>
#include <cstddef>
#include <functional>
#include <optional>
#include <span>

namespace gutyansky_a_monte_carlo_multi_dimension {

class FunctionRegistry final {
 public:
  using IntegralFunction = std::function<double(std::span<double>)>;

  struct FunctionDescription {
    size_t n_dims;
    IntegralFunction func;
  };

  static std::optional<FunctionDescription> GetIntegralFunction(size_t func_id) {
    auto f_const = [](std::span<double> /* not used */) { return 1.0; };
    auto f_linear = [](std::span<double> x) {
      double res = 0.0;

      for (auto val : x) {
        res += val;
      }

      return res;
    };
    auto f_product = [](std::span<double> x) {
      double res = 1.0;

      for (auto val : x) {
        res *= val;
      }

      return res;
    };
    auto f_quadric = [](std::span<double> x) {
      double res = 0.0;

      for (auto val : x) {
        res += val * val;
      }

      return res;
    };
    auto f_gauss = [](std::span<double> x) {
      double sum = 0.0;

      for (auto val : x) {
        sum += val * val;
      }

      return std::exp(-sum);
    };
    auto f_trig = [](std::span<double> x) { return std::sin(x[0]) * std::cos(x[1]); };
    auto f_sphere = [](std::span<double> x) {
      double r2 = 0.0;

      for (auto val : x) {
        r2 += val * val;
      }

      return (r2 <= 1.0) ? 1.0 : 0.0;
    };

    switch (func_id) {
      case 0:
        return FunctionDescription{.n_dims = 0, .func = f_const};
      case 1:
        return FunctionDescription{.n_dims = 0, .func = f_linear};
      case 2:
        return FunctionDescription{.n_dims = 0, .func = f_product};
      case 3:
        return FunctionDescription{.n_dims = 0, .func = f_quadric};
      case 4:
        return FunctionDescription{.n_dims = 0, .func = f_gauss};
      case 5:
        return FunctionDescription{.n_dims = 2, .func = f_trig};
      case 6:
        return FunctionDescription{.n_dims = 0, .func = f_sphere};
      default:
        return std::nullopt;
    }
  }
};

}  // namespace gutyansky_a_monte_carlo_multi_dimension
