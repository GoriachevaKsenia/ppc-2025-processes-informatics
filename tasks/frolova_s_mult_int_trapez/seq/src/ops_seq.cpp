#include "frolova_s_mult_int_trapez/seq/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
// #include <iostream>
#include <vector>

#include "frolova_s_mult_int_trapez/common/include/common.hpp"

namespace frolova_s_mult_int_trapez {

FrolovaSMultIntTrapezSEQ::FrolovaSMultIntTrapezSEQ(const InType &in)
    : limits_(in.limits), number_of_intervals_(in.number_of_intervals) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

unsigned int FrolovaSMultIntTrapezSEQ::CalculationOfCoefficient(const std::vector<double> &point) {
  auto degree = static_cast<unsigned int>(limits_.size());
  for (unsigned int i = 0; i < limits_.size(); i++) {
    if ((limits_[i].first == point[i]) || (limits_[i].second == point[i])) {
      degree--;
    }
  }
  return static_cast<unsigned int>(std::pow(2.0, static_cast<double>(degree)));
}

void FrolovaSMultIntTrapezSEQ::Recursive(std::vector<double> &point, unsigned int &definition) {
  unsigned int temp_def = definition;

  std::vector<unsigned int> divisors(limits_.size(), 1);
  for (unsigned int i = 0; i < limits_.size(); i++) {
    if (i > 0) {
      divisors[i] = divisors[i - 1] * (number_of_intervals_[i - 1] + 1);
    }
  }

  for (int i = static_cast<int>(limits_.size()) - 1; i >= 0; i--) {
    const auto quotient = temp_def / divisors[i];
    point[i] = limits_[i].first + (static_cast<double>(quotient) * (limits_[i].second - limits_[i].first) /
                                   static_cast<double>(number_of_intervals_[i]));
    temp_def = temp_def % divisors[i];
  }

  definition = temp_def;
}

std::vector<double> FrolovaSMultIntTrapezSEQ::GetPointFromNumber(unsigned int number) {
  std::vector<double> point(limits_.size());
  unsigned int definition = number;

  if (!limits_.empty()) {
    Recursive(point, definition);
  }

  return point;
}

bool FrolovaSMultIntTrapezSEQ::ValidationImpl() {
  auto input = GetInput();

  if (input.limits.empty() || input.number_of_intervals.empty()) {
    return false;
  }

  if (input.limits.size() != input.number_of_intervals.size()) {
    return false;
  }

  if (!input.function) {
    return false;
  }

  for (size_t i = 0; i < input.limits.size(); i++) {
    if (input.limits[i].first >= input.limits[i].second) {
      return false;
    }

    if (input.number_of_intervals[i] == 0) {
      return false;
    }
  }

  return true;
}

bool FrolovaSMultIntTrapezSEQ::PreProcessingImpl() {
  auto input = GetInput();
  limits_ = input.limits;
  number_of_intervals_ = input.number_of_intervals;
  result_ = 0.0;

  return true;
}

bool FrolovaSMultIntTrapezSEQ::RunImpl() {
  unsigned int count = 1;
  for (auto number_of_interval : number_of_intervals_) {
    count *= (number_of_interval + 1);
  }

  for (unsigned int i = 0; i < count; i++) {
    std::vector<double> point = GetPointFromNumber(i);
    result_ += static_cast<double>(CalculationOfCoefficient(point)) * GetInput().function(point);
  }

  for (unsigned int i = 0; i < limits_.size(); i++) {
    result_ *= (limits_[i].second - limits_[i].first) / static_cast<double>(number_of_intervals_[i]);
  }

  result_ /= std::pow(2.0, static_cast<double>(limits_.size()));

  return true;
}

bool FrolovaSMultIntTrapezSEQ::PostProcessingImpl() {
  GetOutput() = result_;

  return true;
}

}  // namespace frolova_s_mult_int_trapez
