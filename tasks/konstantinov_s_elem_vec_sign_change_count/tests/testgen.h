#pragma once
#include <cstring>  // memcpy
// #include <iostream>
// #include <vector>

#include "konstantinov_s_elem_vec_sign_change_count/common/include/common.hpp"

namespace konstantinov_s_elem_vec_sign_change_count {
// СТАБИЛЬНЫЙ генератор тестовых данных
// Нужно подать массив с известным колвом смен знаков и одинаковыми знаками первого и последнего элемента
inline int GenerateTestData(const EType *examplearr, size_t arrsz, int sign_change_count, InType &v) {
  size_t m = v.size();
  // std::cout<<"Generating "<<m<<"\n";
  if (arrsz == 0 || m == 0) {
    return 0;
  }

  if (m < arrsz)  // для особо маленьких размеров
  {
    int res = 0;
    for (size_t i = 0; i < m - 1; i++) {
      v[i] = examplearr[i];
      res += static_cast<int>((examplearr[i] > 0) != (examplearr[i + 1] > 0));
    }
    v[m - 1] = examplearr[m - 1];
    return res;
  }

  size_t full_blocks = m / arrsz;
  size_t tail = m % arrsz;

  EType *dst = v.data();

  // Копируем первую копию массива
  memcpy(dst, examplearr, arrsz * sizeof(EType));
  size_t filled = 1;  // число заполненных блоков

  // Удвоительное копирование блоками
  // пока удвоение не превысит количество нужных полных блоков
  while (filled * 2 <= full_blocks) {
    memcpy(dst + (filled * arrsz), dst, filled * arrsz * sizeof(EType));
    filled *= 2;
  }

  // Дозаполняем оставшиеся полные блоки
  while (filled < full_blocks) {
    memcpy(dst + (filled * arrsz), dst, arrsz * sizeof(EType));
    filled++;
  }

  // Хвост = последний элемент чтобы не считать смены знаков там
  if (tail > 0) {
    EType last = examplearr[arrsz - 1];
    for (size_t i = 0; i < tail; ++i) {
      dst[(full_blocks * arrsz) + i] = last;
    }
  }

  //   for(int i=0;i<m;i++)
  //        std::cout<<v[i]<<" ";
  //      std::cout<<"\n\n";

  return static_cast<int>(sign_change_count * full_blocks);
}

}  // namespace konstantinov_s_elem_vec_sign_change_count
