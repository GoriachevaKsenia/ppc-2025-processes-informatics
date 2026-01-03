# Линейная фильтрация изображений (блочное разбиение). Ядро Гаусса 3x3.
**Студент:** Зенин Антон Алексеевич.
**Группа:** 3823Б1ФИ1
**Технология:** SEQ | MPI
**Вариант:** 28 
---

## 1. Введение

В рамках данной лабораторной работы реализована задача линейной фильтрации изображений с использованием ядра Гаусса размером 3×3 с блочным разбиением данных. Целью работы является изучение и сравнение последовательной и параллельной реализаций алгоритма обработки изображений, а также анализ производительности при использовании различного числа процессов.
Особое внимание уделено параллельной реализации с использованием двумерного блочного разбиения изображения между процессами MPI, что позволяет эффективно распределять вычислительную нагрузку и корректно обрабатывать граничные пиксели.
  
---

## 2. Постановка задачи

Требуется реализовать линейную фильтрацию изображения с использованием ядра Гаусса 3×3. Изображение может быть представлено как в оттенках серого, так и в цветном формате (RGB). Фильтрация должна корректно обрабатывать граничные пиксели изображения.  

Необходимо разработать:  
- последовательную реализацию алгоритма;  
- параллельную реализацию с использованием MPI и двумерного блочного разбиения изображения;  
- набор функциональных тестов и тестов производительности.  

**Входные данные**  

Для представления изображения используется структура Image;  

- Высота изображения (height);  
- Ширина изображения (width);  
- Количество каналов (channels):    
  - 1 — оттенки серого;    
  - 3 — RGB;  
- Одномерный массив пикселей `pixels` размером `width × height × channels`.  

  

**Выходные данные**

- Отфильтрованное изображение с теми же размерами height и width и количеством каналов;  
- Значения пикселей находятся в диапазоне `[0; 255]`.  



---

## 3. Описание алгоритма SEQ (Sequential)

В последовательной версии алгоритма для каждого пикселя выполняется свёртка с ядром Гаусса 3×3: 

$$
\begin{bmatrix}
1 & 2 & 1 \\
2 & 4 & 2 \\
1 & 2 & 1 
\end{bmatrix}
$$

Сумма произведений значений пикселей и коэффициентов ядра делится на 16 с округлением по формуле:  

- res = (sum + 8) / 16  

Для обработки граничных пикселей используется метод клэмпинга координат — выход за границы изображения заменяется ближайшим допустимым пикселем.  

---


## 4. Описание алгоритма MPI  

Параллельная версия реализована с использованием MPI и двумерного блочного разбиения изображения.  

Процессы организуются в двумерную решётку с помощью функции `MPI_Dims_create`. Изображение разбивается на прямоугольные блоки, каждый из которых назначается отдельному процессу.  

Для корректной фильтрации граничных пикселей каждый блок расширяется на одну строку и один столбец с каждой стороны (halo-область). Расширенные блоки формируются на корневом процессе с использованием клэмпинга координат и затем передаются соответствующим процессам.  

Каждый процесс выполняет фильтрацию своего блока локально. После завершения вычислений результаты собираются на корневом процессе и формируется итоговое изображение, которое затем рассылается всем процессам с помощью `MPI_Bcast`.    

 
## 5. Особенности реализации  

- Используется двумерное блочное разбиение изображения;  
- Корректная обработка граничных пикселей реализована с помощью halo-областей;  
- Последовательная и параллельная версии используют одинаковую формулу вычислений, что обеспечивает идентичность результатов;  

## Структура кода 

common/include/common.hpp — InType, OutType, формат входных и выходных данных  
seq/src/ops_seq.cpp — SEQ-реализация     
mpi/src/ops_mpi.cpp — MPI-реализация   
tests/functional/main.cpp — functional tests   
tests/performance/main.cpp — performance tests  

## Классы

- `ZeninAGaussFilterMPI : BaseTask`   
- `ZeninAGaussFilterSEQ : BaseTask`   

## 6. Окружение 


### Hardware
- CPU: Intel(R) Core(TM) i5-10400F CPU @ 2.90GHz    
- Cores: 6    
- RAM: 32 GB    
- OS: Windows 10 Pro  

### Toolchain
- Compiler: `C++20`     
- MPI: OpenMPI    
- Build type: Release  

## Данные  

- Используются 44 функциональных теста, которые рассматривают различные изображения с разными размерами и числом каналов.   
 
- Тест на производительность использует изображение с размерами 1500 на 1500 и три канала.   


---

## Результаты и выводы  

### 7.1 Корректность  

Корректность реализации подтверждена функциональными тестами. Результаты параллельной версии полностью совпадают с результатами последовательной реализации для всех тестовых случаев, включая граничные условия и изображения малого размера.  


### 7.2 Производительность  

#### Результаты performance-тестов  

|Processes| Time (s) | Speedup | Efficiency |
|---------|----------|---------|------------|
|     1   | 0.0490   | 1.00    |     N/A    |
|     2   | 0.0374   | 1.31    |    65.5%   |
|     4   | 0.0330   | 1.49    |    37.3%   |
|     6   | 0.0295   | 1.66    |    27.7%   |
|     8   | 0.0329   | 1.49    |    18.6%   |


---  

## Анализ результатов  

С увеличением числа процессов наблюдается уменьшение времени выполнения алгоритма и рост ускорения до определённого предела. Наилучшее время выполнения достигается при использовании 6 процессов.

При дальнейшем увеличении числа процессов (до 8) время выполнения возрастает, что связано с увеличением накладных расходов на обмен данными и синхронизацию между процессами. 


---

## 8. Выводы  
В рамках данной работы был реализован алгоритм линейной фильтрации изображений с использованием ядра Гаусса 3x3 и блочным разбиением. MPI и SEQ версии показывают одинаковые результаты работы.  
Экспериментальные результаты показывают, что параллельная реализация обеспечивает ускорение по сравнению с выполнением на одном процессе. Наибольшая эффективность достигается при использовании 2–6 процессов.

При дальнейшем увеличении числа процессов влияние накладных расходов MPI становится доминирующим, что приводит к снижению эффективности и росту времени выполнения. Это подтверждает, что для рассматриваемого размера задачи существует оптимальное количество процессов.

---

## 9. Источники
- cppreference.com - https://en.cppreference.com  
- Документация по OpenMPI - https://www.open-mpi.org/doc  
- Лекции по параллельному программированию ННГУ им. Лобачевского  
- Практические занятия по параллельному программированию ННГУ им. Лобачевского  

---  

## 10. Приложение. Код MPI реализации  
```cpp

namespace {

constexpr int kHalo = 1;
constexpr int kTagExpanded = 200;
constexpr int kTagResult = 500;

struct BlockInfo {
  int my_h = 0, my_w = 0;
  int start_y = 0, start_x = 0;
};

std::size_t GlobalIdx(int gx, int gy, int chan, int width, int channels) {
  return ((static_cast<std::size_t>(gy) * width + gx) * channels) + static_cast<std::size_t>(chan);
}

int Clampi(int v, int lo, int hi) {
  return std::max(lo, std::min(hi, v));
}

std::uint8_t Clampu8(int v) {
  return static_cast<std::uint8_t>(Clampi(v, 0, 255));
}

std::uint8_t GetLocal(const std::vector<std::uint8_t> &buf, int local_w_with_halo, int ch, int x, int y, int c) {
  const int idx = ((y * local_w_with_halo + x) * ch) + c;
  return buf[idx];
}

BlockInfo CalcBlock(int pr, int pc, int h, int w, int grid_r, int grid_c) {
  const int base_h = h / grid_r;
  const int base_w = w / grid_c;
  const int extra_h = h % grid_r;
  const int extra_w = w % grid_c;

  BlockInfo b;
  b.my_h = base_h + (pr < extra_h ? 1 : 0);
  b.my_w = base_w + (pc < extra_w ? 1 : 0);

  b.start_y = (pr * base_h) + std::min(pr, extra_h);
  b.start_x = (pc * base_w) + std::min(pc, extra_w);
  return b;
}

void FillExpandedBlock(const zenin_a_gauss_filter::Image &img, const zenin_a_gauss_filter::BlockInfo &bb, int width,
                       int height, int channels, std::vector<std::uint8_t> *dst) {
  const int hh = bb.my_h;
  const int ww = bb.my_w;
  const int dst_w = ww + (2 * kHalo);
  const int dst_h = hh + (2 * kHalo);

  dst->assign(static_cast<std::size_t>(dst_h) * dst_w * channels, 0);

  for (int ly = -kHalo; ly < hh + kHalo; ++ly) {
    for (int lx = -kHalo; lx < ww + kHalo; ++lx) {
      int gy = bb.start_y + ly;
      int gx = bb.start_x + lx;

      gy = std::max(0, std::min(height - 1, gy));
      gx = std::max(0, std::min(width - 1, gx));

      const int dy = ly + kHalo;
      const int dx = lx + kHalo;

      for (int chan = 0; chan < channels; ++chan) {
        (*dst)[((dy * dst_w + dx) * channels) + chan] = img.pixels[GlobalIdx(gx, gy, chan, width, channels)];
      }
    }
  }
}

void BuildOrRecvExpandedBlock(int rank, int proc_num, int grid_cols, int width, int height, int channels,
                              const zenin_a_gauss_filter::BlockInfo &my_block,
                              const std::function<zenin_a_gauss_filter::BlockInfo(int, int)> &calc_block,
                              const zenin_a_gauss_filter::Image *root_img, std::vector<std::uint8_t> *local_in) {
  if (rank == 0) {
    FillExpandedBlock(*root_img, my_block, width, height, channels, local_in);

    for (int rnk = 1; rnk < proc_num; ++rnk) {
      const int rpr = rnk / grid_cols;
      const int rpc = rnk % grid_cols;
      const auto rb = calc_block(rpr, rpc);

      std::vector<std::uint8_t> pack;
      FillExpandedBlock(*root_img, rb, width, height, channels, &pack);
      MPI_Send(pack.data(), static_cast<int>(pack.size()), MPI_UNSIGNED_CHAR, rnk, kTagExpanded, MPI_COMM_WORLD);
    }
  } else {
    MPI_Recv(local_in->data(), static_cast<int>(local_in->size()), MPI_UNSIGNED_CHAR, 0, kTagExpanded, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  }
}

void ConvolveLocalBlock(const std::vector<std::uint8_t> &local_in, int lw, int my_w, int my_h, int channels,
                        std::vector<std::uint8_t> *local_out) {
  constexpr int kKernelSum = 16;

  for (int yd = 0; yd < my_h; ++yd) {
    const int ly = yd + kHalo;
    for (int xd = 0; xd < my_w; ++xd) {
      const int lx = xd + kHalo;
      for (int chan = 0; chan < channels; ++chan) {
        const int v00 = static_cast<int>(zenin_a_gauss_filter::GetLocal(local_in, lw, channels, lx - 1, ly - 1, chan));
        const int v01 = static_cast<int>(zenin_a_gauss_filter::GetLocal(local_in, lw, channels, lx, ly - 1, chan));
        const int v02 = static_cast<int>(zenin_a_gauss_filter::GetLocal(local_in, lw, channels, lx + 1, ly - 1, chan));

        const int v10 = static_cast<int>(zenin_a_gauss_filter::GetLocal(local_in, lw, channels, lx - 1, ly, chan));
        const int v11 = static_cast<int>(zenin_a_gauss_filter::GetLocal(local_in, lw, channels, lx, ly, chan));
        const int v12 = static_cast<int>(zenin_a_gauss_filter::GetLocal(local_in, lw, channels, lx + 1, ly, chan));

        const int v20 = static_cast<int>(zenin_a_gauss_filter::GetLocal(local_in, lw, channels, lx - 1, ly + 1, chan));
        const int v21 = static_cast<int>(zenin_a_gauss_filter::GetLocal(local_in, lw, channels, lx, ly + 1, chan));
        const int v22 = static_cast<int>(zenin_a_gauss_filter::GetLocal(local_in, lw, channels, lx + 1, ly + 1, chan));

        int sum = 0;
        sum += v00 * 1;
        sum += v01 * 2;
        sum += v02 * 1;
        sum += v10 * 2;
        sum += v11 * 4;
        sum += v12 * 2;
        sum += v20 * 1;
        sum += v21 * 2;
        sum += v22 * 1;

        const int res = (sum + (kKernelSum / 2)) / kKernelSum;
        (*local_out)[((yd * my_w + xd) * channels) + chan] = zenin_a_gauss_filter::Clampu8(res);
      }
    }
  }
}

void CopyBlockToImage(const BlockInfo &block, const std::vector<std::uint8_t> &src, int src_w, int width, int channels,
                      std::vector<std::uint8_t> *dst) {
  for (int yd = 0; yd < block.my_h; ++yd) {
    for (int xd = 0; xd < block.my_w; ++xd) {
      const int gy = block.start_y + yd;
      const int gx = block.start_x + xd;
      for (int chan = 0; chan < channels; ++chan) {
        (*dst)[((gy * width + gx) * channels) + chan] = src[((yd * src_w + xd) * channels) + chan];
      }
    }
  }
}

void GatherAndBroadcastResult(int rank, int proc_num, int grid_cols, int width, int channels, const BlockInfo &my_block,
                              const std::function<BlockInfo(int, int)> &calc_block,
                              const std::vector<std::uint8_t> &local_out, std::vector<std::uint8_t> *final_image) {
  if (rank == 0) {
    CopyBlockToImage(my_block, local_out, my_block.my_w, width, channels, final_image);

    for (int src_rank = 1; src_rank < proc_num; ++src_rank) {
      const int spr = src_rank / grid_cols;
      const int spc = src_rank % grid_cols;
      const BlockInfo sb = calc_block(spr, spc);

      std::vector<std::uint8_t> recv(static_cast<std::size_t>(sb.my_h) * sb.my_w * channels);
      MPI_Recv(recv.data(), static_cast<int>(recv.size()), MPI_UNSIGNED_CHAR, src_rank, kTagResult, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

      CopyBlockToImage(sb, recv, sb.my_w, width, channels, final_image);
    }
  } else {
    MPI_Send(local_out.data(), static_cast<int>(local_out.size()), MPI_UNSIGNED_CHAR, 0, kTagResult, MPI_COMM_WORLD);
  }

  MPI_Bcast(final_image->data(), static_cast<int>(final_image->size()), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
}

}  // namespace

bool ZeninAGaussFilterMPI::RunImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const int pr = rank / grid_cols_;
  const int pc = rank % grid_cols_;

  const BlockInfo my_block = CalcBlock(pr, pc, height_, width_, grid_rows_, grid_cols_);
  const int my_h = my_block.my_h;
  const int my_w = my_block.my_w;

  const int lw = my_w + (2 * kHalo);
  const int lh = my_h + (2 * kHalo);

  std::vector<std::uint8_t> local_in(static_cast<std::size_t>(lh) * lw * channels_, 0);
  std::vector<std::uint8_t> local_out(static_cast<std::size_t>(my_h) * my_w * channels_, 0);

  auto calc_block = [&](int rpr, int rpc) -> BlockInfo {
    return CalcBlock(rpr, rpc, height_, width_, grid_rows_, grid_cols_);
  };

  if (rank == 0) {
    const auto &img = GetInput();
    BuildOrRecvExpandedBlock(rank, proc_num_, grid_cols_, width_, height_, channels_, my_block, calc_block, &img,
                             &local_in);
  } else {
    BuildOrRecvExpandedBlock(rank, proc_num_, grid_cols_, width_, height_, channels_, my_block, calc_block, nullptr,
                             &local_in);
  }

  ConvolveLocalBlock(local_in, lw, my_w, my_h, channels_, &local_out);

  std::vector<std::uint8_t> final_image(static_cast<std::size_t>(width_) * height_ * channels_, 0);
  GatherAndBroadcastResult(rank, proc_num_, grid_cols_, width_, channels_, my_block, calc_block, local_out,
                           &final_image);

  GetOutput() = OutType{height_, width_, channels_, std::move(final_image)};
  return true;
}

```

  