# ЛР-3. Решение СЛАУ итерационными методами (Jacobi, CUDA)

## Постановка
Решаем `A x = b` методом Якоби для матрицы:
- `A[i,i] = 2N + 1`, `A[i,j] = 1` при `i ≠ j`.
- Генерируем `x_true`, считаем `b = A x_true`, затем восстанавливаем `x`.

## Реализация CUDA
Файл: `lab3/cuda/jacobi_cuda.cu`.

Используется быстрый вариант за счёт структуры матрицы:
- `sum_x = Σ x_j`
- `x^{k+1}_i = (b_i - (sum_x - x_i)) / (2N + 1)`
- `r_i = (2N)*x_i + sum_x - b_i`

Останов: `||A x − b||2 / ||b||2 < eps` или `k >= maxit`.

## Два варианта нормы невязки (по заданию)
1) **atomic**: `Σ r_i^2` суммируется на GPU через `atomicAdd` в глобальной памяти.
2) **host**: на каждой итерации `x_new` копируется на хост, норма считается на CPU, затем решаем продолжать/остановиться.

## Сборка
```bash
make -C lab3/cuda
```

## Прогон и результаты
```bash
bash lab3/scripts/run_cuda.sh 1000000 1e-8 10000 256
# output: lab3/results/timings.csv
```

Пример таблицы (заполни своими значениями из CSV):

| res_mode | N | block | time_sec | iters | residual | rel_err |
|---|---:|---:|---:|---:|---:|---:|
| atomic | ... | ... | ... | ... | ... | ... |
| host   | ... | ... | ... | ... | ... | ... |

## Профилирование (Nsight Systems / Nsight Compute)

### Nsight Systems (timeline)
```bash
nsys profile --output=jacobi_report ./lab3/cuda/jacobi_cuda -n 1000000 --res atomic
```

### Nsight Compute (метрики ядер)
```bash
ncu --section SpeedOfLight --section MemoryWorkloadAnalysis --section LaunchStats ./lab3/cuda/jacobi_cuda -n 1000000 --res atomic
# или полный отчёт:
ncu --set full -o report -f ./lab3/cuda/jacobi_cuda -n 1000000 --res atomic
```

## Примечания
- `--block` поддерживает 128/256/512.
- `--res atomic|host` выбирает способ подсчёта нормы невязки.
- `--device-info` печатает характеристики GPU.
