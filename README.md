# Рандомные графы

Этот проект реализует подход к проверке статистических гипотез о равенстве распределений на основе **графов k ближайших соседей (KNN)** и **дистанционных графов**. Анализируются графовые характеристики:

- **$\delta(G)$** — максимальная степень вершины графа,
- **$\Delta(G)$** — максимальная степень вершины,
- **$\alpha(G)$** — размер независимого множества,
- **$\chi(G)$** — хроматическое число.

## Цель

Проверить, насколько графовые признаки могут эффективно различать выборки из различных распределений:  
- $\mathrm{Exp}(\lambda_0)$  
- $\mathrm{LogNormal}(0, 1)$  
- $\mathcal{N}(0, 1)$  
- $\mathrm{SkewNormal}(0, 1, \xi=5)$

## Основные файлы

- `part1_1.jl` — скрипт с экспериментами по 1 части задания (для первых двух распределений),
- `part1_2.jl` — скрипт с экспериментами по 1 части задания (для вторых двух распределений),
- `part2_1.jl` — скрипт с экспериментами по 2 части задания (для первых двух распределений),
- `part2_2.jl` — скрипт с экспериментами по 2 части задания (для вторых двух распределений),
- `report.pdf` — файл с отчётом по проделанным экспериментам. 

## Запуск

```julia
using Pluto
Pluto.run()
