# Параллельные и распределенные вычисления. Домашнее задание №1

Подсчет интеграла одним потоком и несколькими. Бенчмарки.

Сборка для исполнения без логов:
```
mpic++ main.cpp -std=c++2a 
```

Сборка с логами:
```
mpic++ main.cpp -std=c++2a -DVERBOSE
```

Сборка с выводом статистики для бенчмарков. Сначала идет время взятия интеграла одним потоком, потом несколькими. Время выводится в секундах.
```
mpic++ main.cpp -std=c++2a -DBENCHMARK
```

Сборка с логами и бенчмарком:
```
mpic++ main.cpp -std=c++2a -DBENCHMARK -DVERBOSE
```

Чтобы прогнать бенчмарк надо скомпилировать программу с флагом `-DBENCHMARK`, а затем запустить `benchmark.py`.

`benchmark.py` поддерживает локальный запуск, для этого надо добавить флаг `--local`. По умолчанию `benchmark.py` запускается в конфигурации для кластера. Скрипт будет работать достаточно долго, поскольку он дожидается завершения каждой таски.


