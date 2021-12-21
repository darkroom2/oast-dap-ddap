## Wymagania:

`python` >= 3.6  
`python-pip`

## Instalacja:

```shell
pip install -r requirements.txt
```

## Uruchomienie

Program przyjmuje:

* `--network-path` - ścieżka do pliku XML ze strukturą sieci
* `--algorithm-type` - typ algorytmu: **EA**, **BFA**
* `--problem-type` - typ problemu do rozwiązania: **DAP**, **DDAP**
* `--stop-criterion` - kryterium stopu: **time**, **max_gen**, **max_mut**,
  **no_progress**

Na przykład:

```shell
python3 main.py
```

W wyniku działania programu powstaje plik `simulation.log` w którym zapisana
została trajektoria procesu optymalizacji oraz plik `result.txt` w którym
zapisany został wynik ostateczny.

