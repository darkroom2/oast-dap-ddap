### 1. Opis zadania

Celem zadania jest zaimplementowanie i przetestowanie algorytmu ewolucyjnego
rozwiązujących problemy DAP i DDAP. Wymagania:

- Program powinien wczytywać opis topologii sieci i zapotrzebowań z pliku
  tekstowego (format pliku wejściowego został opisany w Załączniku 1).
- Program powinien zapisywać wyniki obliczeń do pliku tekstowego (przykład
  formatu takiego pliku został przedstawiony w Załączniku 2) - opcjonalnie.
- Program powinien umożliwiać przeglądanie pełnej przestrzeni rozwiązań metodą
  brute force.
- Program powinien rozwiązywać problemy DAP i DDAP z wykorzystaniem algorytmu
  ewolucyjnego.
- Program powinien umożliwiać:
    - określenie liczności populacji startowej,
    - określenie prawdopodobieństwa wystąpienia krzyżowania i mutacji,
    - wybór kryterium stopu (wymagane są: zadany czas, zadana liczba generacji,
      zadana liczba mutacji, brak poprawy najlepszego znanego rozwiązania
      obserwowany w kolejnych N generacjach),
    - zapis trajektorii procesu optymalizacji rozumianej jako sekwencja wartości
      najlepszych rozwiązań (chromosomów) w kolejnych generacjach,
    - wskazanie ziarna dla generatora liczb losowych.

### 2. Zawartość sprawozdania

Powinno zawierać:

- opis zaimplementowanych algorytmów (ewolucyjny i bruteforce),
- krótki (co najwyżej jedna strona) opis implementacji,
- instrukcję uruchomienia programu,
- (dla każdej z sieci net4.txt, net12_1.txt, net12_2.txt, problemy DAP i DDAP)
  opis najlepszego uzyskanego rozwiązania
    - wartość funkcji kosztu,
    - liczbę wykonanych iteracji AE do znalezienia rozwiązania,
    - czas optymalizacji,
    - wartości parametrów algorytmu---liczność populacji, prawdopodobieństwo
      krzyżowania, prawdopodobieństwo mutacji,
    - wynikowe obciążenie łączy, wymiary łączy, rozkład zapotrzebowań na
      poszczególne ścieżki (przykładowy format zapisu można znaleźć w Załączniku
      2).

### Uwaga:

Do generacji chromosomów powinien być zastosowany generator liczb
pseudolosowych. Generator to funkcja deterministyczna. Do losowania kolejnych
liczb wykorzystuje tzw. ziarno (ang. seed), całkowicie determinujące wartości
kolejnych liczb pseudolosowych.
