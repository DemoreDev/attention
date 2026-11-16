# Resultados dos experimentos

## Tabela comparativa

| Experimento | Params | Melhor época | Train loss | Val loss | Test loss | Val EM | Test EM | OOD EM | Tempo de treino |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exp1 (d=32, L=2) | 26,560 | 50 / 50 | 1.66e-03 | 5.62e-06 | 6.11e-06 | 100.00% | 100.00% | 100.00% | 10.7 min |
| exp2 (d=64, L=3) | 152,000 | 50 / 50 | 3.13e-06 | 8.23e-08 | 8.21e-08 | 100.00% | 100.00% | 100.00% | 20.5 min |
| exp3 (d=128, L=4) | 796,672 | 35 / 40 | 2.56e-08 | 0.00e+00 | 0.00e+00 | 100.00% | 100.00% | 100.00% | 42.0 min |

Melhor época = época de menor val loss / épocas rodadas. EM = exact match. Tempo de treino = soma das épocas, incluindo a validação de cada época (loss + geração).

## Diagnóstico: Teste (1800–2049)

| Experimento | EM | Acurácia por caractere | Ano | Mês | Dia | Saídas válidas | Dia 1–9 com 1 dígito | Dia 1–9 com 2 dígitos | Mês maiúsculo | Mês minúsculo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exp1 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| exp2 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| exp3 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |

## Diagnóstico: Teste OOD (2050–2099)

| Experimento | EM | Acurácia por caractere | Ano | Mês | Dia | Saídas válidas | Dia 1–9 com 1 dígito | Dia 1–9 com 2 dígitos | Mês maiúsculo | Mês minúsculo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exp1 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| exp2 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| exp3 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |

## Exact match por mês (teste)

| Experimento | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | 11 | 12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exp1 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| exp2 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| exp3 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |

## Exemplos de erro

**exp1, teste** (0 erros no total)

- nenhum erro

**exp1, OOD** (0 erros no total)

- nenhum erro

**exp2, teste** (0 erros no total)

- nenhum erro

**exp2, OOD** (0 erros no total)

- nenhum erro

**exp3, teste** (0 erros no total)

- nenhum erro

**exp3, OOD** (0 erros no total)

- nenhum erro
