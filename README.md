# Classificação da causa básica de óbito: Random Forest vs. XGBoost

# AINDA FALTA COLOCAR OS ARTIGOS NESSE REPO!!

Projeto de Inteligência Artificial (7º CC, Prof. Ivan Carlos Alcântara de
Oliveira), Universidade Presbiteriana Mackenzie.

Classificamos o capítulo CID-10 da causa básica de óbito a partir de idade,
sexo, raça/cor, escolaridade e UF, usando os dados do SIM (Ministério da
Saúde, Brasil, 2000-2026). Random Forest e XGBoost são comparados com um
baseline de classe majoritária. Os resultados parciais (N1) estão no notebook 03.

## Integrantes

- Arthur Meneses Neves (10425727)
- Danilo Oliveira Santos (10410905)
- Matheus Gabriel Viana Araujo (10420444)
- João Victor Vidal Barbosa (10410165)
- Guilherme Araujo Castro (10427775)

## Organização

- `data/README.md`: fonte, licença e escopo do dataset. Os CSVs não são versionados.
- `notebooks/01_eda_exploratoria.ipynb`: análise exploratória.
- `notebooks/02_preparacao_dados.ipynb`: preparação dos dados e ética/LGPD.
- `notebooks/03_resultados_parciais.ipynb`: baseline, Random Forest e XGBoost.
- `src/sim_utils.py`: leitura e decodificação dos dados do SIM.
- `tests/`: testes do `sim_utils.py`.

## Como rodar

Precisa de Python 3.13+ e do [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
uv run jupyter lab
```

Os CSVs `Mortalidade_Geral_<ano>.csv` (2000 a 2026) precisam ser baixados
pelo navegador em https://dados.gov.br/dados/conjuntos-dados/sim-1979-2019 e
colocados em `data/`. O notebook 03 usa bastante memória (32 GB ou mais) e
leva cerca de 50 min em 16 vCPUs.
