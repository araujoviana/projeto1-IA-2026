# Classificação da causa básica de óbito: Random Forest vs. XGBoost

Projeto de Inteligência Artificial (7º CC, Prof. Ivan Carlos Alcântara de
Oliveira), Universidade Presbiteriana Mackenzie.

Classificamos o capítulo CID-10 da causa básica de óbito a partir de
atributos da Declaração de Óbito que não descrevem a causa, usando os dados
do SIM (Ministério da Saúde, Brasil, 2000-2026). Random Forest e XGBoost são
comparados entre si e com dois baselines (classe majoritária e tabela
idade x sexo).

- **N1 (Parte 2):** cinco atributos sociodemográficos, configurações
  iniciais. Notebook 03 e `docs/ArtigoParcial-Etapa2.tex`.
- **N2 (Parte 3):** ajuste de hiperparâmetros com orçamento igual, conjunto
  ampliado de atributos (mês e ano do óbito, estado civil, local de
  ocorrência, assistência médica) com ablação, intervalos de confiança por
  bootstrap, análise por subgrupo e importância por permutação. Notebook 04
  e `docs/ArtigoFinal-Etapa3.tex`.

## Integrantes

- Arthur Meneses Neves
- Danilo Oliveira Santos
- Matheus Gabriel Viana Araujo
- João Victor Vidal Barbosa
- Guilherme Araujo Castro

## Organização

- `data/README.md`: fonte, licença e escopo do dataset. Os CSVs não são versionados.
- `notebooks/01_eda_exploratoria.ipynb`: análise exploratória.
- `notebooks/02_preparacao_dados.ipynb`: preparação dos dados e ética/LGPD.
- `notebooks/03_resultados_parciais.ipynb`: baseline, Random Forest e XGBoost (N1).
- `notebooks/04_resultados_finais.ipynb`: resultados finais (N2).
- `src/sim_utils.py`: leitura, decodificação e preparação dos dados do SIM.
- `src/avaliacao.py`: avaliação da N2 (busca de hiperparâmetros, bootstrap,
  subgrupos com supressão de células pequenas, importância por permutação).
- `tests/`: testes automatizados dos dois módulos (`uv run pytest`).
- `resultados/`: tabelas e resumo (`n2_*.csv`, `n2_resumo.json`) gerados
  pelo notebook 04; todos os números do artigo final saem daqui.
- `docs/`: artigos parcial e final (fontes LaTeX, figuras e referências).

## Como rodar

Precisa de Python 3.13+ e do [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
uv run jupyter lab
```

Os CSVs `Mortalidade_Geral_<ano>.csv` (2000 a 2026, cerca de 8,5 GB) precisam
ser baixados pelo navegador em
https://dados.gov.br/dados/conjuntos-dados/sim-1979-2019 e colocados em
`data/` (detalhes em `data/README.md`). O notebook 03 usa bastante memória
(32 GB ou mais) e leva cerca de 50 min em 16 vCPUs.

### Notebook 04 (N2)

1. Com os CSVs em `data/`, rode `uv run pytest` (tudo deve passar).
2. Abra `notebooks/04_resultados_finais.ipynb` e ajuste a célula de
   configuração conforme a máquina. O principal é `FRACAO_TREINO`, a fração
   do treino usada nos modelos finais (padrão 0,3, cerca de 7,8 milhões de
   linhas, ~2 GB por matriz no conjunto ampliado). A seção 3 do notebook
   imprime a memória estimada antes de treinar. Com 16 GB de RAM, use 0,2 ou menos.
3. Execute tudo (*Run All*). A primeira execução reconstrói o cache Parquet,
   porque a preparação mudou na N2 (`PREP_VERSION = "2"`). Com os padrões,
   espere algumas horas; `RODAR_TEMPORAL = 0` pula a checagem temporal.
4. Faça commit do notebook executado, de `resultados/` e das figuras
   `docs/figuras/n2_*.png`.

As configurações também podem vir de variáveis de ambiente (`FRACAO_TREINO`,
`N_ITER_BUSCA`, `N_BOOT`, `SIM_DATA_DIR`...). O notebook foi testado de ponta
a ponta com um SIM sintético no mesmo formato dos arquivos reais.

A execução registrada no repositório (notebook 04 e `resultados/`) foi feita
com a configuração padrão em uma VM do Google Colab (8 vCPUs, 55 GB de RAM),
com o ambiente do `uv.lock`, e levou cerca de 5 horas.

## Artigo

Compile em `docs/` com `pdflatex`, `bibtex`, `pdflatex`, `pdflatex`. Sem o
arquivo `docs/ras.tex` sai a versão pública, sem RA. Com ele, que não é
versionado, sai a versão para o AVA, com RA e e-mail. Os números do artigo
final saem de `resultados/`.

Fontes: [artigo parcial (N1)](docs/ArtigoParcial-Etapa2.tex) e
[artigo final (N2)](docs/ArtigoFinal-Etapa3.tex).
