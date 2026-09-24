# Classificação da Causa Básica de Óbito: Random Forest vs. XGBoost (SIM)

Projeto da disciplina Inteligência Artificial (7º CC, Prof. Ivan Carlos
Alcântara de Oliveira), Universidade Presbiteriana Mackenzie.

Compara Random Forest e XGBoost, contra um baseline de classe majoritária,
na classificação do capítulo CID-10 da causa básica de óbito a partir de
variáveis sociodemográficas (idade, sexo, raça/cor, escolaridade e UF, derivada
do município de residência). Os dados vêm do Sistema de Informação sobre Mortalidade (SIM)
do Ministério da Saúde, Brasil, 2000-2026 (ODS 3, Saúde e Bem-Estar).

## Resumo dos resultados (N1, parciais)

Divisão estratificada 80/20 sobre 32.531.918 registros e 19 capítulos CID-10:

| Modelo | F1-macro |
|---|---|
| Baseline (classe majoritária) | 0,0224 |
| Random Forest | 0,1329 |
| XGBoost | 0,1406 |

Os dois modelos superam o baseline, mas o desempenho absoluto é baixo: as
cinco variáveis sociodemográficas carregam pouca informação sobre a causa do
óbito, e os capítulos são muito desbalanceados. Recall por classe, matrizes de
confusão e a discussão completa estão no notebook 03 e no artigo.

## Estrutura

- `data/README.md`: descrição, fonte, licença e escopo do dataset (os CSVs
  brutos não são versionados, ver `.gitignore`).
- `notebooks/01_eda_exploratoria.ipynb`: análise exploratória do SIM
  2000-2026 (volume, ausências, distribuição dos capítulos CID-10).
- `notebooks/02_preparacao_dados.ipynb`: preparação dos dados (filtros,
  codificação, prevenção de vazamento de rótulo) e discussão ética/LGPD.
- `notebooks/03_resultados_parciais.ipynb`: comparação baseline vs. Random
  Forest vs. XGBoost (resultados parciais da N1).
- `src/sim_utils.py`: decodificação de campos do SIM, mapeamento CID-10 para
  capítulo e carga multi-ano com cache.
- `tests/`: testes unitários de `src/sim_utils.py` (`tests/test_sim_utils.py`).
- `docs/artigo/`: artigo científico no template SBC (`artigo.tex` e
  `artigo.pdf`).
- `docs/Rubrica_Analitica_Parte2_preenchida.docx`: rubrica analítica com os
  campos de identificação preenchidos.

## Integrantes

| Integrante | RA |
|---|---|
| Arthur Meneses Neves | 10425727 |
| Danilo Oliveira Santos | 10410905 |
| Matheus Gabriel Viana Araujo | 10420444 |
| João Victor Vidal Barbosa | 10410165 |
| Guilherme Araujo Castro | 10427775 |

## Como reproduzir

Requer Python 3.13 ou superior e o [uv](https://docs.astral.sh/uv/).

1. Instalar as dependências:

   ```bash
   uv sync
   ```

2. Obter os dados. Baixe os arquivos `Mortalidade_Geral_<ano>.csv` (2000 a
   2026) pelo navegador, a partir de
   https://dados.gov.br/dados/conjuntos-dados/sim-1979-2019, e salve-os em
   `data/`, mantendo o padrão de nome. Detalhes em `data/README.md`.

3. Rodar os testes:

   ```bash
   uv run pytest
   ```

4. Executar os notebooks em ordem (`01`, `02`, `03`), por exemplo com
   `uv run jupyter lab`. Na primeira execução, o cache em Parquet é gerado em
   `data/cache/` (ignorado pelo Git) e reaproveitado nas seguintes. A
   execução completa do notebook 03 leva cerca de 50 minutos em um servidor
   de 16 vCPUs.

### Compilar o artigo

```bash
cd docs/artigo && pdflatex artigo && bibtex artigo && pdflatex artigo && pdflatex artigo
```

## Dataset

Fonte, licença (Creative Commons Atribuição) e cobertura estão em
`data/README.md`.

## Status

N1 (Parte 2) entregue; N2 planejada.
