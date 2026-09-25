# Dataset: SIM (Sistema de Informação sobre Mortalidade)

- **Fonte:** Ministério da Saúde, via Portal de Dados Abertos:
  https://dados.gov.br/dados/conjuntos-dados/sim-1979-2019
- **Licença:** Creative Commons Atribuição (CC-BY).
- **Cobertura oficial do conjunto:** óbitos registrados no Brasil, 1979–2019.
- **Conteúdo:** dados socioeconômicos e de residência/ocorrência do óbito,
  óbitos fetais e não fetais, condições e causas de morte (codificadas em
  CID-9 até 1995 e CID-10 a partir de 1996), e causas externas.

## O que está disponível localmente (atualizado 2026-09-22)

27 arquivos `Mortalidade_Geral_<ano>.csv` para **2000–2026**
(2026 parcial: 506.532 registros, ano ainda em andamento), ~8,5GB no total,
não versionados no Git (ver `data/*.csv` e `data/cache/` no `.gitignore`).
Todos são óbitos **não fetais**, formato `;`-separado, `latin1`. O número de colunas cresce de 39
(2000) para 88 (2024) ao longo da série, conforme o SIM incorporou novos
campos. As 8 colunas usadas neste projeto (tabela abaixo) existem com o
mesmo nome em todos os anos conferidos (2000, 2005, 2010, 2015, 2020, 2024,
2026).

Formatos divergentes entre anos (tratados em `src/sim_utils.py`): os CSVs de
2022 e 2023 não têm linha de cabeçalho (2023 também não tem a coluna
`NECROPSIA`), e `DTOBITO` vem como `ddmmaaaa`, `dd-mm-aaaa` (2022-2023) ou
inteiro sem zero à esquerda (2024-2026).

**Faltam 1979–1999** (não baixados). Ver "Escopo temporal do projeto" abaixo.

## Escopo geográfico e temporal (a pedido do professor)

- **Região:** Brasil, âmbito nacional (nenhum recorte por estado/região).
- **Período:** 2000–2026 (2026 parcial).
- **Classes do alvo:** capítulos CID-10 observados em `CAUSABAS` (ver
  `sim_utils.causabas_to_chapter`): 19 capítulos observados (os capítulos XIX
  e XXI nunca aparecem como causa básica nos dados). Números completos no
  "Resumo quantitativo" abaixo.

## Resumo quantitativo (notebook `02_preparacao_dados.ipynb`)

- **Registros brutos (2000–2026):** 32.629.071.
- **Filtros de preparação:** descartados 15.842 registros com `SEXO` fora de
  {1, 2}, 81.308 com `IDADE` desconhecida e 3 sem capítulo CID-10
  derivável; restam **32.531.918** registros para modelagem.
- **Classes:** 19 capítulos CID-10 observados; desbalanceamento de
  **13.972x** entre a classe mais frequente (IX, circulatório) e a menos
  frequente (VII, olho).
- **Entrada geográfica do modelo:** `uf`, derivada de `CODMUNRES` (o município
  entra no modelo no nível de UF: 27 categorias, nenhuma "Ignorado").
- **Valores "Ignorado" (proporção pooled, todos os anos):** `RACACOR`
  5,389%; `ESC` 26,409%. A ausência varia por ano (em 2000, por exemplo,
  `RACACOR` ~16% e `ESC` ~20%), então esses percentuais pooled não devem ser
  extrapolados para um ano específico.

## Escopo temporal do projeto

O projeto usa 2000-2026. Os anos 1979-1999 ficam de fora por uma razão
técnica: até 1995 o SIM usava CID-9 (códigos numéricos) e a partir de 1996
passou a usar CID-10 (letra e número, como `I219`), e não existe mapeamento
direto de capítulo entre os dois esquemas. Se esses anos forem baixados
depois, `sim_utils.causabas_to_chapter` os exclui do alvo (devolve
`(None, None)` para códigos só numéricos). Definir uma estratégia para essa
faixa fica para a N2.

## Como obter os dados

O download automático não funcionou, então os arquivos foram baixados
manualmente pelo navegador. Em 2026-09-22 testamos as outras rotas:

- **API do dados.gov.br** (`/api/3/action/package_show`): responde
  `401 Unauthorized` e exige token pessoal do gov.br.
- **opendatasus.saude.gov.br** (portal antigo): descontinuado, redireciona
  para `dadosabertos.saude.gov.br`, que não tem mais as páginas de recurso.
- **Bucket S3 antigo** (`s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SIM/DOxxOPEN.csv`):
  retorna `403 AccessDenied`.
- **API de Dados Abertos do Ministério da Saúde** (`apidadosabertos.saude.gov.br`):
  não exige autenticação, mas só serve registros recentes (óbitos de 2025 no
  teste), sem filtro por ano.
- **FTP do DataSUS** (`ftp.datasus.gov.br`): resolve no DNS, mas não
  conseguimos baixar os arquivos. Pode funcionar de outra máquina.

Passo a passo (manual):

1. Abrir https://dados.gov.br/dados/conjuntos-dados/sim-1979-2019 no navegador.
2. Baixar o recurso "Mortalidade Geral" do ano desejado.
3. Salvar em `data/` como `Mortalidade_Geral_<ano>.csv`. O Git ignora esses
   arquivos.
4. Anotar aqui a URL usada e a data do download.

## Principais colunas usadas neste projeto

| Coluna | Significado | Observação |
|---|---|---|
| `TIPOBITO` | 1=fetal, 2=não fetal | Este arquivo só tem valor 2 |
| `DTOBITO` | Data do óbito (ddmmaaaa) | |
| `IDADE` | Idade codificada (1º dígito = unidade, 2 últimos = valor) | ver `sim_utils.decode_idade_anos` |
| `SEXO` | 0=ignorado, 1=masculino, 2=feminino | |
| `RACACOR` | Raça/cor autodeclarada | "Ignorado" pooled 5,389% (varia por ano; ~16% em 2000) |
| `ESC` | Escolaridade | "Ignorado" pooled 26,409% (varia por ano; ~20% em 2000) |
| `CODMUNRES` | Código IBGE do município de residência | |
| `CAUSABAS` | Causa básica do óbito, CID-9 (até 1995) ou CID-10 (1996+) | ver `sim_utils.causabas_to_chapter` |
