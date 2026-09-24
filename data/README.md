# Dataset: SIM — Sistema de Informação sobre Mortalidade

- **Fonte:** Ministério da Saúde, via Portal de Dados Abertos —
  https://dados.gov.br/dados/conjuntos-dados/sim-1979-2019
- **Licença:** Creative Commons Atribuição (CC-BY).
- **Cobertura oficial do conjunto:** óbitos registrados no Brasil, 1979–2019.
- **Conteúdo:** dados socioeconômicos e de residência/ocorrência do óbito,
  óbitos fetais e não fetais, condições e causas de morte (codificadas em
  CID-9 até 1995 e CID-10 a partir de 1996), e causas externas.

## O que está disponível localmente (atualizado 2026-09-22)

27 arquivos `Mortalidade_Geral_<ano>.csv` para **2000–2026**
(2026 parcial: 506.532 registros, ano ainda em andamento), ~8,5GB no total,
não versionados no Git (ver `.gitignore:246-251`). Todos são óbitos **não
fetais**, formato `;`-separado, `latin1`. O número de colunas cresce de 39
(2000) para 88 (2024) ao longo da série, conforme o SIM incorporou novos
campos — as 8 colunas usadas neste projeto (tabela abaixo) existem com o
mesmo nome em todos os anos conferidos (2000, 2005, 2010, 2015, 2020, 2024,
2026).

Formatos divergentes entre anos (tratados em `src/sim_utils.py`): os CSVs de
2022 e 2023 não têm linha de cabeçalho (2023 também não tem a coluna
`NECROPSIA`), e `DTOBITO` vem como `ddmmaaaa`, `dd-mm-aaaa` (2022-2023) ou
inteiro sem zero à esquerda (2024-2026).

**Faltam 1979–1999** (não baixados). Ver "Escopo temporal" abaixo sobre por
que isso não é prioridade.

## Escopo geográfico e temporal (declarado explicitamente por pedido do professor)

- **Região:** Brasil, âmbito nacional (nenhum recorte por estado/região).
- **Período:** 2000–2026 (2026 parcial).
- **Classes do alvo:** capítulos CID-10 observados em `CAUSABAS` (ver
  `sim_utils.causabas_to_chapter`) — a contagem exata de capítulos presentes
  e seu desbalanceamento saem do notebook `01_eda_exploratoria.ipynb`.

## Escopo temporal do projeto

O projeto tem como alvo a série completa 1979–2019+, e a Parte 2 (N1) já
consolida 2000–2026. **1979–1999 ficam fora de escopo por uma razão técnica,
não só por conveniência**: até 1995 o SIM usava CID-9 (códigos numéricos) e
passou a usar CID-10 (letra+número, ex. `I219`) a partir de 1996 — não existe
mapeamento limpo de capítulo entre os dois esquemas. Mesmo que 1979–1995
fossem baixados depois, entrariam automaticamente excluídos do alvo de
classificação por `sim_utils.causabas_to_chapter` (retorna `(None, None)`
para códigos puramente numéricos) — mas validar uma estratégia para essa
faixa fica para a Parte 3 (N2). Abaixo está o que foi tentado para baixar os anos programaticamente e por que não funcionou.

## Como obter os anos que faltam (1979–1999, para quem continuar em N2)

Investigamos, em 2026-09-22, todas as rotas programáticas conhecidas para
baixar os anos restantes automaticamente, e nenhuma serve arquivos
históricos em lote de forma não-autenticada no momento:

- **API do dados.gov.br** (`/api/3/action/package_show`): responde
  `401 Unauthorized` com `WWW-Authenticate: Bearer` — exige login/token
  pessoal via gov.br, que não pode ser embutido no repositório.
- **opendatasus.saude.gov.br** (portal antigo do DataSUS, usado por várias
  referências desatualizadas na internet): foi descontinuado e agora
  redireciona para `dadosabertos.saude.gov.br`, que não expõe mais as
  páginas de dataset/recurso antigas.
- **Bucket S3 legado** (`s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SIM/DOxxOPEN.csv`,
  usado por tutoriais antigos): retorna `403 AccessDenied` — não é mais
  público.
- **API de Dados Abertos do Ministério da Saúde** (`apidadosabertos.saude.gov.br`,
  endpoint `/vigilancia-e-meio-ambiente/sistema-de-informacao-sobre-mortalidade`):
  é uma API real e sem autenticação, mas serve um feed de registros
  **recentes** (óbitos de 2025 no teste feito), sem parâmetro de filtro por
  ano e sem contagem total — inadequada para reconstituir lotes históricos
  como 1996 ou 2000.
- **FTP clássico do DataSUS** (`ftp.datasus.gov.br`, usado pelo pacote R
  `microdatasus`): resolve no DNS mas a conexão TCP não completa a partir
  deste ambiente de execução — pode funcionar normalmente de outra máquina
  (ex.: o notebook do próprio integrante, ou Google Colab), então vale
  tentar de lá antes de desistir dessa rota.

**Passo a passo recomendado (manual, via navegador):**

1. Acessar https://dados.gov.br/dados/conjuntos-dados/sim-1979-2019 no navegador (a interface web roda como SPA e busca os recursos via login de sessão do próprio navegador, diferente da chamada direta à API feita acima).
2. Baixar o(s) recurso(s) do(s) ano(s) desejado(s) (arquivo "Mortalidade Geral" por ano).
3. Salvar em `data/` com o mesmo padrão de nome (`Mortalidade_Geral_<ano>.csv`) — o arquivo é automaticamente ignorado pelo Git.
4. Registrar aqui a URL exata usada e a data do download, para reprodutibilidade.

## Principais colunas usadas neste projeto

| Coluna | Significado | Observação |
|---|---|---|
| `TIPOBITO` | 1=fetal, 2=não fetal | Este arquivo só tem valor 2 |
| `DTOBITO` | Data do óbito (ddmmaaaa) | |
| `IDADE` | Idade codificada (1º dígito = unidade, 2 últimos = valor) | ver `sim_utils.decode_idade_anos` |
| `SEXO` | 0=ignorado, 1=masculino, 2=feminino | |
| `RACACOR` | Raça/cor autodeclarada | ~16% ausente |
| `ESC` | Escolaridade | ~20% ausente |
| `CODMUNRES` | Código IBGE do município de residência | |
| `CAUSABAS` | Causa básica do óbito, CID-9 (até 1995) ou CID-10 (1996+) | ver `sim_utils.causabas_to_chapter` |
