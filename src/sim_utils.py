# =============================================================================
# Integrantes: Arthur Meneses Neves, Danilo Oliveira Santos, Matheus Gabriel
# Viana Araujo, João Victor Vidal Barbosa, Guilherme Araujo Castro
#
# Síntese: Funções de decodificação de campos do SIM (Sistema de Informação
# sobre Mortalidade): idade codificada (IDADE) e mapeamento da causa básica
# de óbito (CAUSABAS, CID-10) para capítulo CID-10.
#
# Changelog:
#   2026-09-22 — Matheus Araujo — criação inicial do arquivo
#   2026-09-22 — Matheus Araujo — load_sim_anos: corrige carga de 2022 e
#     2023, cujos arquivos brutos não trazem linha de cabeçalho (descoberto
#     ao rodar a EDA do notebook 01 sobre 2000-2026 — ver Task 3)
#   2026-09-23 — Matheus Araujo — load_sim_anos: normaliza DTOBITO para
#     texto ddmmaaaa de 8 dígitos (formato bruto varia: "dd-mm-aaaa" em
#     2022/2023, inteiro sem zero à esquerda em 2024-2026)
#   2026-09-23 — Matheus Araujo — load_sim_anos: valida a 1ª linha dos anos
#     sem cabeçalho (nº de campos e não ser cabeçalho) e falha com ValueError
#   2026-09-23 — Matheus Araujo — adiciona preparar_df_modelo (frame de
#     modelagem: idade vetorizada, ESC/RACACOR harmonizados, UF derivada do
#     município, filtros com contabilidade) e carregar_df_modelo (cache
#     Parquet com chave = anos + tamanho/mtime dos CSVs + PREP_VERSION)
#   2026-09-23 — Matheus Araujo — adiciona FEATURE_COLUMNS e TARGET_COLUMN
#     (conjunto único de entradas/rótulo dos modelos, sem vazamento)
# =============================================================================
import csv
import hashlib
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Colunas presentes com o mesmo nome em todos os anos do SIM conferidos
# (2000, 2005, 2010, 2015, 2020, 2024, 2026), apesar do esquema geral ter
# crescido de 39 para 88 colunas no período — ver Scope note do plano antes
# de adicionar colunas aqui sem reconferir.
CORE_COLUMNS = [
    "TIPOBITO",
    "DTOBITO",
    "IDADE",
    "SEXO",
    "RACACOR",
    "ESC",
    "CODMUNRES",
    "CAUSABAS",
]

# (numeral do capítulo, início da faixa, fim da faixa, descrição em português)
# Faixas do índice alfabético da CID-10 (OMS). Comparação lexicográfica de
# strings "letra + 2 dígitos" funciona aqui porque nenhuma faixa cruza uma
# letra do meio (ex.: II vai de C00 a D48, mas C < D, então a ordenação
# lexicográfica coincide com a ordenação pretendida).
_CHAPTER_RANGES = [
    ("I", "A00", "B99", "Doenças infecciosas e parasitárias"),
    ("II", "C00", "D48", "Neoplasias (tumores)"),
    (
        "III",
        "D50",
        "D89",
        "Doenças do sangue, órgãos hematopoéticos e transtornos imunitários",
    ),
    ("IV", "E00", "E90", "Doenças endócrinas, nutricionais e metabólicas"),
    ("V", "F00", "F99", "Transtornos mentais e comportamentais"),
    ("VI", "G00", "G99", "Doenças do sistema nervoso"),
    ("VII", "H00", "H59", "Doenças do olho e anexos"),
    ("VIII", "H60", "H95", "Doenças do ouvido e da apófise mastoide"),
    ("IX", "I00", "I99", "Doenças do aparelho circulatório"),
    ("X", "J00", "J99", "Doenças do aparelho respiratório"),
    ("XI", "K00", "K93", "Doenças do aparelho digestivo"),
    ("XII", "L00", "L99", "Doenças da pele e do tecido subcutâneo"),
    ("XIII", "M00", "M99", "Doenças do sistema osteomuscular e do tecido conjuntivo"),
    ("XIV", "N00", "N99", "Doenças do aparelho geniturinário"),
    ("XV", "O00", "O99", "Gravidez, parto e puerpério"),
    ("XVI", "P00", "P96", "Afecções originadas no período perinatal"),
    (
        "XVII",
        "Q00",
        "Q99",
        "Malformações congênitas, deformidades e anomalias cromossômicas",
    ),
    (
        "XVIII",
        "R00",
        "R99",
        "Sintomas, sinais e achados anormais, causas mal definidas",
    ),
    ("XIX", "S00", "T98", "Lesões, envenenamento e outras causas externas"),
    ("XX", "V01", "Y98", "Causas externas de morbidade e mortalidade"),
    ("XXI", "Z00", "Z99", "Fatores que influenciam o estado de saúde"),
]


def decode_idade_anos(idade_codigo):
    """Decodifica o campo IDADE do SIM (código de 3 dígitos onde o 1º
    dígito é a unidade — 1=hora, 2=dia, 3=mês, 4=ano, 5=100+ anos — e os 2
    últimos são o valor naquela unidade) para idade em anos completos.

    Retorna None se o código for nulo ou for um valor de ignorado (000, 999).
    Códigos em horas/dias/meses (unidades 1-3) retornam 0 (menos de 1 ano).
    """
    if idade_codigo is None or (
        isinstance(idade_codigo, float) and math.isnan(idade_codigo)
    ):
        return None
    codigo = int(idade_codigo)
    if codigo in (0, 999):
        return None
    unidade, valor = divmod(codigo, 100)
    if unidade == 4:
        return valor
    if unidade == 5:
        return 100 + valor
    return 0


def causabas_to_chapter(codigo_cid10):
    """Recebe um código CAUSABAS do SIM (ex.: 'I219', '*T794', 'C349') e
    retorna (numeral_do_capitulo, descricao) do capítulo CID-10
    correspondente, ou (None, None) se o código não puder ser interpretado
    como CID-10 — em particular, códigos puramente numéricos (CID-9, usados
    até 1995) estão fora de escopo por design (ver Scope note do plano).
    """
    if not isinstance(codigo_cid10, str) or not codigo_cid10.strip():
        return (None, None)
    codigo = codigo_cid10.strip().lstrip("*").upper()
    if not codigo or not codigo[0].isalpha():
        return (None, None)
    letra = codigo[0]
    numeros = re.match(r"(\d+)", codigo[1:])
    if numeros is None:
        return (None, None)
    numero = int(numeros.group(1)[:2])
    chave = f"{letra}{numero:02d}"
    for numeral, inicio, fim, descricao in _CHAPTER_RANGES:
        if inicio <= chave <= fim:
            return (numeral, descricao)
    return (None, None)


# Esquema completo (87 colunas, na ordem em que aparecem no arquivo) do ano
# de 2021 — usado como referência para reconstruir o cabeçalho de anos cujo
# CSV bruto do SIM não traz uma linha de cabeçalho (ver _ESQUEMAS_SEM_CABECALHO
# abaixo). Column names conferidos batendo o valor de cada posição contra o
# domínio esperado do campo (datas, códigos de município, CID-10 etc.).
_ESQUEMA_2021 = [
    "ORIGEM",
    "TIPOBITO",
    "DTOBITO",
    "HORAOBITO",
    "NATURAL",
    "CODMUNNATU",
    "DTNASC",
    "IDADE",
    "SEXO",
    "RACACOR",
    "ESTCIV",
    "ESC",
    "ESC2010",
    "SERIESCFAL",
    "OCUP",
    "CODMUNRES",
    "LOCOCOR",
    "CODESTAB",
    "ESTABDESCR",
    "CODMUNOCOR",
    "IDADEMAE",
    "ESCMAE",
    "ESCMAE2010",
    "SERIESCMAE",
    "OCUPMAE",
    "QTDFILVIVO",
    "QTDFILMORT",
    "GRAVIDEZ",
    "SEMAGESTAC",
    "GESTACAO",
    "PARTO",
    "OBITOPARTO",
    "PESO",
    "TPMORTEOCO",
    "OBITOGRAV",
    "OBITOPUERP",
    "ASSISTMED",
    "EXAME",
    "CIRURGIA",
    "NECROPSIA",
    "LINHAA",
    "LINHAB",
    "LINHAC",
    "LINHAD",
    "LINHAII",
    "CAUSABAS",
    "CB_PRE",
    "COMUNSVOIM",
    "DTATESTADO",
    "CIRCOBITO",
    "ACIDTRAB",
    "FONTE",
    "NUMEROLOTE",
    "TPPOS",
    "DTINVESTIG",
    "CAUSABAS_O",
    "DTCADASTRO",
    "ATESTANTE",
    "STCODIFICA",
    "CODIFICADO",
    "VERSAOSIST",
    "VERSAOSCB",
    "FONTEINV",
    "DTRECEBIM",
    "ATESTADO",
    "DTRECORIGA",
    "CAUSAMAT",
    "ESCMAEAGR1",
    "ESCFALAGR1",
    "STDOEPIDEM",
    "STDONOVA",
    "DIFDATA",
    "NUDIASOBCO",
    "NUDIASOBIN",
    "DTCADINV",
    "TPOBITOCOR",
    "DTCONINV",
    "FONTES",
    "TPRESGINFO",
    "TPNIVELINV",
    "NUDIASINF",
    "DTCADINF",
    "MORTEPARTO",
    "DTCONCASO",
    "FONTESINF",
    "ALTCAUSA",
    "CONTADOR",
]

# Anos cujo CSV bruto do SIM (baixado de dados.gov.br) não traz uma linha de
# cabeçalho — a primeira linha do arquivo já é um registro de dados. Achado
# ao rodar load_sim_anos sobre 2000-2026 na EDA do notebook 01 (Task 3):
# 2000-2021, 2024-2026 têm cabeçalho normal; 2022 e 2023 não.
#
# 2022 tem as mesmas 87 colunas de 2021 (CONTADOR ao final), só falta a
# linha de cabeçalho. 2023 tem 86 colunas: CONTADOR aparece na 1ª posição
# (não ao final) e a coluna NECROPSIA não existe no arquivo, deslocando
# todas as colunas seguintes uma posição para trás. Ambos os mapeamentos
# foram conferidos comparando o valor de cada posição, em várias linhas,
# contra o domínio esperado do campo (ex.: SEXO ∈ {0,1,2}, datas em
# DD-MM-AAAA, CAUSABAS parecendo um código CID-10).
_ESQUEMAS_SEM_CABECALHO = {
    2022: _ESQUEMA_2021,
    2023: ["CONTADOR"]
    + [c for c in _ESQUEMA_2021 if c not in ("CONTADOR", "NECROPSIA")],
}


def _validar_primeira_linha_sem_cabecalho(caminho, ano, esquema):
    """Confere, lendo só a primeira linha de um CSV registrado em
    _ESQUEMAS_SEM_CABECALHO, que o layout ainda é o esperado: mesmo número
    de campos do esquema e primeira linha sendo um registro de dados (não
    uma linha de cabeçalho). Sem isso, o pandas preencheria com NaN ou
    deslocaria colunas em silêncio caso o arquivo fosse rebaixado com outro
    layout, ou leria um cabeçalho como se fosse um óbito."""
    with open(caminho, encoding="latin1", newline="") as f:
        primeira = next(csv.reader(f, delimiter=";"), [])
    if len(primeira) != len(esquema):
        raise ValueError(
            f"Ano {ano} ({Path(caminho).name}): a primeira linha tem "
            f"{len(primeira)} campos, mas o esquema sem cabeçalho esperado "
            f"tem {len(esquema)} — o layout do arquivo mudou; revise "
            "_ESQUEMAS_SEM_CABECALHO."
        )
    nomes = {c.upper() for c in esquema}
    if primeira and primeira[0].strip().strip('"').upper() in nomes:
        raise ValueError(
            f"Ano {ano} ({Path(caminho).name}): a primeira linha parece um "
            f"cabeçalho (primeiro campo {primeira[0]!r} é um nome de coluna), "
            "mas o ano está registrado como sem cabeçalho em "
            "_ESQUEMAS_SEM_CABECALHO — remova o ano de lá."
        )


def _caminho_ano(data_dir, ano):
    """Caminho do CSV do ano em `data_dir`; FileNotFoundError citando o ano
    se o arquivo não existir."""
    caminho = Path(data_dir) / f"Mortalidade_Geral_{ano}.csv"
    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo do ano {ano} não encontrado em {data_dir} "
            f"(esperado: {caminho.name}) — baixe-o do dados.gov.br e "
            "salve com esse nome (ver data/README.md)."
        )
    return caminho


def load_sim_anos(data_dir, anos, colunas=CORE_COLUMNS):
    """Carrega e concatena os arquivos `Mortalidade_Geral_<ano>.csv` de
    vários anos a partir de `data_dir`, mantendo apenas `colunas` (por
    padrão, as 8 confirmadas presentes em todo o intervalo 2000-2026 — ver
    CORE_COLUMNS) e adicionando a coluna `ANO_ARQUIVO` com o ano de origem
    do arquivo (não necessariamente igual ao ano em DTOBITO, por registros
    tardios).

    Levanta FileNotFoundError, citando o ano e o caminho esperado, se algum
    arquivo não existir em `data_dir` — evita uma mensagem genérica do
    pandas que não diz qual ano falhou.

    Trata corretamente os anos sem linha de cabeçalho no arquivo bruto
    (2022 e 2023 — ver _ESQUEMAS_SEM_CABECALHO), reconstruindo os nomes de
    coluna a partir do esquema completo já conferido para esses anos.

    DTOBITO é sempre devolvido como texto ddmmaaaa de 8 dígitos (o formato
    bruto varia por ano — ver comentário no corpo da função).
    """
    frames = []
    for ano in anos:
        caminho = _caminho_ano(data_dir, ano)
        # DTOBITO vem em formatos diferentes conforme o ano (ddmmaaaa com
        # zero à esquerda em 2000-2021, "dd-mm-aaaa" em 2022-2023, inteiro
        # sem zero à esquerda em 2024-2026); lê-se como texto e normaliza
        # para ddmmaaaa de 8 dígitos abaixo.
        dtype = {"DTOBITO": str} if "DTOBITO" in colunas else None
        if ano in _ESQUEMAS_SEM_CABECALHO:
            _validar_primeira_linha_sem_cabecalho(
                caminho, ano, _ESQUEMAS_SEM_CABECALHO[ano]
            )
            df_ano = pd.read_csv(
                caminho,
                sep=";",
                encoding="latin1",
                header=None,
                names=_ESQUEMAS_SEM_CABECALHO[ano],
                usecols=colunas,
                dtype=dtype,
                low_memory=False,
            )
        else:
            df_ano = pd.read_csv(
                caminho,
                sep=";",
                encoding="latin1",
                usecols=colunas,
                dtype=dtype,
                low_memory=False,
            )
        if "DTOBITO" in df_ano.columns:
            df_ano["DTOBITO"] = (
                df_ano["DTOBITO"].str.replace("-", "", regex=False).str.zfill(8)
            )
        df_ano["ANO_ARQUIVO"] = ano
        frames.append(df_ano)
    return pd.concat(frames, ignore_index=True)


# =============================================================================
# Preparação para modelagem
# =============================================================================

# Bump SEMPRE que mudar a lógica de load_sim_anos / preparar_df_modelo (ou os
# mapeamentos abaixo): a versão entra na chave do cache Parquet, então um
# cache antigo nunca é servido em silêncio.
PREP_VERSION = "1"

# Esquema único de escolaridade (ESC). Dicionário do SIM: 1 nenhuma, 2 de 1 a
# 3 anos, 3 de 4 a 7, 4 de 8 a 11, 5 12 e mais, 9 ignorado. Todo o resto
# (9, NaN, 0, 'A', '8', qualquer valor desconhecido) vira "Ignorado" — o 0
# só aparece em alguns anos e não consta no dicionário de ESC.
ESC_MAP = {
    1: "Nenhuma",
    2: "1a3anos",
    3: "4a7anos",
    4: "8a11anos",
    5: "12+anos",
}
# Raça/cor: 1 branca, 2 preta, 3 amarela, 4 parda, 5 indígena; 9/NaN/outros
# viram "Ignorado".
RACACOR_MAP = {1: "Branca", 2: "Preta", 3: "Amarela", 4: "Parda", 5: "Indígena"}
SEXO_MAP = {1: "Masculino", 2: "Feminino"}
ROTULO_IGNORADO = "Ignorado"

# Código IBGE de 2 dígitos da UF -> sigla (27 UFs).
UF_POR_CODIGO_IBGE = {
    11: "RO", 12: "AC", 13: "AM", 14: "RR", 15: "PA", 16: "AP", 17: "TO",
    21: "MA", 22: "PI", 23: "CE", 24: "RN", 25: "PB", 26: "PE", 27: "AL",
    28: "SE", 29: "BA", 31: "MG", 32: "ES", 33: "RJ", 35: "SP", 41: "PR",
    42: "SC", 43: "RS", 50: "MS", 51: "MT", 52: "GO", 53: "DF",
}

MODEL_COLUMNS = [
    "idade_anos",
    "sexo",
    "racacor",
    "escolaridade",
    "uf",
    "codmun6",
    "ano_arquivo",
    "capitulo_cid10",
]

# Entradas e rótulo dos modelos (notebook 03). Só estas cinco variáveis
# sociodemográficas entram como features: codmun6 (município fino),
# ano_arquivo e qualquer campo da cadeia de causas do atestado (LINHAA-D,
# CIRCOBITO, CAUSABAS_O) ficam de fora para não vazar o rótulo.
FEATURE_COLUMNS = ["idade_anos", "sexo", "racacor", "escolaridade", "uf"]
TARGET_COLUMN = "capitulo_cid10"


def decode_idade_anos_vetorizado(idade):
    """Versão vetorizada de `decode_idade_anos` (mesma regra, sem `.apply`
    linha a linha). Recebe uma Series com o campo IDADE (numérico ou texto) e
    devolve uma Series float com a idade em anos, NaN onde ignorada/ausente
    (000, 999, NaN, não numérico)."""
    codigo = pd.to_numeric(idade, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    unidade = np.floor(codigo / 100)
    valor = codigo - unidade * 100
    anos = np.where(unidade == 4, valor, np.where(unidade == 5, 100 + valor, 0.0))
    ignorado = np.isnan(codigo) | (codigo == 0) | (codigo == 999)
    anos = np.where(ignorado, np.nan, anos)
    return pd.Series(anos, index=getattr(idade, "index", None), name="idade_anos")


def _capitulo_cid10_vetorizado(causabas):
    """Capítulo CID-10 (numeral) de cada linha como Categorical; `causabas_to_chapter`
    é chamada uma única vez por código distinto. Sem capítulo -> código -1 (NaN)."""
    codigos, unicos = pd.factorize(causabas)
    categorias = [numeral for numeral, _, _, _ in _CHAPTER_RANGES]
    posicao = {numeral: i for i, numeral in enumerate(categorias)}
    lookup = np.array(
        [posicao.get(causabas_to_chapter(u)[0], -1) for u in unicos] + [-1],
        dtype=np.int16,
    )
    novos = lookup[np.where(codigos < 0, len(unicos), codigos)]
    return pd.Categorical.from_codes(novos, categories=categorias)


def _rotulos_brutos(serie, numerico):
    """Rótulo texto do valor bruto de cada linha ("1", "9", "A", "NaN"...),
    normalizando 1, 1.0 e "1" para o mesmo rótulo "1"."""
    codigos, unicos = pd.factorize(numerico)
    rot_unicos = [str(int(u)) if float(u).is_integer() else str(u) for u in unicos]
    rot = np.array(rot_unicos + ["NaN"], dtype=object)[
        np.where(codigos < 0, len(unicos), codigos)
    ]
    texto = np.isnan(numerico) & serie.notna().to_numpy()
    if texto.any():
        rot[texto] = serie.to_numpy(dtype=object)[texto].astype(str)
    return rot


def _categorizar(serie, ano_arquivo, mapa):
    """Mapeia um campo categórico numérico do SIM para rótulos texto segundo
    `mapa` (código -> rótulo); tudo que não está em `mapa` vira "Ignorado".
    Devolve (Categorical, distribuição bruta por ano {ano: {rótulo_bruto: n}})."""
    numerico = pd.to_numeric(serie, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    rotulos = list(dict.fromkeys(mapa.values())) + [ROTULO_IGNORADO]
    codigos = np.full(len(numerico), len(rotulos) - 1, dtype=np.int8)
    for codigo, rotulo in mapa.items():
        codigos[numerico == codigo] = rotulos.index(rotulo)
    categorica = pd.Categorical.from_codes(codigos, categories=rotulos)
    tabela = (
        pd.DataFrame({"ano": ano_arquivo, "bruto": _rotulos_brutos(serie, numerico)})
        .groupby(["ano", "bruto"])
        .size()
        .unstack(fill_value=0)
    )
    bruto_por_ano = {
        str(ano): {rot: int(n) for rot, n in linha.items() if n}
        for ano, linha in tabela.iterrows()
    }
    return categorica, bruto_por_ano


def tabela_mapeamento(bruto_por_ano, mapa):
    """Tabela bruto -> mapeado a partir da distribuição bruta por ano gerada
    por `preparar_df_modelo` (attrs["diagnosticos"]["esc_bruto_por_ano"] etc.):
    colunas `bruto`, `n` (total nos anos), `pct` e `mapeado`."""
    totais = pd.DataFrame(bruto_por_ano).T.fillna(0).sum()
    tab = totais.rename("n").astype(int).rename_axis("bruto").reset_index()

    def _mapeado(bruto):
        if bruto.lstrip("-").isdigit():
            return mapa.get(int(bruto), ROTULO_IGNORADO)
        return ROTULO_IGNORADO

    tab["mapeado"] = tab["bruto"].map(_mapeado)
    tab["pct"] = 100 * tab["n"] / tab["n"].sum()
    return tab.sort_values(["mapeado", "bruto"]).reset_index(drop=True)


def _municipio_e_uf(serie):
    """CODMUNRES (6 ou 7 dígitos IBGE, o 7º é o dígito verificador) ->
    (codmun6 Int32 com NA nos inválidos, uf Categorical com sigla ou
    "Ignorado"). Válido = 6 dígitos cujo prefixo de 2 dígitos é uma UF."""
    num = pd.to_numeric(serie, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    sete = (num >= 1_000_000) & (num < 10_000_000)
    seis = (num >= 100_000) & (num < 1_000_000)
    cod6 = np.where(sete, np.floor(num / 10), np.where(seis, num, np.nan))
    prefixo = np.floor(cod6 / 10_000)
    siglas = list(UF_POR_CODIGO_IBGE.values()) + [ROTULO_IGNORADO]
    idx = np.full(len(num), len(siglas) - 1, dtype=np.int8)
    for i, codigo in enumerate(UF_POR_CODIGO_IBGE):
        idx[prefixo == codigo] = i
    valido = idx != len(siglas) - 1
    codmun6 = pd.array(np.where(valido, cod6, np.nan), dtype="Int32")
    return codmun6, pd.Categorical.from_codes(idx, categories=siglas)


def preparar_df_modelo(df):
    """Transforma a saída de `load_sim_anos` no frame de modelagem, com as
    colunas de MODEL_COLUMNS:

    - `idade_anos` (int16): IDADE decodificada (vetorizada); ignorada -> linha descartada.
    - `sexo`: Masculino/Feminino; SEXO fora de {1, 2} (ex.: 0 = ignorado) -> linha descartada.
    - `racacor`, `escolaridade`: esquema único (RACACOR_MAP, ESC_MAP); qualquer
      valor não mapeável (9, NaN, 'A', '8', 0...) vira "Ignorado" (nunca
      descarta a linha).
    - `codmun6` (Int32) e `uf`: CODMUNRES truncado a 6 dígitos e UF derivada dos
      2 primeiros; município inválido -> codmun6 NA e uf "Ignorado".
    - `ano_arquivo` (int16) e `capitulo_cid10` (Categorical; alvo). Linha sem
      capítulo CID-10 (CAUSABAS ausente/fora do escopo) -> descartada.

    Os filtros são aplicados em sequência (sexo, idade, capítulo). O resultado
    traz em `attrs["diagnosticos"]` a contabilidade de linhas, a distribuição
    bruta de ESC e RACACOR por ano (antes do mapeamento) e a contagem de
    TIPOBITO das linhas finais.
    """
    n_bruto = len(df)
    ano = pd.to_numeric(df["ANO_ARQUIVO"]).to_numpy()

    escolaridade, esc_bruto = _categorizar(df["ESC"], ano, ESC_MAP)
    racacor, raca_bruto = _categorizar(df["RACACOR"], ano, RACACOR_MAP)

    sexo_num = pd.to_numeric(df["SEXO"], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    ok_sexo = np.isin(sexo_num, list(SEXO_MAP))
    idade = decode_idade_anos_vetorizado(df["IDADE"]).to_numpy()
    ok_idade = ~np.isnan(idade)
    capitulo = _capitulo_cid10_vetorizado(df["CAUSABAS"])
    ok_cap = capitulo.codes >= 0

    m_sexo = ok_sexo
    m_idade = m_sexo & ok_idade
    m_final = m_idade & ok_cap

    codmun6, uf = _municipio_e_uf(df["CODMUNRES"])
    sexo = pd.Categorical.from_codes(
        np.where(ok_sexo, sexo_num - 1, -1).astype(np.int8), categories=list(SEXO_MAP.values())
    )

    out = pd.DataFrame(
        {
            "idade_anos": pd.Series(idade[m_final]).astype("int16"),
            "sexo": sexo[m_final],
            "racacor": racacor[m_final],
            "escolaridade": escolaridade[m_final],
            "uf": uf[m_final].remove_unused_categories(),
            "codmun6": codmun6[m_final],
            "ano_arquivo": pd.Series(ano[m_final]).astype("int16"),
            "capitulo_cid10": capitulo[m_final].remove_unused_categories(),
        }
    )[MODEL_COLUMNS]

    n_sexo, n_idade, n_final = int(m_sexo.sum()), int(m_idade.sum()), int(m_final.sum())
    diagnosticos = {
        "contabilidade": {
            "n_bruto": n_bruto,
            "n_apos_sexo": n_sexo,
            "n_apos_idade": n_idade,
            "n_apos_capitulo": n_final,
            "n_final": n_final,
            "descartados_sexo": n_bruto - n_sexo,
            "descartados_idade": n_sexo - n_idade,
            "descartados_capitulo": n_idade - n_final,
        },
        "esc_bruto_por_ano": esc_bruto,
        "racacor_bruto_por_ano": raca_bruto,
    }
    if "TIPOBITO" in df.columns:
        tipo = df["TIPOBITO"].to_numpy(dtype=object)[m_final]
        rotulos, contagens = np.unique(pd.Series(tipo).astype(str), return_counts=True)
        diagnosticos["tipobito_final"] = {r: int(c) for r, c in zip(rotulos, contagens)}
    out.attrs["diagnosticos"] = diagnosticos
    return out


def _chave_cache(data_dir, anos):
    """Hash (16 hex) de: PREP_VERSION, lista de anos e (tamanho, mtime) de
    cada CSV de origem. FileNotFoundError (citando o ano) se faltar arquivo."""
    partes = []
    for ano in anos:
        st = _caminho_ano(data_dir, ano).stat()
        partes.append([int(ano), st.st_size, st.st_mtime_ns])
    payload = json.dumps({"prep_version": PREP_VERSION, "arquivos": partes})
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def carregar_df_modelo(
    data_dir="../data", anos=range(2000, 2027), cache_dir="../data/cache", refresh=False
):
    """Devolve `preparar_df_modelo(load_sim_anos(data_dir, anos))`, lendo de /
    gravando em um cache Parquet em `cache_dir` (`df_modelo_<chave>.parquet`
    + `.json` com os diagnósticos). A chave depende dos anos, do tamanho e
    mtime de cada CSV e de PREP_VERSION; mudou qualquer um, o cache antigo é
    ignorado (e um novo é construído). `refresh=True` força a reconstrução.
    """
    anos = list(anos)
    chave = _chave_cache(data_dir, anos)
    cache_dir = Path(cache_dir)
    parquet = cache_dir / f"df_modelo_{chave}.parquet"
    sidecar = cache_dir / f"df_modelo_{chave}.json"
    if not refresh and parquet.exists() and sidecar.exists():
        df = pd.read_parquet(parquet)
        df.attrs["diagnosticos"] = json.loads(sidecar.read_text(encoding="utf-8"))
        return df
    df = preparar_df_modelo(load_sim_anos(data_dir, anos))
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_parquet = parquet.with_name(parquet.name + ".tmp")
    df.to_parquet(tmp_parquet, index=False)
    os.replace(tmp_parquet, parquet)
    sidecar.write_text(json.dumps(df.attrs["diagnosticos"]), encoding="utf-8")
    return df
