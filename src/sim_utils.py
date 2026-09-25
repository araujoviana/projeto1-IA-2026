# =============================================================================
# Integrantes: Arthur Meneses Neves, Danilo Oliveira Santos, Matheus Gabriel
# Viana Araujo, João Victor Vidal Barbosa, Guilherme Araujo Castro
#
# Síntese: Funções de decodificação de campos do SIM (Sistema de Informação
# sobre Mortalidade): idade codificada (IDADE) e mapeamento da causa básica
# de óbito (CAUSABAS, CID-10) para capítulo CID-10.
#
# Changelog:
#   2026-09-22 - Matheus Araujo - criação: decodificação de IDADE e CAUSABAS
#   2026-09-22 - Matheus Araujo - load_sim_anos: anos 2022 e 2023 sem cabeçalho
#   2026-09-23 - Matheus Araujo - load_sim_anos: normaliza DTOBITO e valida a
#     primeira linha dos anos sem cabeçalho
#   2026-09-23 - Matheus Araujo - preparar_df_modelo e carregar_df_modelo
#     (cache em Parquet)
#   2026-09-23 - Matheus Araujo - FEATURE_COLUMNS e TARGET_COLUMN
#   2026-09-24 - Matheus Araujo - ajuste de comentários e docstrings
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

# Colunas que existem com o mesmo nome em todos os anos (conferido em 2000,
# 2005, 2010, 2015, 2020, 2024 e 2026). O esquema completo cresceu de 39 para
# 88 colunas no período (ver data/README.md); reconferir antes de adicionar
# colunas aqui.
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
# Faixas da CID-10 (OMS). Comparar as strings "letra + 2 dígitos" funciona
# porque a ordem alfabética coincide com a ordem das faixas (ex.: o capítulo II
# vai de C00 a D48).
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
    """Converte o campo IDADE do SIM em idade em anos completos.

    O código tem 3 dígitos: o primeiro é a unidade (1=hora, 2=dia, 3=mês,
    4=ano, 5=100 anos ou mais) e os dois últimos são o valor. Devolve None se
    o código for nulo ou ignorado (000, 999) e 0 para horas, dias e meses.
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
    """Recebe um CAUSABAS (ex.: 'I219', '*T794', 'C349') e devolve
    (numeral, descrição) do capítulo da CID-10, ou (None, None) se o código
    não for CID-10. Códigos só numéricos (CID-9, usados até 1995) ficam de
    fora de propósito (ver data/README.md).
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


# Colunas de 2021 (87, na ordem do arquivo), usadas para reconstruir o
# cabeçalho dos anos cujo CSV não tem linha de cabeçalho (ver
# _ESQUEMAS_SEM_CABECALHO). Os nomes foram conferidos comparando o valor de
# cada posição com o que o campo deveria conter (datas, município, CID-10).
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

# Os CSVs de 2022 e 2023 não têm linha de cabeçalho (a primeira linha já é um
# óbito); os demais anos têm. Descoberto ao rodar a EDA do notebook 01.
#
# 2022 tem as mesmas 87 colunas de 2021. 2023 tem 86: CONTADOR vem na primeira
# posição e NECROPSIA não existe, o que desloca as colunas seguintes. Os dois
# mapeamentos foram conferidos olhando o valor de cada posição em várias linhas
# (SEXO em {0,1,2}, datas em DD-MM-AAAA, CAUSABAS parecido com CID-10).
_ESQUEMAS_SEM_CABECALHO = {
    2022: _ESQUEMA_2021,
    2023: ["CONTADOR"]
    + [c for c in _ESQUEMA_2021 if c not in ("CONTADOR", "NECROPSIA")],
}


def _validar_primeira_linha_sem_cabecalho(caminho, ano, esquema):
    """Lê a primeira linha de um CSV de _ESQUEMAS_SEM_CABECALHO e confere se o
    número de campos bate com o esquema e se ela é mesmo um registro, não um
    cabeçalho. Sem isso, um arquivo com outro layout seria lido errado sem
    aviso."""
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
    """Caminho do CSV do ano em `data_dir`; levanta FileNotFoundError citando
    o ano se o arquivo não existir."""
    caminho = Path(data_dir) / f"Mortalidade_Geral_{ano}.csv"
    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo do ano {ano} não encontrado em {data_dir} "
            f"(esperado: {caminho.name}) — baixe-o do dados.gov.br e "
            "salve com esse nome (ver data/README.md)."
        )
    return caminho


def load_sim_anos(data_dir, anos, colunas=CORE_COLUMNS):
    """Carrega e concatena os `Mortalidade_Geral_<ano>.csv` dos `anos` em
    `data_dir`, mantendo só `colunas` (por padrão CORE_COLUMNS) e
    acrescentando `ANO_ARQUIVO`, o ano do arquivo (pode diferir do ano de
    DTOBITO por causa de registros tardios).

    Levanta FileNotFoundError citando o ano se faltar algum arquivo. Os anos
    sem cabeçalho (2022 e 2023) recebem os nomes de _ESQUEMAS_SEM_CABECALHO.
    DTOBITO sai sempre como texto ddmmaaaa de 8 dígitos.
    """
    frames = []
    for ano in anos:
        caminho = _caminho_ano(data_dir, ano)
        # O formato de DTOBITO muda com o ano (ddmmaaaa em 2000-2021,
        # dd-mm-aaaa em 2022-2023, inteiro sem zero à esquerda em 2024-2026).
        # Lê como texto e normaliza para ddmmaaaa mais abaixo.
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

# Aumentar sempre que mudar a lógica de load_sim_anos, preparar_df_modelo ou
# os mapeamentos abaixo. A versão entra na chave do cache Parquet, então um
# cache antigo não é reaproveitado por engano.
PREP_VERSION = "1"

# Escolaridade (ESC), pelo dicionário do SIM: 1 nenhuma, 2 de 1 a 3 anos, 3 de
# 4 a 7, 4 de 8 a 11, 5 12 ou mais, 9 ignorado. Qualquer outro valor (9, NaN,
# 0, 'A', '8') vira "Ignorado". O 0 só aparece em alguns anos e não está no
# dicionário.
ESC_MAP = {
    1: "Nenhuma",
    2: "1a3anos",
    3: "4a7anos",
    4: "8a11anos",
    5: "12+anos",
}
# Raça/cor: 1 branca, 2 preta, 3 amarela, 4 parda, 5 indígena. O resto vira
# "Ignorado".
RACACOR_MAP = {1: "Branca", 2: "Preta", 3: "Amarela", 4: "Parda", 5: "Indígena"}
SEXO_MAP = {1: "Masculino", 2: "Feminino"}
ROTULO_IGNORADO = "Ignorado"

# Código IBGE da UF (2 dígitos) para a sigla.
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

# Entradas e rótulo dos modelos (notebook 03). Só estas cinco variáveis entram
# como features. codmun6, ano_arquivo e os campos da cadeia de causas do
# atestado (LINHAA a LINHAD, CIRCOBITO, CAUSABAS_O) ficam de fora para não
# vazar o rótulo.
FEATURE_COLUMNS = ["idade_anos", "sexo", "racacor", "escolaridade", "uf"]
TARGET_COLUMN = "capitulo_cid10"


def decode_idade_anos_vetorizado(idade):
    """Versão de `decode_idade_anos` para uma Series inteira, sem `.apply`.
    Devolve a idade em anos, com NaN onde o valor é ignorado ou ausente."""
    codigo = pd.to_numeric(idade, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    unidade = np.floor(codigo / 100)
    valor = codigo - unidade * 100
    anos = np.where(unidade == 4, valor, np.where(unidade == 5, 100 + valor, 0.0))
    ignorado = np.isnan(codigo) | (codigo == 0) | (codigo == 999)
    anos = np.where(ignorado, np.nan, anos)
    return pd.Series(anos, index=getattr(idade, "index", None), name="idade_anos")


def _capitulo_cid10_vetorizado(causabas):
    """Capítulo CID-10 de cada linha como Categorical. Chama
    `causabas_to_chapter` uma vez por código distinto; sem capítulo vira NaN."""
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
    """Rótulo em texto do valor bruto de cada linha ("1", "9", "A", "NaN"),
    tratando 1, 1.0 e "1" como o mesmo valor."""
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
    """Mapeia um campo numérico do SIM para rótulos segundo `mapa` (código para
    rótulo); o que não estiver em `mapa` vira "Ignorado". Devolve o
    Categorical e a contagem dos valores brutos por ano."""
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
    """Tabela com as colunas `bruto`, `n`, `pct` e `mapeado`, montada a partir
    da contagem por ano que `preparar_df_modelo` guarda em
    attrs["diagnosticos"] (ex.: "esc_bruto_por_ano")."""
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
    """Converte CODMUNRES (6 ou 7 dígitos IBGE; o 7º é o verificador) em
    codmun6 (Int32, NA se inválido) e uf (sigla ou "Ignorado"). É válido o
    código de 6 dígitos cujo prefixo é uma UF."""
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
    """Transforma a saída de `load_sim_anos` no frame de modelagem
    (colunas de MODEL_COLUMNS).

    - idade_anos: IDADE decodificada; se ignorada, a linha é descartada.
    - sexo: Masculino ou Feminino; qualquer outro valor descarta a linha.
    - racacor e escolaridade: ver RACACOR_MAP e ESC_MAP; o que não mapeia vira
      "Ignorado" e a linha é mantida.
    - codmun6 e uf: derivados de CODMUNRES; município inválido vira uf "Ignorado".
    - ano_arquivo e capitulo_cid10 (o alvo); sem capítulo, a linha é descartada.

    Os filtros rodam em sequência (sexo, idade, capítulo). Em
    `attrs["diagnosticos"]` ficam a contagem de linhas descartadas em cada
    etapa, a distribuição bruta de ESC e RACACOR por ano e a de TIPOBITO.
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
    """Hash de 16 caracteres calculado a partir de PREP_VERSION, dos anos e do
    tamanho e mtime de cada CSV. Levanta FileNotFoundError se faltar arquivo."""
    partes = []
    for ano in anos:
        st = _caminho_ano(data_dir, ano).stat()
        partes.append([int(ano), st.st_size, st.st_mtime_ns])
    payload = json.dumps({"prep_version": PREP_VERSION, "arquivos": partes})
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def carregar_df_modelo(
    data_dir="../data", anos=range(2000, 2027), cache_dir="../data/cache", refresh=False
):
    """Devolve `preparar_df_modelo(load_sim_anos(data_dir, anos))`, usando um
    cache Parquet em `cache_dir` (o arquivo .json ao lado guarda os
    diagnósticos). A chave do cache muda se mudarem os anos, os CSVs ou
    PREP_VERSION. Com `refresh=True` o cache é refeito.
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
