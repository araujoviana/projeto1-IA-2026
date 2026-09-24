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
# =============================================================================
import csv
import math
import re
from pathlib import Path

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
        caminho = Path(data_dir) / f"Mortalidade_Geral_{ano}.csv"
        if not caminho.exists():
            raise FileNotFoundError(
                f"Arquivo do ano {ano} não encontrado em {data_dir} "
                f"(esperado: {caminho.name}) — baixe-o do dados.gov.br e "
                "salve com esse nome (ver data/README.md)."
            )
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
