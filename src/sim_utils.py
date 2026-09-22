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
# =============================================================================
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
        df_ano = pd.read_csv(
            caminho, sep=";", encoding="latin1", usecols=colunas, low_memory=False
        )
        df_ano["ANO_ARQUIVO"] = ano
        frames.append(df_ano)
    return pd.concat(frames, ignore_index=True)
