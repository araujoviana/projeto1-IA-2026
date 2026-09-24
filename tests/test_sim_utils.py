# =============================================================================
# Integrantes: Arthur Meneses Neves, Danilo Oliveira Santos, Matheus Gabriel
# Viana Araujo, João Victor Vidal Barbosa, Guilherme Araujo Castro
#
# Síntese: Testes das funções de decodificação de campos do SIM
# (sim_utils.py): idade codificada e mapeamento de CAUSABAS para capítulo
# CID-10.
#
# Changelog:
#   2026-09-22 — Matheus Araujo — criação inicial do arquivo
#   2026-09-22 — Matheus Araujo — testa load_sim_anos para os anos sem
#     cabeçalho (2022, 2023), achado ao rodar a EDA sobre dados reais
# =============================================================================
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import sim_utils
from sim_utils import causabas_to_chapter, decode_idade_anos, load_sim_anos

assert sim_utils  # usado via sim_utils._ESQUEMAS_SEM_CABECALHO nos testes abaixo


def test_decode_idade_anos_em_anos():
    assert decode_idade_anos(423) == 23
    assert decode_idade_anos(400) == 0
    assert decode_idade_anos(499) == 99


def test_decode_idade_anos_ignorado_ou_ausente():
    assert decode_idade_anos(0) is None
    assert decode_idade_anos(999) is None
    assert decode_idade_anos(None) is None


def test_decode_idade_anos_horas_dias_meses_arredondam_para_zero():
    assert decode_idade_anos(205) == 0  # 5 dias
    assert decode_idade_anos(311) == 0  # 11 meses


def test_causabas_to_chapter_circulatorio():
    assert causabas_to_chapter("I219") == ("IX", "Doenças do aparelho circulatório")


def test_causabas_to_chapter_neoplasia():
    assert causabas_to_chapter("C349") == ("II", "Neoplasias (tumores)")


def test_causabas_to_chapter_causas_mal_definidas():
    assert causabas_to_chapter("R98") == (
        "XVIII",
        "Sintomas, sinais e achados anormais, causas mal definidas",
    )


def test_causabas_to_chapter_causas_externas_com_asterisco():
    assert causabas_to_chapter("*T794") == (
        "XIX",
        "Lesões, envenenamento e outras causas externas",
    )


def test_causabas_to_chapter_codigo_cid9_fora_de_escopo():
    # Registros anteriores a 1996 usam CID-9 (puramente numérico) — fora do
    # escopo do alvo de classificação (ver Scope note do plano).
    assert causabas_to_chapter("4239") == (None, None)


def test_causabas_to_chapter_valor_ausente():
    assert causabas_to_chapter(None) == (None, None)
    assert causabas_to_chapter("") == (None, None)


def _escrever_csv_fake(caminho, linha_dados):
    caminho.write_text(
        '"TIPOBITO";"DTOBITO";"IDADE";"SEXO";"RACACOR";"ESC";"CODMUNRES";"CAUSABAS"\n'
        f"{linha_dados}\n",
        encoding="latin1",
    )


def test_load_sim_anos_concatena_e_marca_ano_de_origem(tmp_path):
    _escrever_csv_fake(
        tmp_path / "Mortalidade_Geral_2000.csv",
        '"2";"01012000";"423";"1";"1";"3";"3550308";"I219"',
    )
    _escrever_csv_fake(
        tmp_path / "Mortalidade_Geral_2001.csv",
        '"2";"02012001";"365";"2";"4";"2";"3304557";"C349"',
    )

    df = load_sim_anos(tmp_path, [2000, 2001])

    assert len(df) == 2
    assert list(df["ANO_ARQUIVO"]) == [2000, 2001]
    assert set(df.columns) == {
        "TIPOBITO",
        "DTOBITO",
        "IDADE",
        "SEXO",
        "RACACOR",
        "ESC",
        "CODMUNRES",
        "CAUSABAS",
        "ANO_ARQUIVO",
    }


def test_load_sim_anos_arquivo_ausente_da_erro_com_o_ano(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError, match="1979"):
        load_sim_anos(tmp_path, [1979])


def _escrever_csv_sem_cabecalho(caminho, esquema, valores_por_coluna):
    """Escreve uma linha de dados (sem cabeçalho) na ordem de `esquema`,
    usando valores_por_coluna para as colunas conhecidas e "" para as
    demais — imita o formato real de 2022/2023 do SIM."""
    linha = ";".join(valores_por_coluna.get(col, "") for col in esquema)
    caminho.write_text(linha + "\n", encoding="latin1")


def test_load_sim_anos_ano_2022_sem_cabecalho():
    # 2022 não tem cabeçalho, mas usa o mesmo esquema de 89 colunas de 2021
    # (CONTADOR ao final) — ver _ESQUEMAS_SEM_CABECALHO.
    import tempfile

    valores = {
        "TIPOBITO": "2",
        "DTOBITO": "21-04-2022",
        "IDADE": "499",
        "SEXO": "2",
        "RACACOR": "1",
        "ESC": "4",
        "CODMUNRES": "354850",
        "CAUSABAS": "I219",
    }
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _escrever_csv_sem_cabecalho(
            tmp_path / "Mortalidade_Geral_2022.csv",
            sim_utils._ESQUEMAS_SEM_CABECALHO[2022],
            valores,
        )
        df = load_sim_anos(tmp_path, [2022])

    assert len(df) == 1
    for col, valor in valores.items():
        esperado = valor.replace("-", "") if col == "DTOBITO" else valor
        assert str(df.iloc[0][col]) == esperado


def test_load_sim_anos_ano_2023_sem_cabecalho_e_sem_necropsia():
    # 2023 não tem cabeçalho e não tem a coluna NECROPSIA (86 colunas,
    # CONTADOR na 1ª posição) — ver _ESQUEMAS_SEM_CABECALHO.
    import tempfile

    valores = {
        "TIPOBITO": "2",
        "DTOBITO": "14-02-2023",
        "IDADE": "468",
        "SEXO": "1",
        "RACACOR": "4",
        "ESC": "2",
        "CODMUNRES": "330190",
        "CAUSABAS": "I10",
    }
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _escrever_csv_sem_cabecalho(
            tmp_path / "Mortalidade_Geral_2023.csv",
            sim_utils._ESQUEMAS_SEM_CABECALHO[2023],
            valores,
        )
        df = load_sim_anos(tmp_path, [2023])

    assert len(df) == 1
    for col, valor in valores.items():
        esperado = valor.replace("-", "") if col == "DTOBITO" else valor
        assert str(df.iloc[0][col]) == esperado


def test_load_sim_anos_normaliza_dtobito_para_ddmmaaaa_de_8_digitos(tmp_path):
    # 2024-2026 trazem DTOBITO como inteiro sem zero à esquerda (ex.: 8112024
    # = 08/11/2024); 2000-2021 já trazem 8 dígitos.
    _escrever_csv_fake(
        tmp_path / "Mortalidade_Geral_2024.csv",
        '"2";"8112024";"476";"1";"3";"1";"241360";"J189"',
    )
    _escrever_csv_fake(
        tmp_path / "Mortalidade_Geral_2000.csv",
        '"2";"23032000";"423";"1";"1";"3";"3550308";"I219"',
    )

    df = load_sim_anos(tmp_path, [2000, 2024])

    assert list(df["DTOBITO"]) == ["23032000", "08112024"]
