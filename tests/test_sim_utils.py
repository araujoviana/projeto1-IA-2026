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
# =============================================================================
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sim_utils import causabas_to_chapter, decode_idade_anos, load_sim_anos


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
