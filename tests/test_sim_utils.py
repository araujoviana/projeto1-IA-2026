# =============================================================================
# Integrantes: Arthur Meneses Neves, Danilo Oliveira Santos, Matheus Gabriel
# Viana Araujo, João Victor Vidal Barbosa, Guilherme Araujo Castro
#
# Síntese: Testes das funções de decodificação de campos do SIM
# (sim_utils.py): idade codificada e mapeamento de CAUSABAS para capítulo
# CID-10.
#
# Changelog:
#   2026-09-22 - Matheus Araujo - criação: testes de idade e CAUSABAS
#   2026-09-22 - Matheus Araujo - testes de load_sim_anos (anos sem cabeçalho)
#   2026-09-23 - Matheus Araujo - testes de preparar_df_modelo e do cache
#   2026-09-23 - Matheus Araujo - testes de FEATURE_COLUMNS e TARGET_COLUMN
#   2026-09-24 - Matheus Araujo - ajuste de comentários
# =============================================================================
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

import sim_utils
from sim_utils import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    carregar_df_modelo,
    causabas_to_chapter,
    decode_idade_anos,
    load_sim_anos,
    preparar_df_modelo,
)

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
    # escopo do alvo de classificação (ver "Escopo temporal do projeto" em data/README.md).
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
    # 2022 não tem cabeçalho, mas usa o mesmo esquema de 87 colunas de 2021
    # (CONTADOR ao final). Ver _ESQUEMAS_SEM_CABECALHO.
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
    # CONTADOR na 1ª posição). Ver _ESQUEMAS_SEM_CABECALHO.
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


def _linha_2022_literal():
    """Linha de dados de 2022 (87 campos) montada com posições LITERAIS
    (não derivadas de _ESQUEMAS_SEM_CABECALHO): 0-based, DTOBITO=2, IDADE=7,
    SEXO=8, RACACOR=9, ESC=11, CODMUNRES=15, CAUSABAS=45 (esquema de 2021, conferido no arquivo real)."""
    campos = [""] * 87
    campos[1] = "2"
    campos[2] = "21-04-2022"
    campos[7] = "499"
    campos[8] = "2"
    campos[9] = "1"
    campos[11] = "4"
    campos[15] = "354850"
    campos[45] = "I219"
    return campos


def test_load_sim_anos_2022_posicoes_literais(tmp_path):
    (tmp_path / "Mortalidade_Geral_2022.csv").write_text(
        ";".join(_linha_2022_literal()) + "\n", encoding="latin1"
    )
    df = load_sim_anos(tmp_path, [2022])
    linha = df.iloc[0]
    assert linha["CAUSABAS"] == "I219"
    assert linha["SEXO"] == 2
    assert linha["CODMUNRES"] == 354850
    assert linha["DTOBITO"] == "21042022"


def test_load_sim_anos_sem_cabecalho_com_numero_de_campos_errado(tmp_path):
    import pytest

    campos = _linha_2022_literal() + ["EXTRA"]  # 88 campos em vez de 87
    (tmp_path / "Mortalidade_Geral_2022.csv").write_text(
        ";".join(campos) + "\n", encoding="latin1"
    )
    with pytest.raises(ValueError, match=r"2022.*88.*87"):
        load_sim_anos(tmp_path, [2022])


def test_load_sim_anos_sem_cabecalho_mas_arquivo_tem_cabecalho(tmp_path):
    import pytest

    cabecalho = ";".join(f'"{c}"' for c in sim_utils._ESQUEMAS_SEM_CABECALHO[2023])
    dado = ";".join([""] * len(sim_utils._ESQUEMAS_SEM_CABECALHO[2023]))
    (tmp_path / "Mortalidade_Geral_2023.csv").write_text(
        cabecalho + "\n" + dado + "\n", encoding="latin1"
    )
    with pytest.raises(ValueError, match=r"2023.*cabeçalho"):
        load_sim_anos(tmp_path, [2023])


# ---------------------------------------------------------------------------
# Preparação para modelagem: preparar_df_modelo / carregar_df_modelo
# ---------------------------------------------------------------------------


def test_decode_idade_vetorizado_igual_ao_escalar_em_todas_as_unidades():
    codigos = [
        np.nan, 0, 999,  # ausente / ignorado
        101, 123,  # horas
        201, 230,  # dias
        301, 311,  # meses
        400, 423, 499,  # anos
        500, 512,  # 100+ anos
        "423", "999", "0",  # texto (arquivos com dtype object)
    ]
    serie = pd.Series(codigos, dtype=object)
    obtido = sim_utils.decode_idade_anos_vetorizado(serie)
    esperado = [
        np.nan if decode_idade_anos(c if not isinstance(c, str) else float(c)) is None
        else decode_idade_anos(c if not isinstance(c, str) else float(c))
        for c in codigos
    ]
    assert len(obtido) == len(esperado)
    np.testing.assert_array_equal(obtido.to_numpy(dtype=float), np.array(esperado, dtype=float))


def _df_bruto():
    """Frame sintético no formato de saída de load_sim_anos, com as
    irregularidades vistas nos dados reais: ESC misto (texto em 2000-2001,
    'A', '8', 0, 9, NaN), SEXO 0, IDADE ignorada, CODMUNRES de 6 e 7 dígitos,
    CAUSABAS ausente."""
    return pd.DataFrame(
        {
            "TIPOBITO": [2, 2, 2, 2, 2, 2, 2, 2, 2],
            "DTOBITO": ["01012000"] * 9,
            "IDADE": [423, 512, 205, 999, 430, 425, 440, 450, 460],
            "SEXO": [1, 2, 0, 1, 2, 1, 2, 1, 1],
            "RACACOR": [1, 2, 3, 4, 5, np.nan, 9, 1, 1],
            "ESC": ["1", "2", "9", "A", "8", 3.0, np.nan, 0.0, 5.0],
            "CODMUNRES": [3550308, 355030, 3304557, 330455, 5300108, 999999, np.nan, 3550308, 4314902],
            "CAUSABAS": ["I219", "C349", "R98", "I10", "V892", "*T794", "J189", None, "O800"],
            "ANO_ARQUIVO": [2000, 2000, 2001, 2001, 2022, 2022, 2023, 2023, 2023],
        }
    )


def test_preparar_df_modelo_colunas_e_tipos():
    out = preparar_df_modelo(_df_bruto())
    assert list(out.columns) == [
        "idade_anos", "sexo", "racacor", "escolaridade", "uf",
        "codmun6", "ano_arquivo", "capitulo_cid10",
    ]
    assert not out.isna().any().any() or list(out.columns[out.isna().any()]) == ["codmun6"]


def test_preparar_df_modelo_filtros_e_contabilidade():
    out = preparar_df_modelo(_df_bruto())
    c = out.attrs["diagnosticos"]["contabilidade"]
    # linha 2 (SEXO 0) sai; linha 3 (IDADE 999) sai; linha 7 (CAUSABAS
    # ausente, IDADE 460) sai; e "*T794" (linha 5) vira capítulo XIX.
    assert c["n_bruto"] == 9
    assert c["n_apos_sexo"] == 8
    assert c["n_apos_idade"] == 7
    assert c["n_apos_capitulo"] == 6
    assert c["n_final"] == 6 == len(out)
    assert c["descartados_sexo"] == 1
    assert c["descartados_idade"] == 1
    assert c["descartados_capitulo"] == 1
    assert set(out["sexo"].astype(str)) == {"Masculino", "Feminino"}


def test_preparar_df_modelo_valores_por_linha():
    out = preparar_df_modelo(_df_bruto()).reset_index(drop=True)
    # sobreviventes: linhas 0, 1, 4, 5, 6, 8 do bruto
    assert list(out["idade_anos"]) == [23, 112, 30, 25, 40, 60]
    assert list(out["capitulo_cid10"].astype(str)) == ["IX", "II", "XX", "XIX", "X", "XV"]
    assert list(out["sexo"].astype(str)) == ["Masculino", "Feminino", "Feminino", "Masculino", "Feminino", "Masculino"]
    assert list(out["racacor"].astype(str)) == ["Branca", "Preta", "Indígena", "Ignorado", "Ignorado", "Branca"]
    # ESC: "1"->Nenhuma, "2"->1a3, "8"->Ignorado, 3.0->4a7, NaN->Ignorado, 5.0->12+
    assert list(out["escolaridade"].astype(str)) == [
        "Nenhuma", "1a3anos", "Ignorado", "4a7anos", "Ignorado", "12+anos",
    ]
    assert list(out["ano_arquivo"]) == [2000, 2000, 2022, 2022, 2023, 2023]


def test_preparar_df_modelo_municipio_seis_digitos_e_uf():
    out = preparar_df_modelo(_df_bruto()).reset_index(drop=True)
    # 3550308 e 355030 -> 355030; 5300108 -> 530010 (DF); 999999 e NaN -> inválido
    assert out["codmun6"].iloc[0] == 355030
    assert out["codmun6"].iloc[1] == 355030
    assert out["codmun6"].iloc[2] == 530010
    assert pd.isna(out["codmun6"].iloc[3])  # 999999 não é município válido
    assert pd.isna(out["codmun6"].iloc[4])  # NaN
    assert list(out["uf"].astype(str)) == ["SP", "SP", "DF", "Ignorado", "Ignorado", "RS"]


def test_preparar_df_modelo_diagnostico_bruto_por_ano_e_tabela_de_mapeamento():
    out = preparar_df_modelo(_df_bruto())
    diag = out.attrs["diagnosticos"]
    esc = pd.DataFrame(diag["esc_bruto_por_ano"]).T.fillna(0).astype(int)
    # ano 2000 tem ESC "1" e "2"; 2001 tem "9" e "A"; 2022 tem "8" e 3
    assert esc.loc["2000", "1"] == 1 and esc.loc["2000", "2"] == 1
    assert esc.loc["2001", "9"] == 1 and esc.loc["2001", "A"] == 1
    assert esc.loc["2022", "8"] == 1 and esc.loc["2022", "3"] == 1
    assert esc.loc["2023", "NaN"] == 1 and esc.loc["2023", "0"] == 1
    assert esc.to_numpy().sum() == 9  # calculado sobre os 9 registros brutos
    tab = sim_utils.tabela_mapeamento(diag["esc_bruto_por_ano"], sim_utils.ESC_MAP)
    assert dict(zip(tab["bruto"], tab["mapeado"])) == {
        "0": "Ignorado", "1": "Nenhuma", "2": "1a3anos", "3": "4a7anos",
        "5": "12+anos", "8": "Ignorado", "9": "Ignorado", "A": "Ignorado",
        "NaN": "Ignorado",
    }
    assert tab["n"].sum() == 9


def _csv_ano(tmp_path, ano, linhas):
    tmp_path.mkdir(parents=True, exist_ok=True)
    caminho = tmp_path / f"Mortalidade_Geral_{ano}.csv"
    caminho.write_text(
        '"TIPOBITO";"DTOBITO";"IDADE";"SEXO";"RACACOR";"ESC";"CODMUNRES";"CAUSABAS"\n'
        + "".join(l + "\n" for l in linhas),
        encoding="latin1",
    )
    return caminho


def _dataset_sintetico(tmp_path):
    _csv_ano(tmp_path, 2000, [
        '"2";"01012000";"423";"1";"1";"3";"3550308";"I219"',
        '"2";"02012000";"365";"2";"4";"2";"3304557";"C349"',
    ])
    _csv_ano(tmp_path, 2001, [
        '"2";"02012001";"430";"2";"4";"9";"355030";"J189"',
    ])
    return tmp_path


def _contador_de_leituras(monkeypatch):
    chamadas = []
    original = sim_utils.load_sim_anos

    def contando(*args, **kwargs):
        chamadas.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(sim_utils, "load_sim_anos", contando)
    return chamadas


def test_carregar_df_modelo_segunda_chamada_le_do_cache_sem_reler_csvs(tmp_path, monkeypatch):
    dados = _dataset_sintetico(tmp_path / "data")
    cache = tmp_path / "cache"
    chamadas = _contador_de_leituras(monkeypatch)

    df1 = carregar_df_modelo(dados, [2000, 2001], cache)
    assert len(chamadas) == 1
    assert len(list(cache.glob("*.parquet"))) == 1
    df2 = carregar_df_modelo(dados, [2000, 2001], cache)
    assert len(chamadas) == 1  # cache hit: CSVs não foram relidos
    pd.testing.assert_frame_equal(df1, df2)
    assert df2.attrs["diagnosticos"]["contabilidade"]["n_final"] == 3
    assert df2.attrs["diagnosticos"] == df1.attrs["diagnosticos"]


def test_carregar_df_modelo_invalida_cache_quando_prep_version_muda(tmp_path, monkeypatch):
    dados = _dataset_sintetico(tmp_path / "data")
    cache = tmp_path / "cache"
    chamadas = _contador_de_leituras(monkeypatch)
    carregar_df_modelo(dados, [2000, 2001], cache)
    monkeypatch.setattr(sim_utils, "PREP_VERSION", sim_utils.PREP_VERSION + "-novo")
    carregar_df_modelo(dados, [2000, 2001], cache)
    assert len(chamadas) == 2


def test_carregar_df_modelo_invalida_cache_quando_csv_muda_de_tamanho(tmp_path, monkeypatch):
    dados = _dataset_sintetico(tmp_path / "data")
    cache = tmp_path / "cache"
    chamadas = _contador_de_leituras(monkeypatch)
    df1 = carregar_df_modelo(dados, [2000, 2001], cache)
    with open(dados / "Mortalidade_Geral_2001.csv", "a", encoding="latin1") as f:
        f.write('"2";"03012001";"440";"1";"1";"4";"355030";"C349"\n')
    df2 = carregar_df_modelo(dados, [2000, 2001], cache)
    assert len(chamadas) == 2
    assert len(df2) == len(df1) + 1


def test_carregar_df_modelo_invalida_cache_quando_csv_muda_de_mtime(tmp_path, monkeypatch):
    import os

    dados = _dataset_sintetico(tmp_path / "data")
    cache = tmp_path / "cache"
    chamadas = _contador_de_leituras(monkeypatch)
    carregar_df_modelo(dados, [2000, 2001], cache)
    arq = dados / "Mortalidade_Geral_2000.csv"
    st = arq.stat()
    os.utime(arq, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    carregar_df_modelo(dados, [2000, 2001], cache)
    assert len(chamadas) == 2


def test_carregar_df_modelo_conjunto_de_anos_diferente_usa_outra_chave(tmp_path, monkeypatch):
    dados = _dataset_sintetico(tmp_path / "data")
    cache = tmp_path / "cache"
    chamadas = _contador_de_leituras(monkeypatch)
    a = carregar_df_modelo(dados, [2000, 2001], cache)
    b = carregar_df_modelo(dados, [2000], cache)
    assert len(chamadas) == 2
    assert len(b) < len(a)
    carregar_df_modelo(dados, [2000, 2001], cache)  # ainda em cache
    assert len(chamadas) == 2


def test_carregar_df_modelo_refresh_forca_reconstrucao(tmp_path, monkeypatch):
    dados = _dataset_sintetico(tmp_path / "data")
    cache = tmp_path / "cache"
    chamadas = _contador_de_leituras(monkeypatch)
    carregar_df_modelo(dados, [2000, 2001], cache)
    carregar_df_modelo(dados, [2000, 2001], cache, refresh=True)
    assert len(chamadas) == 2


def test_carregar_df_modelo_arquivo_ausente_da_erro_com_o_ano(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError, match="1979"):
        carregar_df_modelo(tmp_path, [1979], tmp_path / "cache")


def test_carregar_df_modelo_equivale_a_preparar_sobre_load_sim_anos(tmp_path):
    dados = _dataset_sintetico(tmp_path / "data")
    direto = preparar_df_modelo(load_sim_anos(dados, [2000, 2001]))
    via_cache = carregar_df_modelo(dados, [2000, 2001], tmp_path / "cache")
    pd.testing.assert_frame_equal(direto, via_cache)


def test_feature_columns_sao_as_cinco_variaveis_sociodemograficas():
    assert FEATURE_COLUMNS == ["idade_anos", "sexo", "racacor", "escolaridade", "uf"]
    assert TARGET_COLUMN == "capitulo_cid10"


def test_feature_columns_nao_vazam_o_rotulo():
    proibidas = {"codmun6", "ano_arquivo", "capitulo_cid10"}
    assert proibidas.isdisjoint(FEATURE_COLUMNS)
    assert TARGET_COLUMN not in FEATURE_COLUMNS
    for nome in FEATURE_COLUMNS:
        maiusculo = nome.upper()
        assert not maiusculo.startswith("LINHA")
        assert maiusculo not in {"CIRCOBITO", "CAUSABAS", "CAUSABAS_O"}
    # todas existem no frame de modelagem
    assert set(FEATURE_COLUMNS + [TARGET_COLUMN]) <= set(sim_utils.MODEL_COLUMNS)
