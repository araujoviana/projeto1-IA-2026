# =============================================================================
# Integrantes: Arthur Meneses Neves, Danilo Oliveira Santos, Matheus Gabriel
# Viana Araujo, João Victor Vidal Barbosa, Guilherme Araujo Castro
#
# Síntese: Testes das funções de avaliação da N2 (avaliacao.py), com dados
# sintéticos: métricas conferidas contra o scikit-learn, bootstrap pareado,
# supressão de células pequenas, baseline por tabela, busca aleatória e
# importância por permutação.
#
# Changelog:
#   2026-11-16 - Matheus Araujo - criação
# =============================================================================
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.tree import DecisionTreeClassifier

import avaliacao as av


def _y_pred(n=5000, k=6, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.choice(k, size=n, p=[0.5, 0.2, 0.15, 0.1, 0.04, 0.01])
    pred = np.where(rng.random(n) < 0.6, y, rng.integers(0, k, size=n))
    return y, pred


def test_metricas_de_confusao_batem_com_sklearn():
    k = 6
    y, pred = _y_pred(k=k)
    pred[pred == 5] = 4  # classe 5 nunca prevista: zero_division entra em jogo
    m = av.metricas_de_confusao(av.matriz_confusao(y, pred, k))
    labels = list(range(k))
    assert m["f1_macro"] == pytest.approx(f1_score(y, pred, average="macro", labels=labels, zero_division=0))
    assert m["acuracia"] == pytest.approx(accuracy_score(y, pred))
    assert m["recall_macro"] == pytest.approx(recall_score(y, pred, average="macro", labels=labels, zero_division=0))


def test_f1_macro_conta_classe_ausente_como_zero_como_no_notebook_03():
    # Classe 2 sem suporte e sem predição: o sklearn com labels=range(k)
    # conta F1 0 para ela; manter a mesma convenção do notebook 03.
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 0, 1, 1])
    assert av.f1_macro(y, pred, 3) == pytest.approx(
        f1_score(y, pred, average="macro", labels=[0, 1, 2], zero_division=0)
    )


def test_bootstrap_pareado_contem_valor_observado_e_e_reprodutivel():
    k = 6
    y, pred_a = _y_pred(k=k, seed=1)
    _, pred_b = _y_pred(k=k, seed=2)
    amostras = av.bootstrap_f1_macro(y, {"a": pred_a, "b": pred_b}, k, n_boot=60, seed=3)
    assert amostras.shape == (60, 2)
    ic = av.resumo_ic(amostras)
    observado = av.f1_macro(y, pred_a, k)
    assert ic.loc["a", "ic_inf"] <= observado <= ic.loc["a", "ic_sup"]
    pd.testing.assert_frame_equal(
        amostras, av.bootstrap_f1_macro(y, {"a": pred_a, "b": pred_b}, k, n_boot=60, seed=3)
    )
    dif = av.diferencas_pareadas(amostras, [("a", "b")])
    assert list(dif.columns) == ["a - b"]
    np.testing.assert_allclose(dif["a - b"], amostras["a"] - amostras["b"])


def test_bootstrap_modelo_identico_tem_diferenca_zero():
    y, pred = _y_pred()
    amostras = av.bootstrap_f1_macro(y, {"a": pred, "b": pred.copy()}, 6, n_boot=20, seed=0)
    assert (av.diferencas_pareadas(amostras, [("a", "b")]).abs() < 1e-12).all().all()


def test_suprimir_celulas_pequenas():
    tab = pd.DataFrame({"n": [5, 10, 200], "f1_macro": [0.9, 0.5, 0.3]}, index=["x", "y", "z"])
    out = av.suprimir_celulas_pequenas(tab, limite=10)
    assert out["suprimido"].tolist() == [True, False, False]
    assert np.isnan(out.loc["x", "f1_macro"]) and np.isnan(out.loc["x", "n"])
    assert out.loc["y", "f1_macro"] == 0.5 and out.loc["z", "n"] == 200
    assert tab.loc["x", "f1_macro"] == 0.9  # não altera a original


def test_metricas_por_subgrupo_suprime_e_ignora_classes_ausentes():
    # grupo "M": só as classes 0 e 1 aparecem, previsão perfeita -> F1 1,0
    # (a classe 2, ausente no grupo, não deve contar como F1 zero).
    y = np.array([0, 1, 0, 1, 2, 2, 0, 1, 2, 0, 1] + [0] * 3)
    pred = np.array([0, 1, 0, 1, 2, 0, 0, 1, 2, 0, 1] + [1] * 3)
    grupo = ["M"] * 4 + ["F"] * 7 + ["raro"] * 3
    tab = av.metricas_por_subgrupo(y, pred, grupo, k=3, limite=4)
    assert tab.loc["M", "f1_macro"] == pytest.approx(1.0)
    assert tab.loc["M", "classes_presentes"] == 2
    assert tab.loc["raro", "suprimido"] and np.isnan(tab.loc["raro", "f1_macro"])
    assert tab.loc["F", "n"] == 7


def test_faixa_etaria_e_quinquenal():
    assert list(av.faixa_etaria([0, 0.5, 1, 14, 15, 74, 75, 110]).astype(str)) == [
        "<1", "<1", "1-14", "1-14", "15-29", "60-74", "75+", "75+",
    ]
    assert list(av.faixa_quinquenal([0, 4, 5, 94, 95, 120]).astype(str)) == [
        "0-4", "0-4", "5-9", "90-94", "95+", "95+",
    ]


def test_amostra_estratificada_respeita_piso_e_exclusao():
    y = np.array([0] * 10_000 + [1] * 1_000 + [2] * 20)
    idx = av.amostra_estratificada(y, n=1_100, seed=0, minimo_por_classe=50)
    contagem = np.bincount(y[idx], minlength=3)
    assert contagem[0] == pytest.approx(1000, abs=2)
    assert contagem[1] == 100
    assert contagem[2] == 20  # classe com menos que o piso entra inteira
    assert len(np.unique(idx)) == len(idx)
    outra = av.amostra_estratificada(y, n=1_100, seed=1, excluir=idx)
    assert np.intersect1d(idx, outra).size == 0


def test_baseline_tabela_aprende_regra_de_idade_e_sexo():
    X = pd.DataFrame(
        {
            "idade_anos": [0, 1, 2, 30, 31, 32, 70, 71, 72, 73],
            "sexo": ["M", "M", "F", "F", "F", "F", "M", "M", "M", "F"],
        }
    )
    y = np.array([5, 5, 5, 3, 3, 3, 1, 1, 1, 1])
    modelo = av.BaselineTabela().fit(X, y)
    teste = pd.DataFrame({"idade_anos": [3, 33, 74, 50], "sexo": ["M", "F", "M", "F"]})
    pred = modelo.predict(teste)
    assert pred[:3].tolist() == [5, 3, 1]
    assert pred[3] == 1  # célula 50-54|F nunca vista -> classe mais frequente
    assert pred.dtype.kind in "iu"


def test_baseline_tabela_usa_pesos_balanceados():
    # Na célula, 3 casos da classe 0 (frequente) e 2 da classe 1 (rara no
    # total): com pesos balanceados a classe 1 vence.
    X = pd.DataFrame({"idade_anos": [40] * 5 + [80] * 95, "sexo": ["M"] * 100})
    y = np.array([0, 0, 0, 1, 1] + [0] * 95)
    assert av.BaselineTabela().fit(X, y).predict(X.iloc[:1])[0] == 1


def test_sortear_configuracoes_sem_repeticao():
    espaco = {"a": [1, 2, 3], "b": [None, "x"]}
    confs = av.sortear_configuracoes(espaco, 4, seed=0)
    assert len(confs) == 4
    assert len({tuple(sorted(c.items(), key=str)) for c in confs}) == 4
    assert len(av.sortear_configuracoes(espaco, 100, seed=0)) == 6


def test_busca_aleatoria_e_melhores_parametros():
    rng = np.random.default_rng(0)
    X = rng.random((600, 3))
    y = (X[:, 0] > 0.5).astype(int) + (X[:, 1] > 0.5).astype(int)
    espaco = {"max_depth": [1, 4, None], "min_samples_leaf": [1, 50]}
    res = av.busca_aleatoria(
        lambda **p: DecisionTreeClassifier(random_state=0, **p), espaco, n_iter=6, seed=0,
        X_tr=X[:400], y_tr=y[:400], w_tr=np.ones(400), X_val=X[400:], y_val=y[400:], k=3, log=None,
    )
    assert len(res) == 6
    assert res["f1_macro_val"].is_monotonic_decreasing
    melhor = av.melhores_parametros(res, espaco)
    assert set(melhor) == set(espaco)
    assert melhor["max_depth"] in (4, None) and melhor["max_depth"] != 1
    assert all(isinstance(v, int) or v is None for v in melhor.values())


def test_importancia_por_permutacao_identifica_atributo_relevante():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"util": rng.integers(0, 3, 3000), "ruido": rng.integers(0, 3, 3000)})
    y = X["util"].to_numpy()
    tab = av.importancia_por_permutacao(lambda d: d["util"].to_numpy(), X, y, k=3, n_rep=2, seed=0)
    assert tab.iloc[0]["atributo"] == "util"
    assert tab.iloc[0]["queda_f1_media"] > 0.5
    assert tab.set_index("atributo").loc["ruido", "queda_f1_media"] == 0
    assert tab.attrs["f1_base"] == 1.0


def test_metricas_por_subgrupo_respeita_ordem_categorica():
    grupo = av.faixa_etaria([80, 0, 20, 0, 80, 20])
    tab = av.metricas_por_subgrupo(np.zeros(6, int), np.zeros(6, int), grupo, k=2, limite=1)
    assert list(tab.index) == ["<1", "15-29", "75+"]
