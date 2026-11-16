# =============================================================================
# Integrantes: Arthur Meneses Neves, Danilo Oliveira Santos, Matheus Gabriel
# Viana Araujo, João Victor Vidal Barbosa, Guilherme Araujo Castro
#
# Síntese: Funções de avaliação da N2 (notebook 04): amostragem estratificada
# com piso por classe, busca aleatória de hiperparâmetros com orçamento igual
# para os modelos, baseline por tabela (faixa etária x sexo), F1-macro a
# partir da matriz de confusão, intervalos de confiança por bootstrap pareado,
# métricas por subgrupo com supressão de células pequenas e importância por
# permutação (XAI) sobre as colunas originais.
#
# Changelog:
#   2026-11-16 - Matheus Araujo - criação
# =============================================================================
import itertools
import time

import numpy as np
import pandas as pd

# Células com menos casos que isto têm as métricas suprimidas nas tabelas
# por subgrupo (minimização da LGPD; ver artigo, seção de ética).
LIMITE_SUPRESSAO = 10

# Faixas etárias usadas na análise por subgrupo (limites inferiores).
FAIXAS_SUBGRUPO = [0, 1, 15, 30, 45, 60, 75]
ROTULOS_SUBGRUPO = ["<1", "1-14", "15-29", "30-44", "45-59", "60-74", "75+"]


def faixa_etaria(idade, limites=FAIXAS_SUBGRUPO, rotulos=ROTULOS_SUBGRUPO):
    """Categoriza a idade em anos pelas faixas `limites` (limite inferior de
    cada faixa; a última é aberta)."""
    idade = np.asarray(idade, dtype=float)
    pos = np.searchsorted(np.asarray(limites, dtype=float), idade, side="right") - 1
    return pd.Categorical.from_codes(pos.astype(np.int8), categories=list(rotulos))


def faixa_quinquenal(idade):
    """Faixa de 5 anos (0-4, 5-9, ..., 95+), usada pelo baseline por tabela."""
    limites = list(range(0, 100, 5))
    rotulos = [f"{a}-{a + 4}" for a in limites[:-1]] + ["95+"]
    return faixa_etaria(idade, limites, rotulos)


# -----------------------------------------------------------------------------
# Amostragem
# -----------------------------------------------------------------------------


def amostra_estratificada(y, n, seed, minimo_por_classe=50, excluir=None):
    """Índices de uma amostra de cerca de `n` linhas, proporcional por classe,
    mas com pelo menos `minimo_por_classe` linhas de cada classe (ou todas,
    se a classe tiver menos). Linhas em `excluir` (índices) não são
    sorteadas. Devolve os índices ordenados."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    disponivel = np.ones(len(y), dtype=bool)
    if excluir is not None:
        disponivel[np.asarray(excluir)] = False
    fracao = min(1.0, n / max(1, disponivel.sum()))
    escolhidos = []
    for classe in np.unique(y[disponivel]):
        idx = np.flatnonzero((y == classe) & disponivel)
        alvo = max(int(round(fracao * len(idx))), minimo_por_classe)
        alvo = min(alvo, len(idx))
        escolhidos.append(rng.choice(idx, size=alvo, replace=False))
    return np.sort(np.concatenate(escolhidos))


# -----------------------------------------------------------------------------
# Métricas a partir da matriz de confusão
# -----------------------------------------------------------------------------


def matriz_confusao(y, pred, k):
    """Matriz k x k (linhas = verdadeiro, colunas = previsto) via bincount."""
    y = np.asarray(y, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    return np.bincount(y * k + pred, minlength=k * k).reshape(k, k)


def metricas_de_confusao(cm):
    """F1-macro, acurácia e recall macro a partir da matriz de confusão.

    Segue a convenção do scikit-learn com zero_division=0 e todas as k
    classes como rótulos: classe sem suporte e sem predição conta como F1 0.
    """
    cm = np.asarray(cm, dtype=float)
    tp = np.diag(cm)
    suporte = cm.sum(axis=1)
    previstos = cm.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        precisao = np.where(previstos > 0, tp / previstos, 0.0)
        recall = np.where(suporte > 0, tp / suporte, 0.0)
        f1 = np.where(precisao + recall > 0, 2 * precisao * recall / (precisao + recall), 0.0)
    total = cm.sum()
    return {
        "f1_macro": float(f1.mean()),
        "acuracia": float(tp.sum() / total) if total else 0.0,
        "recall_macro": float(recall.mean()),
        "f1_por_classe": f1,
        "recall_por_classe": recall,
        "precisao_por_classe": precisao,
    }


def f1_macro(y, pred, k):
    return metricas_de_confusao(matriz_confusao(y, pred, k))["f1_macro"]


# -----------------------------------------------------------------------------
# Intervalos de confiança: bootstrap pareado sobre o conjunto de teste
# -----------------------------------------------------------------------------


def bootstrap_f1_macro(y, preds, k, n_boot=200, seed=0):
    """Reamostra as linhas do teste com reposição `n_boot` vezes e calcula o
    F1-macro de cada modelo de `preds` ({nome: predições}) nas MESMAS linhas
    de cada réplica (bootstrap pareado). Devolve um DataFrame n_boot x
    modelos; a diferença entre colunas dá o IC da diferença entre modelos.

    Mede só a incerteza do conjunto de teste, não a do treino (para isso o
    notebook repete o treino com sementes diferentes).
    """
    y = np.asarray(y, dtype=np.int64)
    rng = np.random.default_rng(seed)
    n = len(y)
    # Contar quantas vezes cada linha foi sorteada (pesos) equivale a sortear
    # n índices com reposição e permite montar a matriz de confusão por
    # bincount ponderado, sem copiar as predições.
    codigos = {nome: y * k + np.asarray(p, dtype=np.int64) for nome, p in preds.items()}
    linhas = []
    for _ in range(n_boot):
        pesos = np.bincount(rng.integers(0, n, size=n), minlength=n)
        linha = {}
        for nome, cod in codigos.items():
            cm = np.bincount(cod, weights=pesos, minlength=k * k).reshape(k, k)
            linha[nome] = metricas_de_confusao(cm)["f1_macro"]
        linhas.append(linha)
    return pd.DataFrame(linhas)


def resumo_ic(amostras, alpha=0.05):
    """Média e intervalo percentil (1 - alpha) de cada coluna."""
    amostras = pd.DataFrame(amostras)
    return pd.DataFrame(
        {
            "media": amostras.mean(),
            "ic_inf": amostras.quantile(alpha / 2),
            "ic_sup": amostras.quantile(1 - alpha / 2),
        }
    )


def diferencas_pareadas(amostras, pares):
    """Para cada par (a, b), amostras de F1(a) - F1(b) do bootstrap pareado."""
    return pd.DataFrame({f"{a} - {b}": amostras[a] - amostras[b] for a, b in pares})


# -----------------------------------------------------------------------------
# Subgrupos e supressão de células pequenas
# -----------------------------------------------------------------------------


def suprimir_celulas_pequenas(tabela, col_n="n", colunas=None, limite=LIMITE_SUPRESSAO):
    """Copia `tabela` trocando por NaN as `colunas` (padrão: todas menos
    `col_n`) das linhas com `col_n` < `limite`, e marca essas linhas na
    coluna "suprimido". A contagem também é escondida (vira NaN)."""
    out = tabela.copy()
    pequenas = out[col_n] < limite
    colunas = [c for c in out.columns if c != col_n] if colunas is None else list(colunas)
    out[colunas] = out[colunas].astype(float)
    out.loc[pequenas, colunas] = np.nan
    out[col_n] = out[col_n].astype(float)
    out.loc[pequenas, col_n] = np.nan
    out["suprimido"] = pequenas
    return out


def metricas_por_subgrupo(y, pred, grupo, k, limite=LIMITE_SUPRESSAO):
    """F1-macro, acurácia e recall macro por valor de `grupo`, com supressão
    das células com menos de `limite` linhas. O F1-macro de cada subgrupo é
    calculado sobre as classes presentes nele (verdadeiras ou previstas):
    uma classe que não ocorre em um subgrupo não deve puxar a média para
    baixo (por exemplo, gravidez entre homens)."""
    y = np.asarray(y, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    # Se `grupo` for categórico, a tabela segue a ordem das categorias (faixas
    # etárias em ordem crescente, por exemplo); senão, ordem alfabética.
    ordem = list(grupo.categories) if isinstance(grupo, pd.Categorical) else None
    grupo = pd.Series(np.asarray(grupo, dtype=object))
    linhas = []
    for valor, idx in grupo.groupby(grupo, sort=True).indices.items():
        cm = matriz_confusao(y[idx], pred[idx], k)
        presentes = (cm.sum(axis=0) + cm.sum(axis=1)) > 0
        m = metricas_de_confusao(cm[np.ix_(presentes, presentes)])
        linhas.append(
            {
                "grupo": valor,
                "n": len(idx),
                "classes_presentes": int(presentes.sum()),
                "f1_macro": m["f1_macro"],
                "acuracia": m["acuracia"],
                "recall_macro": m["recall_macro"],
            }
        )
    tabela = pd.DataFrame(linhas).set_index("grupo")
    if ordem is not None:
        tabela = tabela.loc[[g for g in ordem if g in tabela.index]]
    return suprimir_celulas_pequenas(tabela, limite=limite)


# -----------------------------------------------------------------------------
# Baseline por tabela (faixa etária x sexo)
# -----------------------------------------------------------------------------


class BaselineTabela:
    """Baseline sem aprendizado de máquina: para cada célula das colunas
    `chaves` (por padrão faixa de 5 anos x sexo), prevê a classe com a maior
    frequência ponderada pelos mesmos pesos balanceados dos modelos
    (n / (k * n_classe)). Células não vistas no treino recebem a classe mais
    frequente do treino.

    Serve para responder se os modelos aprendem algo além de "uma tabela de
    idade e sexo", que é o que a N1 sugeria.
    """

    def __init__(self, chaves=("faixa5", "sexo")):
        self.chaves = list(chaves)

    def _celulas(self, X):
        partes = []
        for chave in self.chaves:
            if chave == "faixa5":
                partes.append(pd.Series(faixa_quinquenal(X["idade_anos"]).astype(str), index=X.index))
            else:
                partes.append(X[chave].astype(str))
        return pd.concat(partes, axis=1).agg("|".join, axis=1) if len(partes) > 1 else partes[0]

    def fit(self, X, y):
        y = np.asarray(y)
        classes, contagem = np.unique(y, return_counts=True)
        peso = pd.Series(len(y) / (len(classes) * contagem), index=classes)
        tabela = (
            pd.DataFrame({"celula": self._celulas(X).to_numpy(), "y": y})
            .groupby(["celula", "y"])
            .size()
            .unstack(fill_value=0)
        )
        ponderada = tabela * peso.reindex(tabela.columns).to_numpy()
        self.regra_ = ponderada.idxmax(axis=1)
        self.padrao_ = classes[np.argmax(contagem)]
        return self

    def predict(self, X):
        celulas = self._celulas(X)
        pred = celulas.map(self.regra_)
        return pred.fillna(self.padrao_).to_numpy().astype(np.asarray(self.padrao_).dtype)


# -----------------------------------------------------------------------------
# Busca aleatória de hiperparâmetros
# -----------------------------------------------------------------------------


def sortear_configuracoes(espaco, n_iter, seed):
    """Sorteia até `n_iter` combinações distintas do produto cartesiano de
    `espaco` ({parâmetro: lista de valores}), sem repetição."""
    nomes = list(espaco)
    todas = list(itertools.product(*(espaco[n] for n in nomes)))
    rng = np.random.default_rng(seed)
    escolha = rng.choice(len(todas), size=min(n_iter, len(todas)), replace=False)
    return [dict(zip(nomes, todas[i])) for i in escolha]


def busca_aleatoria(construir, espaco, n_iter, seed, X_tr, y_tr, w_tr, X_val, y_val, k, log=print):
    """Treina `construir(**params)` para cada configuração sorteada (pesos de
    amostra `w_tr`) e mede o F1-macro na validação. O teste nunca é usado
    aqui. Devolve um DataFrame ordenado do melhor para o pior, com os
    parâmetros, o F1-macro de validação e o tempo de treino."""
    linhas = []
    for i, params in enumerate(sortear_configuracoes(espaco, n_iter, seed), start=1):
        t0 = time.perf_counter()
        modelo = construir(**params)
        modelo.fit(X_tr, y_tr, sample_weight=w_tr)
        tempo = time.perf_counter() - t0
        f1 = f1_macro(y_val, modelo.predict(X_val), k)
        linhas.append({**params, "f1_macro_val": f1, "tempo_treino_s": tempo})
        if log:
            log(f"[{i:2d}/{n_iter}] F1-macro val = {f1:.4f} | {tempo:6.1f} s | {params}")
    return pd.DataFrame(linhas).sort_values("f1_macro_val", ascending=False).reset_index(drop=True)


def melhores_parametros(resultado_busca, espaco):
    """Parâmetros da primeira linha (a melhor) de `busca_aleatoria`, com os
    tipos originais do espaço (o DataFrame transforma None em NaN)."""
    linha = resultado_busca.iloc[0]
    params = {}
    for nome, valores in espaco.items():
        bruto = linha[nome]
        if isinstance(bruto, float) and np.isnan(bruto):
            params[nome] = None
            continue
        for v in valores:
            if v is not None and (v == bruto or (isinstance(v, float) and np.isclose(v, bruto))):
                params[nome] = v
                break
        else:
            params[nome] = bruto
    return params


# -----------------------------------------------------------------------------
# Importância por permutação (XAI)
# -----------------------------------------------------------------------------


def importancia_por_permutacao(prever, X, y, k, colunas=None, n_rep=3, seed=0):
    """Queda de F1-macro quando cada coluna ORIGINAL de `X` (antes do
    one-hot) é embaralhada. `prever(X_df)` devolve as classes previstas.
    Embaralhar a coluna original mede a importância do atributo inteiro, e
    não de cada coluna do one-hot separada."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    base = f1_macro(y, prever(X), k)
    colunas = list(X.columns) if colunas is None else list(colunas)
    linhas = []
    for coluna in colunas:
        quedas = []
        for _ in range(n_rep):
            X_perm = X.copy()
            X_perm[coluna] = X[coluna].to_numpy()[rng.permutation(len(X))]
            quedas.append(base - f1_macro(y, prever(X_perm), k))
        linhas.append(
            {"atributo": coluna, "queda_f1_media": float(np.mean(quedas)), "queda_f1_dp": float(np.std(quedas))}
        )
    tabela = pd.DataFrame(linhas).sort_values("queda_f1_media", ascending=False).reset_index(drop=True)
    tabela.attrs["f1_base"] = base
    return tabela
