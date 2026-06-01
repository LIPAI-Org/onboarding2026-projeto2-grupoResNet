""" Para poder ter os caminhos em variáveis """

import os

PATH_RAIZ = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

# DIRETÓRIOS PRINCIPAIS

PATH_CONFIGS = os.path.join(
    PATH_RAIZ,
    "configs"
)

PATH_DATA = os.path.join(
    PATH_RAIZ,
    "data"
)

PATH_SRC = os.path.join(
    PATH_RAIZ,
    "src"
)

PATH_RESULTS = os.path.join(
    PATH_RAIZ,
    "results"
)


# DATASETS E SPLITS

PATH_SPLITS = os.path.join(
    PATH_DATA,
    "splits"
)

PATH_SPLIT_DISPLASIA = os.path.join(
    PATH_SPLITS,
    "manifest_split_multiclass_NDB-UFES.csv"
)

PATH_SPLIT_NDB = os.path.join(
    PATH_SPLITS,
    "manifest_split_oralepitheliumdb.csv"
)


# RESULTADOS

PATH_RESULTADOS_DIR = PATH_RESULTS

PATH_PLANILHA_RESULTADOS = os.path.join(
    PATH_RESULTS,
    "planilha_resultados.csv"
)

PATH_TABELAS = os.path.join(
    PATH_RESULTS,
    "tabelas"
)

PATH_RESUMO_GLOBAL = os.path.join(
    PATH_TABELAS,
    "resumo_global.csv"
)


# FIGURAS E PLOTS

PATH_FIGURES = os.path.join(
    PATH_RESULTS,
    "figures"
)


PATH_PLOTS_GLOBAIS = os.path.join(
    PATH_FIGURES,
    "globais"
)

PATH_TABELAS_GLOBAIS = os.path.join(
    PATH_FIGURES,
    "tabelas_globais"
)

PATH_MATRIZES_CONFUSAO = os.path.join(
    PATH_FIGURES,
    "matrizes_confusao"
)

PATH_CURVAS_APRENDIZADO = os.path.join(
    PATH_FIGURES,
    "curvas_aprendizado"
)

# CSVs E RELATÓRIOS

PATH_CSVS = os.path.join(
    PATH_RESULTS,
    "csvs"
)

PATH_RELATORIOS = os.path.join(
    PATH_RESULTS,
    "relatorios"
)
