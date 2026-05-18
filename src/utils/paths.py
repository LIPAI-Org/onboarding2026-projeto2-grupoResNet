""" Para poder ter os caminhos em variáveis """

import os

PATH_RAIZ = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

# Paths gerais, para caso de usar um path específico
PATH_CONFIGS = PATH_RAIZ + "\\configs"
PATH_DATA = PATH_RAIZ + "\\data"
PATH_SRC = PATH_RAIZ + "\\src"
PATH_RESULTS = PATH_RAIZ + "\\results"

# Paths mais específicos (+ usados)
PATH_PLANILHA_RESULTADOS = PATH_RESULTS + "\\planilha_resultados.csv"
# FAZER: Colocar os paths dos splits de treino, teste e validação de cada dataset
# Analisar, ao longo do desenvolvimento, outros paths necessários, se for usar mt coloquem aqui!
