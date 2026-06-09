"Main do projeto"

from __future__ import annotations
from os.path import join

import configs.datasets.displasia as disp
import configs.datasets.ndb_ufes as ndb
from src.utils.rodar_experimentos import (
    rodar_todos_experimentos,
    rodar_experimentos_baseado_em_parametros,
)

from src.utils.paths import (
    PATH_PLANILHA_RESULTADOS,
    PATH_PLOTS_GLOBAIS,
    PATH_RESUMO_GLOBAL,
    PATH_TABELAS_GLOBAIS,
    PATH_CHECKPOINTS,
)

from src.analise.visualizar_experimentos_csv import (
    carregar_resultados,
    aplicar_filtros,
    ordenar_resultados,
    mostrar_resultados,
    mostrar_melhor_resultado,
    mostrar_resumo,
    mostrar_top_resultados,
    listar_valores_unicos,
)

from src.analise.plots_globais import (
    generate_reports,
)

from src.data.datasets import datasets
from src.utils.checkpoints import load_checkpoint
from src.modelos.modelo_factory import get_model
from configs.gradcam import gerar_pdfs_gradcam


def limpar_texto(valor):
    if valor is None:
        return None

    valor = str(valor).strip()

    if valor == "":
        return None

    return valor


def pedir_filtros():
    print()
    print("DEIXE VAZIO PARA IGNORAR O FILTRO")
    print()

    dataset = limpar_texto(
        input("Dataset: ")
    )

    model = limpar_texto(
        input("Modelo: ")
    )

    training_mode = limpar_texto(
        input("Training mode: ")
    )

    augmentation = limpar_texto(
        input("Augmentation (True/False): ")
    )

    seed = limpar_texto(
        input("Seed: ")
    )

    repetition = limpar_texto(
        input("Repetition: ")
    )

    if seed is not None:
        seed = int(seed)

    if repetition is not None:
        repetition = int(repetition)

    return {
        "dataset": dataset,
        "model": model,
        "training_mode": training_mode,
        "augmentation": augmentation,
        "seed": seed,
        "repetition": repetition,
    }


def menu():
    print()
    print("=" * 50)
    print("SISTEMA DE EXPERIMENTOS")
    print("=" * 50)
    print()
    print("1 - Mostrar todos os resultados")
    print("2 - Filtrar resultados")
    print("3 - Mostrar melhor resultado")
    print("4 - Mostrar top resultados")
    print("5 - Mostrar resumo")
    print("6 - Listar valores únicos")
    print("7 - Gerar gráficos globais")
    print("8 - Rodar TODOS os experimentos")
    print("9 - Rodar experimentos filtrados")
    print("10 - Produzir os Grad-CAMs")
    print("11 - Sair")
    print()


def executar():
    while True:
        try:
            df = carregar_resultados(
                PATH_PLANILHA_RESULTADOS
            )

        except (ValueError, FileNotFoundError) as e:
            print()
            print("ERRO AO CARREGAR CSV")
            print(e)
            print()
            return

        menu()

        opcao = input(
            "Escolha uma opção: "
        ).strip()

        if opcao == "1":
            resultado = ordenar_resultados(
                df,
                metric="f1_macro_test",
                ascending=False,
            )

            mostrar_resultados(
                resultado
            )

        elif opcao in ["2", "3", "4", "5", "9"]:
            filtros = pedir_filtros()

            resultado = aplicar_filtros(
                df=df,
                dataset=filtros["dataset"],
                model=filtros["model"],
                training_mode=filtros["training_mode"],
                augmentation=filtros["augmentation"],
                seed=filtros["seed"],
                repetition=filtros["repetition"],
            )

            if opcao == "2":
                resultado = ordenar_resultados(
                    resultado,
                    metric="f1_macro_test",
                    ascending=False,
                )

                mostrar_resultados(
                    resultado
                )

            elif opcao == "3":
                mostrar_melhor_resultado(
                    resultado,
                    metric="f1_macro_test",
                )

            elif opcao == "4":
                top_n = input(
                    "Quantidade do top: "
                ).strip()

                if top_n == "":
                    top_n = 10

                mostrar_top_resultados(
                    resultado,
                    top_n=int(top_n),
                    metric="f1_macro_test",
                )

            elif opcao == "5":
                mostrar_resumo(
                    resultado
                )

            elif opcao == "9":
                print()
                print("RODANDO EXPERIMENTOS FILTRADOS...")
                print()

                rodar_experimentos_baseado_em_parametros(
                    seed=filtros["seed"],
                    modelo=filtros["model"],
                    modo_treinamento=filtros["training_mode"],
                    aumento=filtros["augmentation"],
                    dataset=filtros["dataset"],
                )

                print()
                print("EXPERIMENTOS FINALIZADOS")
                print()

        elif opcao == "6":
            listar_valores_unicos(
                df
            )

        elif opcao == "7":
            print()
            print("GERANDO GRÁFICOS GLOBAIS...")
            print()

            generate_reports(
                csv_path=PATH_PLANILHA_RESULTADOS,
                output_dir=PATH_PLOTS_GLOBAIS,
                summary_csv=PATH_RESUMO_GLOBAL,
                table_pdf_dir=PATH_TABELAS_GLOBAIS,
            )

            print()
            print("GRÁFICOS GERADOS COM SUCESSO")
            print()

        elif opcao == "8":
            print()
            print("RODANDO TODOS OS EXPERIMENTOS...")
            print()

            rodar_todos_experimentos()

            print()
            print("TODOS OS EXPERIMENTOS FORAM EXECUTADOS")
            print()

        elif opcao == "10":
            bufferROI = datasets(config=disp.DATASET_CONFIG,
                                 escolha_transformada='base')
            bufferNDB = datasets(config= ndb.DATASET_CONFIG,
                                 escolha_transformada='base')
            _, _, test_ROI = bufferROI.carregar_dados_ROI_base()
            _, _, test_NDB = bufferNDB.carregar_dados_NDB_base()
            datasetes = [
                {"dataset": test_ROI, "config": disp.DATASET_CONFIG},
                {"dataset": test_NDB, "config": ndb.DATASET_CONFIG}
            ]
            
            r18roi = get_model(
                model_name="resnet18",
                num_classes=disp.DATASET_CONFIG.nro_classes,
                training_mode="fs"
            )
            r34roi = get_model(
                model_name="resnet34",
                num_classes=disp.DATASET_CONFIG.nro_classes,
                training_mode="fs"
            )
            r18ndb = get_model(
                model_name="resnet18",
                num_classes=ndb.DATASET_CONFIG.nro_classes,
                training_mode="fs"
            )
            r34ndb = get_model(
                model_name="resnet34",
                num_classes=ndb.DATASET_CONFIG.nro_classes,
                training_mode="fs"
            )
            a = load_checkpoint(
                path= join(PATH_CHECKPOINTS, "displasia/melhor_resnet18.pth"),
                modelo=r18roi
            )
            b = load_checkpoint(
                path= join(PATH_CHECKPOINTS, "displasia/melhor_resnet34.pth"),
                modelo=r34roi
            )
            c = load_checkpoint(
                path= join(PATH_CHECKPOINTS, "ndb/melhor_resnet18.pth"),
                modelo=r18ndb
            )
            d = load_checkpoint(
                path= join(PATH_CHECKPOINTS, "ndb/melhor_resnet34.pth"),
                modelo=r34ndb
            )
            melhor_roi = r18roi if a["best_val_acc"] > b["best_val_acc"] else r34roi
            melhor_ndb = r18ndb if c["best_val_acc"] > d["best_val_acc"] else r34ndb
            modelos = [
                {
                 "nome": "Melhor ROI",
                 "modelo": melhor_roi,
                 "layer_alvo": melhor_roi.layer4[-1],
                 "config": disp.DATASET_CONFIG
                },
                {
                 "nome": "Melhor NDB",
                 "modelo": melhor_ndb,
                 "layer_alvo": melhor_ndb.layer4[-1],
                 "config": ndb.DATASET_CONFIG
                }
            ]

            gerar_pdfs_gradcam(
                datasets= [datasetes[0]],
                modelos= [modelos[0]],
                n_por_classe=2
            )

            gerar_pdfs_gradcam(
                datasets= [datasetes[1]],
                modelos= [modelos[1]],
                n_por_classe=2
            )

        elif opcao == "11":
            print()
            print("ENCERRANDO...")
            print()
            break

        else:
            print()
            print("OPÇÃO INVÁLIDA")
            print()


if __name__ == "__main__":
    executar()