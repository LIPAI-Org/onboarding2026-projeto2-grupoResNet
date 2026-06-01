"Main do projeto"

from __future__ import annotations

from src.utils.rodar_experimentos import (
    rodar_todos_experimentos,
    rodar_experimentos_baseado_em_parametros,
)

from src.utils.paths import (
    PATH_PLANILHA_RESULTADOS,
    PATH_PLOTS_GLOBAIS,
    PATH_RESUMO_GLOBAL,
    PATH_TABELAS_GLOBAIS,
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
    print("10 - Sair")
    print()


def executar():
    while True:
        try:
            df = carregar_resultados(
                PATH_PLANILHA_RESULTADOS
            )

        except Exception as e:
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