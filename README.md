# Projeto 2 - Grupo 1 - ResNet18 e ResNet34

Este projeto tem como objetivo realizar experimentos de classificação de imagens médicas utilizando arquiteturas ResNet com diferentes estratégias de treinamento, datasets e configurações de data augmentation.

O sistema permite:

* Treinar modelos automaticamente
* Rodar grids completos de experimentos
* Avaliar métricas de desempenho
* Salvar resultados em CSV
* Gerar gráficos globais
* Visualizar resultados de forma interativa

---

# Estrutura do Projeto



---

# Funcionalidades

## Treinamento de Modelos

O projeto suporta treinamento utilizando:

* ResNet18
* ResNet34

Modos de treinamento:

* `fs` → treinamento do zero
* `pt_fc` → fine tuning apenas da camada final
* `pt_all` → fine tuning completo

---

# Datasets

Atualmente o projeto suporta:

* Displasia
* NDB (Oral Epithelial Dysplasia Database)

Os manifests dos datasets ficam em:

```bash
data/splits/
```

---

# Execução do Projeto

## Criar ambiente virtual

### Windows

```bash
python -m venv .venv
```

Ativar:

```bash
.venv\Scripts\activate
```

---

### Linux / macOS

Criar ambiente virtual:

```bash
python3 -m venv .venv
```

Ativar:

```bash
source .venv/bin/activate
```

---

# Instalação das Dependências

```bash
pip install torch torchvision pandas matplotlib scikit-learn tqdm thop ptflops seaborn
```

---

# Executar o Sistema

```bash
python main.py
```

---

# Menu Interativo

O sistema possui um menu interativo que permite:

* Rodar todos os experimentos
* Rodar experimentos filtrados
* Visualizar resultados
* Filtrar experimentos
* Mostrar melhores resultados
* Gerar gráficos globais
* Gerar tabelas resumo

---

# Resultados

Os resultados dos experimentos são salvos em:

```bash
results/planilha_resultados.csv
```

Métricas armazenadas:

* Accuracy
* F1 Macro
* F1 Weighted
* Número de parâmetros
* GFLOPs
* Melhor época
* Melhor validação

---

# Visualização dos Resultados

O módulo:

```bash
src/analise/visualizar_experimentos_csv.py
```

permite:

* Filtrar experimentos
* Ordenar resultados
* Buscar melhores execuções
* Mostrar resumos estatísticos
* Comparar modelos
* Comparar datasets
* Comparar estratégias de treinamento

---

# Geração de Gráficos Globais

O sistema gera automaticamente:

* Gráficos de accuracy
* Gráficos de loss
* Comparações globais
* Tabelas resumo

Saídas:

```bash
results/figures/
results/tabelas/
```

---

# Organização dos Experimentos

Os experimentos são definidos em:

```bash
configs/grid_experimentos.py
```

O grid controla:

* Seeds
* Modelos
* Datasets
* Data augmentation
* Estratégias de treinamento

---

# Tecnologias Utilizadas

* Python
* PyTorch
* Pandas
* Matplotlib
* Scikit-learn
* THOP
* tqdm

---

# Autores

Gabriel Dos Santos Do Amaral
João Geiger Piza
Gabriel Lemes

---

# Observações

* O projeto detecta automaticamente CUDA caso disponível.
* Os resultados são persistidos em CSV.
* Os gráficos são salvos automaticamente após cada experimento.
* É possível expandir facilmente para novos modelos e datasets.
