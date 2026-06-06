""" Lida com a lógica do grad-cam """

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from configs.configs_base import DEVICE
from src.utils.seed import definir_seed
from src.utils.paths import PATH_GRAD_CAM

class GradCAM:
    def __init__(self, modelo, layer_alvo):
        self.modelo = modelo.eval()
        self.layer_alvo = layer_alvo
        self.ativacoes = None
        self.gradientes = None

        self.gancho_fwd = layer_alvo.register_forward_hook(self.salvar_ativacoes)
        self.gancho_bwd = layer_alvo.register_full_backward_hook(self.salvar_gradientes)

    # AMBAS PRECISAM DOS PARÂMETROS NÃO UTILIZADOS! NÂO MEXER...
    def salvar_ativacoes(self, module, inputs, output):
        self.ativacoes = output.detach()

    def salvar_gradientes(self, module, grad_input, grad_output):
        self.gradientes = grad_output[0].detach()

    def remover_ganchos(self):
        self.gancho_fwd.remove()
        self.gancho_bwd.remove()

    def __call__(self, x, class_idx=None):
        self.modelo.zero_grad()
        logits = self.modelo(x)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        score = logits[0, class_idx]
        score.backward()

        # Pesos = média dos gradientes na dimensão espacial
        pesos = self.gradientes.mean(dim=(2, 3), keepdim=True)

        # Combina pesos com ativações
        cam = (pesos * self.ativacoes).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normaliza para [0, 1]
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.cpu(), class_idx
    
def _extrair_entrada_modelo(item, indice):
    if isinstance(item, dict):
        modelo = item.get("modelo", item.get("model"))
        camada_alvo = item.get("layer_alvo", item.get("camada_alvo"))
        config = item.get("config")
        nome = item.get("nome", item.get("name", f"modelo_{indice}"))
        return nome, modelo, camada_alvo, config

    if isinstance(item, (tuple, list)):
        if len(item) == 3:
            modelo, camada_alvo, config = item
            nome = f"modelo_{indice}"
            return nome, modelo, camada_alvo, config
        if len(item) == 4:
            nome, modelo, camada_alvo, config = item
            return nome, modelo, camada_alvo, config

    raise TypeError(
        "Cada item de 'modelos' deve ser dict com chaves "
        "'modelo/model', 'layer_alvo/camada_alvo' e 'config', "
        "ou tupla/lista no formato (modelo, camada_alvo, config) "
        "ou (nome, modelo, camada_alvo, config)."
    )


def _amostras_por_classe(dataset, n_por_classe, seed=42):
    if not hasattr(dataset, "data"):
        raise AttributeError("O dataset precisa ter o atributo 'data'.")

    col_label = getattr(dataset, "col_label", None)
    if col_label is None:
        raise AttributeError("Não encontrei 'col_label' no dataset.")

    definicoes = {}
    for idx, rotulo in enumerate(dataset.data[col_label].tolist()):
        rotulo = int(rotulo)
        definicoes.setdefault(rotulo, []).append(idx)

    g = torch.Generator()
    g.manual_seed(seed)

    saida = {}
    for rotulo, indices in definicoes.items():
        if len(indices) <= n_por_classe:
            saida[rotulo] = list(indices)
        else:
            perm = torch.randperm(len(indices), generator=g).tolist()
            saida[rotulo] = [indices[i] for i in perm[:n_por_classe]]

    return saida


def _carregar_imagem_original(dataset, idx):
    col_path = getattr(dataset, "col_path", None)
    if col_path is None:
        raise AttributeError("Não encontrei 'col_path' no dataset.")

    caminho_relativo = dataset.data.iloc[idx][col_path]
    caminho_absoluto = Path(dataset.root_dir) / caminho_relativo

    with Image.open(caminho_absoluto) as img:
        img = img.convert("RGB")
        return img, str(caminho_absoluto)


def gerar_pdfs_gradcam(
    datasets,
    modelos,
    n_por_classe,
    seed=42,
):
    definir_seed(seed)

    pasta_saida = Path(PATH_GRAD_CAM)
    pasta_saida.mkdir(parents=True, exist_ok=True)

    for _, item_dataset in enumerate(datasets):
        dataset = item_dataset["dataset"]
        config_dataset = item_dataset["config"]

        nome_dataset = config_dataset.nome
        classes = config_dataset.labels_classes

        amostras = _amostras_por_classe(dataset, n_por_classe=n_por_classe, seed=seed)

        for idx_modelo, item_modelo in enumerate(modelos):
            nome_modelo, modelo, camada_alvo, _ = _extrair_entrada_modelo(item_modelo, idx_modelo)

            modelo = modelo.to(DEVICE)
            modelo.eval()

            gradcam = GradCAM(modelo, camada_alvo)

            try:
                for rotulo, indices in amostras.items():
                    nome_classe = classes[rotulo] if rotulo < len(classes) else str(rotulo)
                    pdf_saida = (
                        pasta_saida
                        / f"gradcam_{nome_dataset}_classe_{nome_classe}_modelo_{nome_modelo}.pdf"
                    )

                    with PdfPages(pdf_saida) as pdf:
                        for idx_amostra in indices:
                            imagem_original, caminho_img = _carregar_imagem_original(dataset, idx_amostra)

                            x, _ = dataset[idx_amostra]
                            if not torch.is_tensor(x):
                                raise TypeError(
                                    f"O dataset retornou {type(x)} em idx={idx_amostra}, "
                                    "mas era esperado torch.Tensor."
                                )

                            x = x.unsqueeze(0).to(DEVICE)

                            with torch.no_grad():
                                logits_pre = modelo(x)
                            if config_dataset.is_binario:
                                if logits_pre.shape[1] == 1:
                                    probas = torch.sigmoid(logits_pre)[0, 0]
                                    classe_prevista = int((probas >= 0.5).item())
                                    certeza = float(probas.item() if classe_prevista == 1 else (1.0 - probas.item()))
                                else:
                                    probas = torch.softmax(logits_pre, dim=1)[0]
                                    classe_prevista = int(torch.argmax(probas).item())
                                    certeza = float(probas[classe_prevista].item())
                            else:
                                probas = torch.softmax(logits_pre, dim=1)[0]
                                classe_prevista = int(torch.argmax(probas).item())
                                certeza = float(probas[classe_prevista].item())

                            cam, classe_gradcam = gradcam(x, class_idx=classe_prevista)
                            y_predito = classes[classe_gradcam] if classe_gradcam < len(classes) else str(classe_gradcam)

                            fig, eixos = plt.subplots(1, 3, figsize=(15, 5))

                            eixos[0].imshow(imagem_original)
                            eixos[0].set_title("imagem original")
                            eixos[0].axis("off")

                            eixos[1].imshow(cam.numpy(), cmap="jet")
                            eixos[1].set_title("Grad-CAM")
                            eixos[1].axis("off")

                            eixos[2].imshow(imagem_original)
                            eixos[2].imshow(cam.numpy(), cmap="jet", alpha=0.45)
                            eixos[2].set_title(f"Grad-CAM: {y_predito} {certeza:.2%}")
                            eixos[2].axis("off")

                            fig.suptitle(
                                f"{nome_dataset} | classe real: {nome_classe} | arquivo: {Path(caminho_img).name}",
                                fontsize=10
                            )
                            plt.tight_layout()
                            pdf.savefig(fig, bbox_inches="tight")
                            plt.close(fig)

            finally:
                gradcam.remover_ganchos()
