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

import numpy as np
import torch
import matplotlib.pyplot as plt


def _desnormalizar_tensor(x, media, desvio):
    media = torch.tensor(media, device=x.device).view(-1, 1, 1)
    desvio = torch.tensor(desvio, device=x.device).view(-1, 1, 1)
    return x * desvio + media


def _tensor_para_imagem(x, media, desvio):
    x = _desnormalizar_tensor(x, media, desvio)
    x = x.permute(1, 2, 0).detach().cpu().numpy()
    x = np.clip(x, 0.0, 1.0)
    return x


def _superpor_cam(imagem, cam, alpha=0.45):
    cmap = plt.get_cmap("jet")(cam)[..., :3]
    return np.clip((1 - alpha) * imagem + alpha * cmap, 0.0, 1.0)

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

        if logits.dim() != 2:
            raise ValueError(f"Saída inesperada do modelo: {logits.shape}")

        n_saidas = logits.size(1)

        if n_saidas == 1:
            score = logits[0, 0]
            classe_usada = 0
        else:
            if class_idx is None:
                class_idx = logits.argmax(dim=1).item()
            classe_usada = int(class_idx)
            score = logits[0, classe_usada]

        score.backward()

        pesos = self.gradientes.mean(dim=(2, 3), keepdim=True)
        cam = (pesos * self.ativacoes).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.cpu(), classe_usada
    
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
                    nome_dataset = str(config_dataset.nome).replace("/", "_").replace("\\", "_").replace(":", "_")
                    nome_classe = str(classes[rotulo]).replace("/", "_").replace("\\", "_").replace(":", "_")
                    nome_modelo = str(nome_modelo).replace("/", "_").replace("\\", "_").replace(":", "_")
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
                                if logits_pre.size(1) == 1:
                                    probas = torch.sigmoid(logits_pre)[0, 0].item()
                                    classe_prevista = 1 if probas >= 0.5 else 0
                                    certeza = probas if classe_prevista == 1 else (1.0 - probas)
                                    class_idx_gradcam = 0
                                else:
                                    probas = torch.softmax(logits_pre, dim=1)[0]
                                    classe_prevista = int(torch.argmax(probas).item())
                                    certeza = float(probas[classe_prevista].item())
                                    class_idx_gradcam = classe_prevista
                            else:
                                probas = torch.softmax(logits_pre, dim=1)[0]
                                classe_prevista = int(torch.argmax(probas).item())
                                certeza = float(probas[classe_prevista].item())
                                class_idx_gradcam = classe_prevista

                            cam, _ = gradcam(x, class_idx=class_idx_gradcam)
                            imagem_visivel = _tensor_para_imagem(
                                x[0],
                                config_dataset.normalizacao_mean,
                                config_dataset.normalizacao_std
                            )
                            cam_np = cam.numpy()
                            imagem_superposta = _superpor_cam(imagem_visivel, cam_np, alpha=0.45)

                            fig, eixos = plt.subplots(1, 3, figsize=(15, 5))

                            eixos[0].imshow(imagem_visivel)
                            eixos[0].set_title("imagem original")
                            eixos[0].axis("off")

                            eixos[1].imshow(cam_np, cmap="jet")
                            eixos[1].set_title("Grad-CAM")
                            eixos[1].axis("off")

                            eixos[2].imshow(imagem_superposta)
                            eixos[2].imshow(cam.numpy(), cmap="jet", alpha=0.45)
                            eixos[2].set_title(f"Grad-CAM: {classes[classe_prevista]} {certeza:.2%}")
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
