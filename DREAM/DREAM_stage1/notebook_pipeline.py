from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import squidpy as sq
import torch
from scipy.special import softmax
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from tqdm.auto import trange

import train
import utils
from cell_type_annotation_model import SpatialModel_cov


@dataclass
class SpleenTutorialConfig:
    data_dir: str = "/home/Data/"
    batches: List[str] = field(default_factory=lambda: ["BALBc-1", "BALBc-2", "BALBc-3"])
    dnn_model: str = "/home/Result/Spleen/DNN_model.pth"
    seed: int = 2025
    n_layers: int = 4
    agg_method: str = "Mean"
    prune_long_links: bool = False
    model_name: str = "Muti"
    network = None
    gae_dim: List[int] = field(default_factory=lambda: [128, 64])
    dae_dim: List[int] = field(default_factory=lambda: [128, 64])
    feat_dim: int = 64
    include_cat_covariates_contrastive_loss: bool = False
    epochs: int = 1000
    optimizer: str = "Adam"
    use_dnn: bool = True
    lr: float = 0.001
    attr_loss_weight: float = 1.0
    bottleneck: bool = False
    n_attributes: int = 1
    edge_weight: bool = True
    kd_T: int = 1
    w_dae: float = 1.0
    w_gae: float = 1.0
    n_cluster: int = 4
    batch_size: int = 4096
    weight_decay: float = 1e-4
    scheduler_step: int = 20
    device: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    def __post_init__(self) -> None:

        self.Model_name = self.model_name
        self.Network = self.network
        self.Prun = self.prune_long_links
        self.DNN = self.use_dnn
        self.CBM_attr = None


def set_seed(seed: int = 2025) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        import torch_geometric

        torch_geometric.seed_everything(seed)
    except Exception:
        pass


def _concat_batch_connectivities(adata: ad.AnnData, adata_batch_list: List[ad.AnnData]) -> None:
    batch_connectivities = []
    len_before_batch = 0
    for i, batch_adata in enumerate(adata_batch_list):
        if i == 0:
            right_pad = sp.csr_matrix((batch_adata.shape[0], adata.shape[0] - batch_adata.shape[0]))
            batch_connectivities.append(sp.hstack((batch_adata.obsp["spatial_connectivities"], right_pad)))
        elif i == len(adata_batch_list) - 1:
            left_pad = sp.csr_matrix((batch_adata.shape[0], adata.shape[0] - batch_adata.shape[0]))
            batch_connectivities.append(sp.hstack((left_pad, batch_adata.obsp["spatial_connectivities"])))
        else:
            left_pad = sp.csr_matrix((batch_adata.shape[0], len_before_batch))
            right_pad = sp.csr_matrix(
                (batch_adata.shape[0], adata.shape[0] - batch_adata.shape[0] - len_before_batch)
            )
            batch_connectivities.append(
                sp.hstack((left_pad, batch_adata.obsp["spatial_connectivities"], right_pad))
            )
        len_before_batch += batch_adata.shape[0]
    adata.obsp["spatial_connectivities"] = sp.vstack(batch_connectivities)


def prepare_adata(cfg: SpleenTutorialConfig) -> ad.AnnData:
    set_seed(cfg.seed)
    adata_batch_list = []
    for batch in cfg.batches:
        adata_batch = sc.read_h5ad(os.path.join(cfg.data_dir, f"{batch}.h5ad"))
        adata_batch.obs["batch"] = batch
        sq.gr.spatial_neighbors(adata_batch, coord_type="generic", delaunay=True)
        if cfg.prune_long_links:
            utils.remove_long_links(adata_batch)
        utils.aggregate_neighbors(
            adata_batch,
            n_layers=cfg.n_layers,
            aggregations=["mean"],
            connectivity_key="spatial_connectivities",
            use_rep=None,
            out_key="X_cellcharter",
            copy=False,
        )
        x = adata_batch.obsm["X_cellcharter"]
        x_reshaped = x.reshape(-1, cfg.n_layers + 1, adata_batch.X.shape[1])
        if cfg.agg_method == "Mean":
            adata_batch.X = np.mean(x_reshaped, axis=1)
        elif cfg.agg_method == "Last":
            adata_batch.X = x_reshaped[:, -1, :]

        adata_batch.obsp["spatial_connectivities"] = adata_batch.obsp["spatial_connectivities"].maximum(
            adata_batch.obsp["spatial_connectivities"].T
        )
        adata_batch.obs["batch"] = adata_batch.obs["batch"].astype("category")
        adata_batch_list.append(adata_batch)

    adata = ad.concat(adata_batch_list, join="inner")
    _concat_batch_connectivities(adata, adata_batch_list)
    return adata


def attach_pseudo_labels(adata: ad.AnnData, cfg: SpleenTutorialConfig) -> None:
    use_dnn = getattr(cfg, "DNN", cfg.use_dnn)
    if use_dnn:
        checkpoint = torch.load(cfg.dnn_model, map_location=cfg.device, weights_only=False)
        dnn_model = checkpoint["model"].to(cfg.device)
        dnn_model.eval()
        adata_x = adata.X
        dnn_inputs = torch.Tensor(adata_x).split(cfg.batch_size)
        dnn_predictions = []
        with torch.no_grad():
            for inputs in dnn_inputs:
                outputs = dnn_model(inputs.to(cfg.device))
                dnn_predictions.append(outputs.detach().cpu().numpy())
        label_names = checkpoint["label_names"]
        adata.obsm["pseudo_label"] = np.concatenate(dnn_predictions)
        adata.obsm["pseudo_label"] = softmax(adata.obsm["pseudo_label"], axis=1)
        adata.obs["pseudo_class"] = pd.Categorical(
            [label_names[i] for i in adata.obsm["pseudo_label"].argmax(1)]
        )
        adata.uns["pseudo_classes"] = label_names
    else:
        adata.obs["pseudo_class"] = adata.obs["CellType"]
    adata.obs["pseudo_class_classes"] = adata.obs["pseudo_class"].cat.codes


def build_model_and_optimizer(adata: ad.AnnData, cfg: SpleenTutorialConfig):
    cat_covariates_keys = ["batch"]
    cat_covariates_embeds_injection = ["decoder"]
    cat_covariates_no_edges = [True]
    cats, dims, skip_edges = utils.Cov_propress(
        adata, cat_covariates_keys=cat_covariates_keys, cat_covariates_no_edges=cat_covariates_no_edges
    )

    model = SpatialModel_cov(
        input_dim=adata.X.shape[1],
        num_classes=[27],
        gae_dim=cfg.gae_dim,
        dae_dim=cfg.dae_dim,
        feat_dim=cfg.feat_dim,
        cat_covariates_keys=cat_covariates_keys,
        include_cat_covariates_contrastive_loss=cfg.include_cat_covariates_contrastive_loss,
        cat_covariates_embeds_injection=cat_covariates_embeds_injection,
        cat_covariates_no_edges=skip_edges,
        cat_covariates_embeds_nums=dims,
        cat_covariates_cats=cats,
    ).to(cfg.device)

    if cfg.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
    elif cfg.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.lr,
            momentum=0.9,
            weight_decay=cfg.weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.lr,
            momentum=0.9,
            weight_decay=cfg.weight_decay,
        )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    return model, optimizer, scheduler


def train_embedding(model, optimizer, adata: ad.AnnData, cfg: SpleenTutorialConfig) -> Dict[str, List[float]]:
    losses: List[float] = []
    dae_losses: List[float] = []
    bce_losses: List[float] = []
    concept_losses: List[float] = []
    contrastive_losses: List[float] = []

    for _ in trange(cfg.epochs, desc="Training Epoch"):
        z, loss, dae_loss, bce_loss, concept_loss, contrastive_loss = train.run_epoch(
            model, optimizer, adata, cfg, is_training=True
        )
        losses.append(loss.item())
        dae_losses.append(dae_loss.item())
        bce_losses.append(bce_loss.item())
        concept_losses.append(concept_loss.item())
        contrastive_losses.append(contrastive_loss.item())

    z, _, _, _, _, _ = train.run_epoch(model, optimizer, adata, cfg, is_training=False)
    adata.obsm["latent"] = z.cpu().detach().numpy().astype(np.float64)
    return {
        "losses": losses,
        "dae_losses": dae_losses,
        "bce_losses": bce_losses,
        "concept_losses": concept_losses,
        "contrastive_losses": contrastive_losses,
    }


def cluster_and_report(
    adata: ad.AnnData,
    batches: List[str],
    n_cluster: int = 4,
    seed: int = 2025,
    pred_key: str = "GM",
    truth_key: str = "Compartment",
) -> pd.DataFrame:
    np.random.seed(seed)
    gm = GaussianMixture(n_components=n_cluster, covariance_type="tied", init_params="kmeans", reg_covar=1e-3)
    y = gm.fit_predict(adata.obsm["latent"], y=None)
    adata.obs[pred_key] = pd.Categorical(y)

    records = []
    for batch in batches:
        adata_sub = adata[adata.obs["batch"] == batch]
        ari_sub = adjusted_rand_score(adata_sub.obs[pred_key], adata_sub.obs[truth_key])
        nmi_sub = normalized_mutual_info_score(adata_sub.obs[pred_key], adata_sub.obs[truth_key])
        records.append({"batch": batch, "ari": ari_sub, "nmi": nmi_sub})
    return pd.DataFrame(records)

