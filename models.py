import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as nn_init
from typing import Optional
from module.CN import *
from protein_types import ProteinBatch

class TvTPPIMI(nn.Module):
    def __init__(self, **config):
        super(TvTPPIMI, self).__init__()
        """
        fm (drug branch): small-molecule modulator features;
        fp (protein branch): protein complex features.
        """
        
        drug_cfg = config["MODULATOR"]
        drug_external = bool(getattr(drug_cfg, "EXTERNAL", True))
        drug_external_dim = int(getattr(drug_cfg, "EXTERNAL_DIM", 0))

        protein_cfg = config["PROTEIN"]
        protein_external = bool(getattr(protein_cfg, "EXTERNAL", True))
        protein_external_dim = int(getattr(protein_cfg, "EXTERNAL_DIM", 0))
        protein_static_cfg = getattr(protein_cfg, "STATIC_FEATURES", None)
        self.use_static_esm = bool(getattr(protein_static_cfg, "USE_ESM", True)) if protein_static_cfg is not None else True
        static_esm_dim = int(getattr(protein_static_cfg, "ESM_DIM", 1280)) if protein_static_cfg is not None else 1280
        self.static_esm_dim = static_esm_dim if self.use_static_esm else 0
        self.static_input_dim = self.static_esm_dim
        separator_cfg = getattr(protein_cfg, "SEPARATOR", None)
        self.separator_enabled = bool(getattr(separator_cfg, "ENABLED", False)) if separator_cfg is not None else False
        self.separator_add_endpoints = bool(getattr(separator_cfg, "ADD_ENDPOINTS", False)) if separator_cfg is not None else False

        self.protein_dynamic_enabled = False
        self.protein_dynamic_dim = 0
        self.expected_static_dim = None
        self.static_target_dim = protein_external_dim
        if self.static_input_dim == 0:
            raise ValueError("Enable static ESM protein representations.")
        self.esm_projector = None
        if self.use_static_esm and self.static_esm_dim != self.static_target_dim:
            self.esm_projector = nn.Linear(self.static_esm_dim, self.static_target_dim)

        if protein_external:
            self.expected_static_dim = self.static_input_dim if self.static_input_dim > 0 else None
            final_total = self.static_target_dim + self.protein_dynamic_dim
            if final_total <= 0:
                raise ValueError("Configured protein features produce zero-dimensional representations.")

        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        out_binary = config["DECODER"]["BINARY"]

        self.special_token_embeddings = None
        self.special_token_codes = {}
        if self.separator_enabled:
            token_dim = self.expected_static_dim if self.expected_static_dim is not None else self.static_input_dim
            self.special_token_embeddings = nn.ParameterDict()
            # Always create separator token
            self.special_token_embeddings["sep"] = nn.Parameter(torch.zeros(token_dim))
            self.special_token_codes["sep"] = 2
            if self.separator_add_endpoints:
                self.special_token_embeddings["cls_head"] = nn.Parameter(torch.zeros(token_dim))
                self.special_token_embeddings["cls_tail"] = nn.Parameter(torch.zeros(token_dim))
                self.special_token_codes["cls_head"] = 1
                self.special_token_codes["cls_tail"] = 3
        
        self.stage_num = config["STAGE"]["NUM"]
        self.bcfm_flag = config["STAGE"]["BCFM"]
        self.ffm_flag = config["STAGE"]["FFM"]

        self.bcfm_dim = config["BCFM"]["DIM"]
        self.bcfm_drop = float(config["BCFM"].get("DROP_RATE", 0.1))
        self.ffm_dim_cfg = config["FFM"]["DIM"]
        self.ffm_head = config["FFM"]["NUM_HEAD"]
        self.ffm_drop = config["FFM"]["DROP_RATE"]

        static_target_total = self.static_target_dim + self.protein_dynamic_dim
        base_protein_dim = static_target_total

        if self.bcfm_flag:
            self.token_dim = self.bcfm_dim
        elif self.ffm_flag:
            self.token_dim = self.ffm_dim_cfg
        else:
            self.token_dim = base_protein_dim

        if self.bcfm_flag and self.ffm_flag and self.ffm_dim_cfg != self.token_dim:
            raise ValueError("BCFM and FFM output dimensions must match. Set FFM.DIM equal to BCFM.DIM.")

        if not drug_external:
            raise ValueError(
                "OmniPPIMI expects MODULATOR.EXTERNAL to be True. "
                "Pre-compute ligand features and set MODULATOR.EXTERNAL_DIM accordingly."
            )
        self.drug_dim = drug_external_dim
        self.drug_projector = nn.Linear(self.drug_dim, self.token_dim) if self.drug_dim != self.token_dim else nn.Identity()

        if not protein_external:
            raise ValueError("OmniPPIMI expects PROTEIN.EXTERNAL to be True. Provide ESM features via FEATURE configuration.")
        self.protein_extractor = None
        self.protein_raw_dim = static_target_total
        self.protein_projector = nn.Linear(self.protein_raw_dim, self.token_dim) if self.protein_raw_dim != self.token_dim else nn.Identity()
        
        if self.bcfm_flag:
            self.bcfm_list = nn.ModuleList([BCFM(dim_model=self.bcfm_dim, drop_rate=self.bcfm_drop) for i in range(self.stage_num)])
        
        fusion_dim = None
        if self.ffm_flag:
            self.fusion = FFM(dim_model=self.token_dim, num_head=self.ffm_head, drop_rate=self.ffm_drop)
            fusion_dim = self.token_dim
        else:
            self.fusion = SimpleFusion()
            fusion_dim = self.token_dim

        if fusion_dim is not None and fusion_dim != mlp_in_dim:
            mlp_in_dim = fusion_dim
            
        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)

    def _blend_static_features(self, static_feat: torch.Tensor) -> torch.Tensor:
        """
        Prepare static ESM features for the protein branch.
        """
        if self.expected_static_dim and static_feat.size(-1) != self.expected_static_dim:
            raise ValueError(
                f"Received protein static feature dim {static_feat.size(-1)} but expected {self.expected_static_dim}."
            )

        if not self.use_static_esm:
            raise ValueError("ESM features are disabled but required for protein encoding.")
        if static_feat is None or static_feat.size(-1) < self.static_esm_dim:
            raise ValueError("ESM features are enabled but missing from the protein batch.")

        esm_feat = static_feat[..., : self.static_esm_dim]
        if self.esm_projector is not None:
            esm_feat = self.esm_projector(esm_feat)

        return esm_feat

    def forward(self, v_d, v_p, v_d_mask, v_p_mask, return_attention=False):
        v_d = self.drug_projector(v_d)

        protein_mask = v_p_mask
        protein_features = v_p

        if isinstance(v_p, ProteinBatch):
            protein_mask = v_p.mask
            static_feat = v_p.static
            special_tokens = getattr(v_p, "special_tokens", None)
            if (
                static_feat is not None
                and special_tokens is not None
                and self.special_token_embeddings is not None
                and self.special_token_codes
            ):
                for name, code in self.special_token_codes.items():
                    token_mask = (special_tokens == code).unsqueeze(-1)
                    if token_mask.any():
                        token_value = self.special_token_embeddings[name].view(1, 1, -1).to(static_feat.dtype)
                        static_feat = torch.where(token_mask, token_value.expand_as(static_feat), static_feat)
            if static_feat is not None:
                protein_features = self._blend_static_features(static_feat)
            else:
                raise ValueError("Protein batch is missing static features required by the configuration.")
        
        if self.protein_extractor is not None:
            protein_features = self.protein_extractor(protein_features)
        protein_features = self.protein_projector(protein_features)
        protein_features = protein_features.masked_fill(protein_mask.unsqueeze(-1), 0)
        
        if self.bcfm_flag:
            for i in range(self.stage_num):
                protein_features, v_d = self.bcfm_list[i](protein_features, v_d, protein_mask, v_d_mask)
        attn_outputs = None
        if self.ffm_flag:
            fusion_out = self.fusion(protein_features, v_d, protein_mask, v_d_mask, return_attn=return_attention)
            if return_attention:
                if isinstance(fusion_out, tuple):
                    if len(fusion_out) == 3:
                        f, attn_fp, attn_fm = fusion_out
                        z_fp = None
                        z_fm = None
                    else:
                        f, attn_fp, attn_fm, z_fp, z_fm = fusion_out
                else:
                    raise ValueError("Expected FFM fusion to return a tuple when attention outputs are requested.")
                attn_outputs = {"protein": attn_fp, "drug": attn_fm, "fusion_repr": f}
                if z_fp is not None:
                    attn_outputs["protein_repr"] = z_fp
                if z_fm is not None:
                    attn_outputs["drug_repr"] = z_fm
            else:
                f = fusion_out
        else:
            f = self.fusion(protein_features, v_d, protein_mask, v_d_mask)

        score = self.mlp_classifier(f)

        outputs = (v_d, protein_features, f, score)
        if return_attention and attn_outputs is not None:
            return outputs + (attn_outputs,)
        return outputs

class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Tanh(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.Tanh(),
        )
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

def binary_cross_entropy(logits, labels, label_smoothing=0.0):
    loss_fct = nn.BCEWithLogitsLoss()
    labels = labels.view_as(logits)
    if label_smoothing > 0:
        smooth_labels = labels * (1 - label_smoothing) + 0.5 * label_smoothing
    else:
        smooth_labels = labels
    loss = loss_fct(logits, smooth_labels)
    probs = torch.sigmoid(logits)
    if probs.size(-1) == 1:
        probs = probs.squeeze(-1)
    return probs, loss

def cross_entropy_logits(linear_output, label, weights=None, label_smoothing=0.0):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]
    if label_smoothing > 0 and weights is None:
        loss = F.cross_entropy(
            linear_output,
            label.type_as(y_hat).view(label.size(0)),
            label_smoothing=label_smoothing,
        )
    elif weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss


class SimpleFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = MaskedAveragePooling()

    def forward(self, fp_feat, fm_feat, fp_mask, fm_mask):
        fp_pooled = self.avgpool(fp_feat, fp_mask)
        fm_pooled = self.avgpool(fm_feat, fm_mask)

        interaction = fp_pooled + fm_pooled

        return interaction
