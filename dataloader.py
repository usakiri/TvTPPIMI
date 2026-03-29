import torch.utils.data as data
import torch
import torch.nn.functional as F
from pathlib import Path

import pandas as pd
from protein_types import ProteinSample

"""
Load feature from hard disk or pre-computed tensor files.
"""


class PPIMIDataset(data.Dataset):
    def __init__(self, list_IDs, df, task='binary', cfg=None):
        self.list_IDs = list_IDs
        self.df = df
        self.task = task
        self.cfg = cfg

        self.use_external_drug = False
        self.use_external_protein = False
        self.max_protein_pair = 2

        self.separator_enabled = False
        self.separator_add_endpoints = False
        self.special_token_codes = {"cls_head": 1, "sep": 2, "cls_tail": 3}

        if cfg is not None:
            self.use_external_drug = bool(cfg["MODULATOR"].EXTERNAL if "EXTERNAL" in cfg["MODULATOR"] else False)
            self.use_external_protein = bool(cfg["PROTEIN"].EXTERNAL if "EXTERNAL" in cfg["PROTEIN"] else False)
            separator_cfg = cfg["PROTEIN"].SEPARATOR if "SEPARATOR" in cfg["PROTEIN"] else None
            self.separator_enabled = bool(separator_cfg.ENABLED) if separator_cfg is not None and "ENABLED" in separator_cfg else False
            self.separator_add_endpoints = bool(separator_cfg.ADD_ENDPOINTS) if (
                separator_cfg is not None and "ADD_ENDPOINTS" in separator_cfg
            ) else False
            if self.use_external_drug:
                feature_cfg = cfg.FEATURE if "FEATURE" in cfg else None
                compound_dir = feature_cfg.COMPOUND_DIR if feature_cfg is not None and "COMPOUND_DIR" in feature_cfg else ""
                compound_mapping = feature_cfg.COMPOUND_MAPPING if feature_cfg is not None and "COMPOUND_MAPPING" in feature_cfg else ""
                if not compound_dir or not compound_mapping:
                    raise ValueError("Compound external features enabled but FEATURE.COMPOUND_DIR or "
                                     "FEATURE.COMPOUND_MAPPING not configured.")
                self.compound_dir = Path(compound_dir)
                mapping_df = pd.read_csv(compound_mapping)
                self.smiles_to_compound = {
                    row["smiles"]: self.compound_dir / row["pt_file"]
                    for _, row in mapping_df.iterrows()
                }
            else:
                raise ValueError(
                    "TvTPPIMI expects MODULATOR.EXTERNAL to be True. "
                )

            if self.use_external_protein:
                feature_cfg = cfg.FEATURE if "FEATURE" in cfg else None
                esm_dir = feature_cfg.PROTEIN_ESM_DIR if feature_cfg is not None and "PROTEIN_ESM_DIR" in feature_cfg else ""
                protein_static_cfg = cfg["PROTEIN"].STATIC_FEATURES if "STATIC_FEATURES" in cfg["PROTEIN"] else None
                use_static_esm = bool(protein_static_cfg.USE_ESM) if protein_static_cfg is not None and "USE_ESM" in protein_static_cfg else True
                self.protein_feature_dirs = {}
                if use_static_esm:
                    if not esm_dir:
                        raise ValueError("Protein external features enabled but FEATURE.PROTEIN_ESM_DIR not configured.")
                    self.protein_feature_dirs["esm"] = Path(esm_dir)
                if not self.protein_feature_dirs:
                    raise ValueError("No static protein feature sources enabled. Enable ESM features.")
        if self.use_external_protein:
            filtered_indices = []
            skipped = 0
            for idx in self.list_IDs:
                row = self.df.iloc[idx]
                if self._has_valid_protein_id(row):
                    filtered_indices.append(idx)
                else:
                    skipped += 1
            if skipped:
                print(f"[PPIMIDataset] Skipped {skipped} samples due to missing/invalid protein UniProt IDs.")
            self.list_IDs = filtered_indices

    def __len__(self):
        return len(self.list_IDs)

    def _has_valid_protein_id(self, row):
        for idx in range(1, self.max_protein_pair + 1):
            col = f'uniprot_id{idx}'
            if col not in row or pd.isna(row[col]):
                continue
            uniprot_id = str(row[col]).strip()
            if uniprot_id and uniprot_id.lower() not in {"na", "nan"}:
                return True
        return False

    def __getitem__(self, index):
        index = self.list_IDs[index]
        row = self.df.iloc[index]

        smiles = row['SMILES']
        try:
            compound_path = self.smiles_to_compound[smiles]
        except KeyError as exc:
            raise KeyError(f"Compound feature for SMILES {smiles} not found in mapping file.") from exc
        if not compound_path.exists():
            raise FileNotFoundError(f"Compound feature file not found: {compound_path}")
        v_d = torch.load(compound_path)

        if self.use_external_protein:
            static_segments = []
            special_segments = []
            total_length_hint = 0
            for idx in range(1, self.max_protein_pair + 1):
                col = f"uniprot_id{idx}"
                if col not in row or pd.isna(row[col]):
                    continue
                uniprot_id = str(row[col]).strip()
                if uniprot_id == "" or uniprot_id.lower() in {"na", "nan"}:
                    continue

                concat_parts = []
                for feature_name, feat_dir in self.protein_feature_dirs.items():
                    feat_path = feat_dir / f"{uniprot_id}.pt"
                    if not feat_path.exists():
                        raise FileNotFoundError(f"Protein feature file not found: {feat_path}")
                    tensor = torch.load(feat_path)
                    if isinstance(tensor, dict) and "representations" in tensor:
                        reps = tensor["representations"]
                        layer_key = max(reps.keys())
                        tensor = reps[layer_key]
                    concat_parts.append(tensor)

                component_tensor = None
                component_length = None
                if concat_parts:
                    base_len = concat_parts[0].size(0)
                    aligned_parts = [concat_parts[0]]
                    for part in concat_parts[1:]:
                        if part.size(0) != base_len:
                            part = part.unsqueeze(0).transpose(1, 2)
                            part = F.interpolate(part, size=base_len, mode="linear", align_corners=False)
                            part = part.transpose(1, 2).squeeze(0)
                        aligned_parts.append(part)
                    component_tensor = torch.cat(aligned_parts, dim=-1)
                    component_length = component_tensor.size(0)
                    static_segments.append(component_tensor)
                    total_length_hint += component_length
                    special_segments.append(torch.zeros((component_length,), dtype=torch.long))

            if self.separator_enabled and static_segments:
                feature_dim = static_segments[0].size(1)
                separated_segments = []
                separated_special_segments = []
                extra_tokens = 0
                if self.separator_add_endpoints:
                    separator_tensor = torch.zeros((1, feature_dim), dtype=static_segments[0].dtype)
                    separated_segments.append(separator_tensor)
                    separated_special_segments.append(torch.tensor([self.special_token_codes["cls_head"]], dtype=torch.long))
                    extra_tokens += 1
                for idx, segment in enumerate(static_segments):
                    if idx > 0:
                        separator_tensor = torch.zeros((1, feature_dim), dtype=segment.dtype)
                        separated_segments.append(separator_tensor)
                        extra_tokens += 1
                        separated_special_segments.append(torch.tensor([self.special_token_codes["sep"]], dtype=torch.long))
                    separated_segments.append(segment)
                    separated_special_segments.append(special_segments[idx])
                if self.separator_add_endpoints:
                    separator_tensor = torch.zeros((1, feature_dim), dtype=static_segments[-1].dtype)
                    separated_segments.append(separator_tensor)
                    extra_tokens += 1
                    separated_special_segments.append(torch.tensor([self.special_token_codes["cls_tail"]], dtype=torch.long))
                static_segments = separated_segments
                special_segments = separated_special_segments
                total_length_hint += extra_tokens

            static_tensor = torch.cat(static_segments, dim=0) if static_segments else None
            special_tensor = torch.cat(special_segments, dim=0) if special_segments else None
            if special_tensor is not None and torch.all(special_tensor == 0):
                special_tensor = None
            length_hint = total_length_hint if total_length_hint > 0 else None
            protein_sample = ProteinSample(
                static=static_tensor,
                total_length_hint=length_hint,
                special_tokens=special_tensor,
            )
            if protein_sample.static is None:
                raise ValueError(f"No protein features found for row index {index}.")
        else:
            v_p_tensor = torch.load(row["Protein_Path"])
            protein_sample = ProteinSample(static=v_p_tensor, total_length_hint=v_p_tensor.size(0), special_tokens=None)

        if self.task == 'binary':
            if "Y" in self.df.columns:
                y = float(row["Y"])
            elif "label" in self.df.columns:
                y = float(row["label"])
            else:
                raise KeyError("Cannot locate binary label column (expected 'Y' or 'label').")
        else:
            if "Score" not in self.df.columns:
                raise KeyError("Cannot locate regression label column 'Score'.")
            y = float(row["Score"])
        return v_d, protein_sample, y
