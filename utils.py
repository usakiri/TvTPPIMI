import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from protein_types import ProteinBatch, ProteinSample


def _resize_feature(feature: torch.Tensor, target_len: int) -> torch.Tensor:
    if feature.size(0) == target_len:
        return feature
    feature = feature.unsqueeze(0).transpose(1, 2)
    feature = F.interpolate(feature, size=target_len, mode="linear", align_corners=False)
    feature = feature.transpose(1, 2).squeeze(0)
    return feature


def _prepare_protein_batch(samples):
    if not samples:
        raise ValueError("Protein batch is empty.")

    first_sample = samples[0]
    if isinstance(first_sample, ProteinSample):
        sample_lengths = []
        static_dim = None
        static_dtype = torch.float32
        static_features = []
        special_tokens_enabled = any(sample.special_tokens is not None for sample in samples)
        special_tokens_list = []
        for sample in samples:
            if sample.static is not None:
                static_dim = sample.static.size(1)
                static_dtype = sample.static.dtype
                break

        for idx, sample in enumerate(samples):
            length_hint = sample.total_length_hint
            if length_hint is None or length_hint <= 0:
                if sample.static is not None:
                    length_hint = sample.static.size(0)
                else:
                    raise ValueError("Unable to determine protein length for sample index {}.".format(idx))
            sample_lengths.append(int(length_hint))
            if sample.static is not None:
                feature = sample.static
                if feature.size(0) != length_hint:
                    feature = _resize_feature(feature, int(length_hint))
                static_features.append(feature)
            else:
                static_features.append(None)
            if special_tokens_enabled:
                special_tensor = sample.special_tokens
                if special_tensor is None:
                    special_tensor = torch.zeros(int(length_hint), dtype=torch.long)
                if special_tensor.size(0) != length_hint:
                    special_tensor = special_tensor[: int(length_hint)]
                    if special_tensor.size(0) < length_hint:
                        pad_len = int(length_hint) - special_tensor.size(0)
                        padding = torch.zeros(pad_len, dtype=torch.long)
                        special_tensor = torch.cat([special_tensor, padding], dim=0)
                special_tokens_list.append(special_tensor.long())

        max_len = max(sample_lengths) if sample_lengths else 0
        if static_dim is not None:
            padded_static = torch.zeros(len(samples), max_len, static_dim, dtype=static_dtype)
            for idx, feature in enumerate(static_features):
                if feature is None:
                    continue
                current = feature
                if current.size(0) != sample_lengths[idx]:
                    current = _resize_feature(current, sample_lengths[idx])
                pad_len = max_len - current.size(0)
                if pad_len > 0:
                    current = F.pad(current, (0, 0, 0, pad_len), value=0)
                padded_static[idx] = current
        else:
            padded_static = None

        padded_special_tokens = None
        if special_tokens_enabled:
            padded_special_tokens = torch.zeros(len(samples), max_len, dtype=torch.long)
            for idx, special_tensor in enumerate(special_tokens_list):
                current = special_tensor
                if current.size(0) < sample_lengths[idx]:
                    padding = torch.zeros(sample_lengths[idx] - current.size(0), dtype=torch.long)
                    current = torch.cat([current, padding], dim=0)
                elif current.size(0) > sample_lengths[idx]:
                    current = current[: sample_lengths[idx]]
                pad_len = max_len - current.size(0)
                if pad_len > 0:
                    padding = torch.zeros(pad_len, dtype=torch.long)
                    current = torch.cat([current, padding], dim=0)
                padded_special_tokens[idx] = current

        mask = torch.ones(len(samples), max_len, dtype=torch.bool)
        for idx, length in enumerate(sample_lengths):
            mask[idx, :length] = False
        lengths_tensor = torch.tensor(sample_lengths, dtype=torch.long)
        return ProteinBatch(
            static=padded_static,
            mask=mask,
            lengths=lengths_tensor,
            special_tokens=padded_special_tokens if special_tokens_enabled else None,
        )

    # Legacy path: plain tensors
    lengths = [protein.size(0) for protein in samples]
    max_len = max(lengths)
    padded_protein_feat = []
    protein_masks = torch.ones(len(samples), max_len, dtype=torch.bool)
    for idx, protein in enumerate(samples):
        pad_len = max_len - protein.size(0)
        if pad_len > 0:
            padded = F.pad(protein, (0, 0, 0, pad_len), value=0)
        else:
            padded = protein
        protein_masks[idx, :protein.size(0)] = False
        padded_protein_feat.append(padded.unsqueeze(0))
    padded_protein_feat = torch.cat(padded_protein_feat, dim=0)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    return ProteinBatch(static=padded_protein_feat, mask=protein_masks, lengths=lengths_tensor, special_tokens=None)


def set_seed(seed=1000):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.enabled = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["PYTHONHASHSEED"] = str(seed)
        

def graph_collate_func(x):
    d, p, y = zip(*x)
    protein_batch = _prepare_protein_batch(p)
    labels = torch.tensor(y, dtype=torch.float32)

    drug_lengths = [drug.size(0) for drug in d]
    max_drug_length = max(drug_lengths)

    padded_drug_feat = []
    drug_masks = torch.ones(len(d), max_drug_length, dtype=torch.bool)
    for idx, drug in enumerate(d):
        pad_len = max_drug_length - drug.size(0)
        if pad_len > 0:
            padded = F.pad(drug, (0, 0, 0, pad_len), value=0)
        else:
            padded = drug
        drug_masks[idx, :drug.size(0)] = False
        padded_drug_feat.append(padded.unsqueeze(0))

    padded_drug_feat = torch.cat(padded_drug_feat, dim=0)

    return padded_drug_feat, protein_batch, labels, drug_masks, protein_batch.mask



