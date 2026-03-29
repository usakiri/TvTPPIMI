import argparse
import csv
import hashlib
from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem
from tqdm import tqdm

from module.geppimi_gnn import GNNEncoder, smiles_to_pyg


def _hash_smiles(smiles: str) -> str:
    return hashlib.md5(smiles.encode("utf-8")).hexdigest()


def load_model(weight_path: Path, device: torch.device, emb_dim: int = 300) -> GNNEncoder:
    model = GNNEncoder(num_layer=5, emb_dim=emb_dim, JK="last", drop_ratio=0.0, gnn_type="gin")
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Pre-compute ligand node embeddings with GraphMVP_C.")
    parser.add_argument("--input-csv", required=True, help="Path to CSV containing SMILES column.")
    parser.add_argument("--smiles-col", default="SMILES", help="Column name for SMILES (default: SMILES).")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save per-molecule .pt feature files (will be created).",
    )
    parser.add_argument(
        "--mapping",
        required=True,
        help="Output CSV path for mapping (columns: smiles,pt_file).",
    )
    parser.add_argument(
        "--weight",
        default="./weights/GraphMVP_C.model",
        help="Path to GraphMVP_C pretrained weights.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to run encoder (cpu or cuda:0). CPU avoids GPU non-determinism.",
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    mapping_path = Path(args.mapping)
    weight_path = Path(args.weight)
    device = torch.device(args.device)

    output_dir.mkdir(parents=True, exist_ok=True)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    if args.smiles_col not in df.columns:
        raise KeyError(f"Column {args.smiles_col} not found in {input_csv}")

    smiles_list = df[args.smiles_col].dropna().astype(str).tolist()
    unique_smiles = list(dict.fromkeys(smiles_list))

    model = load_model(weight_path, device)

    rows = []
    skipped = 0
    with torch.no_grad():
        with tqdm(unique_smiles, desc="Encoding ligands", total=len(unique_smiles)) as pbar:
            for smi in pbar:
                data = smiles_to_pyg(smi, Chem.MolFromSmiles)
                if data is None:
                    skipped += 1
                    pbar.set_postfix(saved=len(rows), skipped=skipped)
                    continue
                data = data.to(device)
                node_repr = model(data).cpu()
                fname = f"{_hash_smiles(smi)}.pt"
                torch.save(node_repr, output_dir / fname)
                rows.append({"smiles": smi, "pt_file": fname})
                pbar.set_postfix(saved=len(rows), skipped=skipped)

    with open(mapping_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["smiles", "pt_file"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done: generated {len(rows)} molecules, skipped {skipped} invalid SMILES.")
    print(f"Feature dir: {output_dir}")
    print(f"Mapping file: {mapping_path}")


if __name__ == "__main__":
    main()
