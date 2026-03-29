'''
python tools/extract_esm2_csv.py \
  --csv data/protein_sequences.csv \
  --out-dir data\features\protein_esm2
'''
import argparse
import csv
import re
from pathlib import Path

import torch
import esm


def sanitize_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r'[\\/:*?"<>|]+', "_", name)
    return name or "unnamed"


def read_csv_sequences(csv_path: Path):
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "Name" not in reader.fieldnames or "Sequence" not in reader.fieldnames:
            raise ValueError("CSV must contain 'Name' and 'Sequence' columns.")
        for row in reader:
            name = (row.get("Name") or "").strip()
            seq = (row.get("Sequence") or "").strip()
            if not name or not seq:
                continue
            yield name, seq


def build_batches(entries, toks_per_batch, extra_toks, truncation_seq_length):
    sizes = []
    for i, (_, seq) in enumerate(entries):
        seq_len = len(seq)
        if truncation_seq_length:
            seq_len = min(seq_len, truncation_seq_length)
        sizes.append((seq_len, i))
    sizes.sort()

    batches = []
    buf = []
    max_len = 0
    for seq_len, i in sizes:
        seq_len += extra_toks
        if buf and max(seq_len, max_len) * (len(buf) + 1) > toks_per_batch:
            batches.append(buf)
            buf = []
            max_len = 0
        max_len = max(max_len, seq_len)
        buf.append(i)

    if buf:
        batches.append(buf)
    return batches


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract ESM2-650M per-residue embeddings from a Name/Sequence CSV."
    )
    parser.add_argument("--csv", required=True, type=Path, help="Path to input CSV.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for .pt files.")
    parser.add_argument(
        "--model",
        default="esm2_t33_650M_UR50D",
        help="Model name or local .pt path (default: esm2_t33_650M_UR50D).",
    )
    parser.add_argument("--toks-per-batch", type=int, default=4096, help="Max tokens per batch.")
    parser.add_argument(
        "--truncation-seq-length",
        type=int,
        default=1022,
        help="Truncate sequences longer than this length (default: 1022).",
    )
    parser.add_argument("--nogpu", action="store_true", help="Force CPU even if CUDA is available.")
    return parser.parse_args()


def main():
    args = parse_args()
    entries = list(read_csv_sequences(args.csv))
    if not entries:
        raise ValueError("No valid sequences found in CSV.")

    model, alphabet = esm.pretrained.load_model_and_alphabet(args.model)
    model.eval()

    device = "cpu"
    if torch.cuda.is_available() and not args.nogpu:
        device = "cuda"
        model = model.to(device)

    batch_converter = alphabet.get_batch_converter(args.truncation_seq_length)
    extra_toks = int(alphabet.prepend_bos) + int(alphabet.append_eos)
    batches = build_batches(entries, args.toks_per_batch, extra_toks, args.truncation_seq_length)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    layer = model.num_layers
    with torch.no_grad():
        for batch_idx, batch_indices in enumerate(batches, start=1):
            raw_batch = [(entries[i][0], entries[i][1]) for i in batch_indices]
            labels, strs, toks = batch_converter(raw_batch)
            toks = toks.to(device=device, non_blocking=True)
            out = model(toks, repr_layers=[layer], return_contacts=False)
            reps = out["representations"][layer].to(device="cpu")

            for i, idx in enumerate(batch_indices):
                name, seq = entries[idx]
                trunc_len = len(seq)
                if args.truncation_seq_length:
                    trunc_len = min(trunc_len, args.truncation_seq_length)
                tensor = reps[i, 1 : trunc_len + 1].clone()
                out_path = args.out_dir / f"{sanitize_name(name)}.pt"
                torch.save(tensor, out_path)

            print(f"Processed batch {batch_idx}/{len(batches)} ({len(batch_indices)} sequences).")


if __name__ == "__main__":
    main()
