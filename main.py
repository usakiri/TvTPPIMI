from models import TvTPPIMI
from time import time
from utils import set_seed, graph_collate_func
from configs import get_cfg_defaults
from dataloader import PPIMIDataset
from torch.utils.data import DataLoader

from run.trainer import Trainer

import torch
import argparse
import warnings, os
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description="TvTPPIMI for multi-purpose ESI prediction [TRAIN]")
parser.add_argument('--model', required=True, help="path to model config file", type=str)
parser.add_argument('--data', required=True, help="path to data config file", type=str)
parser.add_argument(
    '--init-checkpoint',
    default="",
    type=str,
    help="Optional path to a .pth checkpoint to initialize model weights",
)
parser.add_argument(
    '--seed',
    type=int,
    default=None,
    help="Override SOLVER.SEED with a single seed value for this run",
)
args = parser.parse_args()

def main():
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.model)
    cfg.merge_from_file(args.data)
    if args.seed is not None:
        cfg.SOLVER.SEED = [args.seed]

    print(f"Model Config: {args.model}")
    print(f"Data Config: {args.data}")
    print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}", end="\n\n")

    dataFolder = cfg.SOLVER.DATA

    train_path = os.path.join(dataFolder, 'train.csv')
    val_path = os.path.join(dataFolder, "val.csv")
    if not os.path.exists(val_path):
        alt_val_path = os.path.join(dataFolder, "valid.csv")
        if os.path.exists(alt_val_path):
            val_path = alt_val_path
    test_path = os.path.join(dataFolder, "test.csv")

    if not os.path.exists(train_path):
        fold_index = getattr(cfg.SOLVER, "FOLD_INDEX", None)
        if fold_index is None:
            raise FileNotFoundError(
                f"Cannot find train.csv under {dataFolder}. "
                "Set SOLVER.FOLD_INDEX in the config to use fold-based CSVs."
            )
        train_path = os.path.join(dataFolder, f"train_fold{fold_index}.csv")
        val_path = os.path.join(dataFolder, f"valid_fold{fold_index}.csv")
        test_path = os.path.join(dataFolder, f"test_fold{fold_index}.csv")
        for path in (train_path, val_path, test_path):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Expected file not found: {path}")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)
    
    train_dataset = PPIMIDataset(df_train.index.values, df_train, "binary", cfg)
    val_dataset = PPIMIDataset(df_val.index.values, df_val, "binary", cfg)
    test_dataset = PPIMIDataset(df_test.index.values, df_test, "binary", cfg)
    
    train_params = {
        'batch_size': cfg.SOLVER.BATCH_SIZE,
        'shuffle': True,
        'num_workers': cfg.SOLVER.NUM_WORKERS,
        'drop_last': True,
        'collate_fn': graph_collate_func,
    }
    eval_params = {
        'batch_size': cfg.SOLVER.BATCH_SIZE,
        'shuffle': False,
        'num_workers': cfg.SOLVER.NUM_WORKERS,
        'drop_last': False,
        'collate_fn': graph_collate_func,
    }

    training_generator = DataLoader(train_dataset, **train_params)
    val_generator = DataLoader(val_dataset, **eval_params)
    test_generator = DataLoader(test_dataset, **eval_params)


    torch.backends.cudnn.benchmark = True

    output_dir = os.path.join(cfg.RESULT.OUTPUT_DIR, cfg.SOLVER.SAVE)
    
    base_dir = os.path.join(cfg.RESULT.OUTPUT_DIR, cfg.SOLVER.SAVE)
    model_name = os.path.splitext(os.path.basename(args.model))[0]
    
    test_metrics = None
    for seed in cfg.SOLVER.SEED:
        print(f"=====> Start Training for Seed {seed}")
        set_seed(seed)
        
        model = TvTPPIMI(**cfg)

        # Optional: initialize from a saved checkpoint
        if args.init_checkpoint:
            print(f"[INIT] Loading checkpoint from: {args.init_checkpoint}")
            state = torch.load(args.init_checkpoint, map_location="cpu")
            # Allow both pure state_dict and dict with 'state_dict' key
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            # Strip leading 'module.' if checkpoint was saved from a DDP-wrapped model
            if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
                state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f"[INIT] Checkpoint loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

        model.to(device)

        weight_decay = cfg.SOLVER.L2_LAMBDA if cfg.SOLVER.USE_L2_REGULARIZATION else 0.0
        opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR, weight_decay=weight_decay)

        model_seed_name = f"{model_name}_{seed}"
        output_dir = os.path.join(base_dir, model_seed_name)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        os.system(f'cp -r ./module {output_dir}/')
        os.system(f'cp ./models.py {output_dir}/')
        os.system(f'cp -r {args.data} {output_dir}/')
        os.system(f'cp -r {args.model} {output_dir}/')

        trainer = Trainer(
            seed,
            model,
            opt,
            device,
            training_generator,
            val_generator,
            test_generator,
            output=output_dir,
            **cfg,
        )

        test_metrics, y_pred, y_label = trainer.train()

        y_pred = np.array(y_pred)
        y_label = np.array(y_label)
        df_test['y_pred'] = y_pred
        df_test['y_label'] = y_label
        df_test.to_csv(os.path.join(output_dir, f"prediction_{seed}.csv"))

    print(f"Directory for saving result: {output_dir}")

    return test_metrics


if __name__ == '__main__':
    s = time()
    result = main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")
