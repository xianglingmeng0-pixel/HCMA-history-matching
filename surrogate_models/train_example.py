import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

try:
    from .hrcn import HRCN
    from .ivit import IVIT
    from .HCMA_1dcnn import HCMA_1dcnn
    from .HCMA_2dcnn import HCMA_2DCNN
    from .training_utils import SurrogateDataset, save_surrogate_checkpoint, set_seed, train_surrogate
except ImportError:
    from hrcn import HRCN
    from ivit import IVIT
    from HCMA_1dcnn import HCMA_1dcnn
    from HCMA_2dcnn import HCMA_2DCNN
    from training_utils import SurrogateDataset, save_surrogate_checkpoint, set_seed, train_surrogate


MODEL_REGISTRY = {
    "hrcn": HRCN,
    "ivit": IVIT,
    "hcma_1dcnn": HCMA_1dcnn,
    "hcma_2dcnn": HCMA_2DCNN,
}


def parse_tuple(value, expected_length, name):
    parts = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(parts) != expected_length:
        raise argparse.ArgumentTypeError(f"{name} must contain {expected_length} comma-separated integers.")
    return parts


def parse_args():
    parser = argparse.ArgumentParser(description="Train a surrogate model on .npy arrays or synthetic demo data.")
    parser.add_argument("--model", choices=MODEL_REGISTRY.keys(), default="hrcn", help="Model architecture.")
    parser.add_argument("--x-path", type=Path, default=None, help="Input parameter array saved as .npy.")
    parser.add_argument("--y-path", type=Path, default=None, help="Production target array saved as .npy.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/surrogate_demo"))
    parser.add_argument("--num-samples", type=int, default=64, help="Number of synthetic samples used when no data files are provided.")
    parser.add_argument("--grid-shape", type=lambda x: parse_tuple(x, 3, "grid_shape"), default=(1, 60, 60))
    parser.add_argument("--ts-feature", type=lambda x: parse_tuple(x, 2, "ts_feature"), default=(50, 8))
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-encoder", type=int, default=1)
    parser.add_argument("--n-decoder", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (args.x_path is None) != (args.y_path is None):
        parser.error("--x-path and --y-path must be provided together.")
    return args


def load_arrays(args, grid_shape, ts_feature):
    if args.x_path is not None and args.y_path is not None:
        return np.load(args.x_path), np.load(args.y_path)

    n_samples = int(args.num_samples)
    x_dim = grid_shape[0] * grid_shape[1] * grid_shape[2]
    y_dim = ts_feature[0] * ts_feature[1]
    x = np.random.rand(n_samples, x_dim).astype(np.float32)
    y = np.random.rand(n_samples, y_dim).astype(np.float32)
    print(
        "No data files provided. Using synthetic random data: "
        f"x={x.shape}, y={y.shape}."
    )
    return x, y


def build_model(args):
    common_config = {
        "ts_feature": args.ts_feature,
        "grid_shape": args.grid_shape,
        "d_model": args.d_model,
        "n_encoder": args.n_encoder,
        "n_decoder": args.n_decoder,
    }

    if args.model in {"hrcn", "ivit"}:
        model_config = {
            "ts_feature": args.ts_feature,
            "in_channels": args.grid_shape[0],
            "d_model": args.d_model,
            "n_encoder": args.n_encoder,
            "n_decoder": args.n_decoder,
        }
        return MODEL_REGISTRY[args.model](**model_config), model_config

    return MODEL_REGISTRY[args.model](**common_config), common_config


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sequence_input = args.model == "hcma_1dcnn"

    x, y = load_arrays(args, args.grid_shape, args.ts_feature)
    dataset = SurrogateDataset(
        x,
        y,
        grid_shape=args.grid_shape,
        ts_feature=args.ts_feature,
        sequence_input=sequence_input,
    )
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model, model_config = build_model(args)
    model_config["model"] = args.model

    model, optimizer, history = train_surrogate(
        model,
        train_loader,
        val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
    )
    save_surrogate_checkpoint(args.output_dir, model, optimizer, history, model_config=model_config)
    print(f"Saved checkpoint and training summary to: {args.output_dir}")


if __name__ == "__main__":
    main()
