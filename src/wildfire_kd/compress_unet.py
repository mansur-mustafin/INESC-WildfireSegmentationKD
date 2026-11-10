import torch
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
import argparse
import os
from .models.UNet import UNet


def count_nonzero_parameters(model):
    """Count nonzero (active) parameters and compute sparsity."""
    total = 0
    nonzero = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        tensor = param.detach()
        total += tensor.numel()
        nonzero += torch.count_nonzero(tensor).item()
    sparsity = 1 - (nonzero / total)
    return {"total": total, "nonzero": nonzero, "sparsity": sparsity}


def prune_model(model, target_sparsity=0.3, structured=False, remove=True):
    """
    Prune a copy of the given model to reach a desired global sparsity.
    """
    model = deepcopy(model)
    parameters_to_prune = []

    # Gather all convolutional layers
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            parameters_to_prune.append((module, 'weight'))

    # Apply global pruning
    if structured:
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.LnStructured,
            amount=target_sparsity,
            n=2, dim=0
        )
    else:
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=target_sparsity,
        )

    # Optionally remove reparametrization hooks
    if remove:
        for (module, _) in parameters_to_prune:
            prune.remove(module, 'weight')

    # Compute sparsity statistics
    stats = count_nonzero_parameters(model)
    report = {
        "target_sparsity": target_sparsity,
        "achieved_sparsity": stats["sparsity"],
        "remaining_params": stats["nonzero"],
        "total_params": stats["total"],
    }

    return model, report


def main():
    parser = argparse.ArgumentParser(description="Prune U-Net models for knowledge distillation.")
    parser.add_argument("--output_dir", type=str, default="./weights/pruned_models/", help="Where to save pruned models")
    parser.add_argument("--filters", type=int, default=16, help="Number of base filters in U-Net")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    parser.add_argument("--structured", action="store_true", help="Use structured pruning (channel-level)")
    parser.add_argument("--sparsities", type=float, nargs="+", default=[ 0.5,.7,.8,.9], help="Target sparsities to prune")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained model weights")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    base_model = UNet(n_classes=1, n_filters=args.filters, dropout=args.dropout, batchnorm=True, n_channels=3)

    # Load pretrained weights before pruning
    if args.checkpoint is not None:
        print(f"Loading pretrained weights from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        base_model.load_state_dict(checkpoint)
    else:
        print("⚠️ No checkpoint provided — pruning untrained weights (not recommended).")

    init_stats = count_nonzero_parameters(base_model)
    print(f"Base model parameters: {init_stats['nonzero']:,}/{init_stats['total']:,} "
          f"({(1-init_stats['sparsity'])*100:.2f}% kept)")

    for sparsity in args.sparsities:
        pruned_model, info = prune_model(base_model, target_sparsity=sparsity, structured=args.structured)
        print(f"\n➡️  Pruned {sparsity*100:.0f}%")
        print(f"   Achieved sparsity: {info['achieved_sparsity']*100:.2f}%")
        print(f"   Active parameters: {info['remaining_params']:,}/{info['total_params']:,} "
              f"({(1-info['achieved_sparsity'])*100:.2f}% kept)")

        save_path = os.path.join(args.output_dir, f"unet_{int(sparsity*100)}_pruned.pt")
        torch.save(pruned_model.state_dict(), save_path)
        print(f"   Saved to {save_path}")


if __name__ == "__main__":
    main()
