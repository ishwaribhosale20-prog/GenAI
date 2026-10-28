from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    epoch: int,
    history: dict[str, list[float]],
    config: dict[str, Any],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "generator_optimizer": generator_optimizer.state_dict(),
            "discriminator_optimizer": discriminator_optimizer.state_dict(),
            "history": history,
            "config": config,
        },
        temporary,
    )
    temporary.replace(destination)


def load_checkpoint(
    path: str | Path,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    generator_optimizer: torch.optim.Optimizer | None = None,
    discriminator_optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        if generator_optimizer is not None:
            generator_optimizer.load_state_dict(checkpoint["generator_optimizer"])
        if discriminator_optimizer is not None:
            discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer"])
    except (KeyError, RuntimeError, OSError, ValueError) as exc:
        raise ValueError(f"Checkpoint is invalid or incompatible: {checkpoint_path}") from exc
    return checkpoint
