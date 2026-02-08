from typing import Optional, Dict, Any, Tuple
from pathlib import Path

import torch

from model.ssd import SSD


class Checkpointer:

    def __init__(
        self,
        checkpoint_dir: str,
        monitor: str = "val_loss",
        mode: str = "min", 
        save_best_only: bool = True,
    ) -> None:
        assert mode in {"min", "max"}

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only

        self.best_score = float("inf") if mode == "min" else -float("inf")

    def _is_better(self, score: float) -> bool:
        if self.mode == "min":
            return score < self.best_score
        return score > self.best_score

    def save(
        self,
        epoch: int,
        metrics: Dict[str, float],
        model: SSD,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ) -> None:
        score = metrics.get(self.monitor)
        assert score is not None, f"Metric '{self.monitor}' not found in metrics"

        is_best = self._is_better(score)

        if self.save_best_only and not is_best:
            return

        if is_best:
            self.best_score = score

        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "model_config": getattr(model, "config", None),
            "optimizer_state": optimizer.state_dict() if optimizer else None,
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "metrics": metrics,
            "best_score": self.best_score,
        }

        last_path = self.checkpoint_dir / "last.pt"
        torch.save(state, last_path)

        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(state, best_path)

    def load(
        self,
        name: str = "best", 
        model: Optional[SSD] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: torch.device = torch.device("cpu"),
        strict: bool = True,
    ) -> Tuple[Optional[SSD], int, Dict[str, float]]:

        checkpoint_path = self.checkpoint_dir / f"{name}.pt"
        assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

        self.best_score = checkpoint.get("best_score", self.best_score)

        if model is None:
            model_config = checkpoint.get("model_config")
            assert model_config is not None, "model_config not found in checkpoint"
            model = SSD(**model_config)

        model.load_state_dict(
            checkpoint["model_state"],
            strict=strict,
        )
        model.to(device)

        if optimizer and checkpoint["optimizer_state"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            self._move_optimizer_to_device(optimizer, device)

        if scheduler and checkpoint["scheduler_state"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])

        epoch = checkpoint["epoch"]
        metrics = checkpoint["metrics"]

        return model, epoch, metrics

    @staticmethod
    def _move_optimizer_to_device(
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ):
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)