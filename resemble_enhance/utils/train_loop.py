import json
import logging
import time
from dataclasses import KW_ONLY, dataclass
from pathlib import Path
from typing import Protocol

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .control import non_blocking_input
from .distributed import is_global_leader
from .engine import Engine
from .utils import tree_map
from ..data import create_dataloaders
from ..hparams import HParams


logger = logging.getLogger(__name__)

import wandb
wandb_logger = wandb.init(project="resemble-enhance-denoiser", job_type="train")


class EvalFn(Protocol):
    def __call__(self, engine: Engine, eval_dir: Path) -> None:
        ...


class EngineLoader(Protocol):
    def __call__(self, run_dir: Path) -> Engine:
        ...


class GenFeeder(Protocol):
    def __call__(self, engine: Engine, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        ...


class DisFeeder(Protocol):
    def __call__(self, engine: Engine, batch: dict[str, Tensor] | None, fake: Tensor) -> dict[str, Tensor]:
        ...


@dataclass
class TrainLoop:
    _ = KW_ONLY

    run_dir: Path
    train_dl: DataLoader

    load_G: EngineLoader
    feed_G: GenFeeder
    load_D: EngineLoader | None = None
    feed_D: DisFeeder | None = None

    # Set update_every and eval_every to be the same
    update_every: int = 5_000
    eval_every: int = update_every
    backup_steps: tuple[int, ...] = (5_000, 100_000, 500_000)
    # backup_steps: tuple[int, ...] = (20, 30, 40)

    hp: HParams  = None
    refresh_train_dl_every: int = -1

    device: str = "cuda"
    eval_fn: EvalFn | None = None
    gan_training_start_step: int | None = None

    @property
    def global_step(self):
        return self.engine_G.global_step  # How many steps have been completed?

    @property
    def eval_dir(self) -> Path | None:
        if self.eval_every != 0:
            eval_dir = self.run_dir.joinpath("eval")
            eval_dir.mkdir(exist_ok=True)
        else:
            eval_dir = None
        return eval_dir

    @property
    def viz_dir(self) -> Path:
        return Path(self.run_dir / "viz")

    def make_current_step_viz_path(self, name: str, suffix: str) -> Path:
        path = (self.viz_dir / name / f"{self.global_step}").with_suffix(suffix)
        path.parent.mkdir(exist_ok=True, parents=True)
        return path

    def __post_init__(self):
        engine_G = self.load_G(self.run_dir)
        if self.load_D is None:
            engine_D = None
        else:
            engine_D = self.load_D(self.run_dir)
        self.engine_G = engine_G
        self.engine_D = engine_D

        self.early_stopper = EarlyStopper(patience=10, min_delta=0.0001)

    @property
    def model_G(self):
        return self.engine_G.module

    @property
    def model_D(self):
        if self.engine_D is None:
            return None
        return self.engine_D.module

    def save_checkpoint(self, tag="default"):
        engine_G = self.engine_G
        engine_D = self.engine_D
        engine_G.save_checkpoint(tag=tag)
        if engine_D is not None:
            engine_D.save_checkpoint(tag=tag)

    def run(self, max_steps: int = -1):
        self.set_running_loop_(self)

        train_dl = self.train_dl
        update_every = self.update_every
        eval_every = self.eval_every
        device = self.device
        eval_fn = self.eval_fn

        engine_G = self.engine_G
        engine_D = self.engine_D
        eval_dir = self.eval_dir

        init_step = self.global_step

        logger.info(f"\nTraining from step {init_step} to step {max_steps}")
        warmup_steps = {init_step + x for x in [50, 100, 500]}

        engine_G.train()

        if engine_D is not None:
            engine_D.train()

        gan_start_step = self.gan_training_start_step

        should_stop = False
        epoch_counter = 0
        while True:
            loss_G = loss_D = 0

            if self.refresh_train_dl_every > 0 and epoch_counter >= self.refresh_train_dl_every:
                logger.info("Refreshing the dataloader")
                # Reset the dataloader
                epoch_counter = 0
                train_dl, _ = create_dataloaders(self.hp, mode="denoiser")

            for batch in train_dl:
                torch.cuda.synchronize()
                start_time = time.time()

                # What's the step after this batch?
                step = self.global_step + 1

                # Send data to the GPU
                batch = tree_map(lambda x: x.to(device) if isinstance(x, Tensor) else x, batch)

                stats = {"step": step}

                # Include step == 1 for sanity check
                gan_started = gan_start_step is not None and (step >= gan_start_step or step == 1)
                gan_started &= engine_D is not None

                # Generator step
                fake, losses = self.feed_G(engine=engine_G, batch=batch)

                # Train generator
                if gan_started:
                    assert engine_D is not None
                    assert self.feed_D is not None

                    # Freeze the discriminator to let gradient go through fake
                    engine_D.freeze_()
                    losses |= self.feed_D(engine=engine_D, batch=None, fake=fake)

                loss_G = sum(losses.values())
                stats |= {f"G/{k}": v.item() for k, v in losses.items()}
                stats |= {f"G/{k}": v for k, v in engine_G.gather_attribute("stats").items()}
                del losses

                assert isinstance(loss_G, Tensor)
                stats["G/loss"] = loss_G.item()
                stats["G/lr"] = engine_G.get_lr()[0]
                g_grad_norm = engine_G.get_grad_norm()
                
                if g_grad_norm:
                    stats["G/grad_norm"] = g_grad_norm.cpu().item()
                else:
                    stats["G/grad_norm"] = 0

                if loss_G.isnan().item():
                    logger.error("Generator loss is NaN, skipping step")
                    continue

                engine_G.backward(loss_G)
                engine_G.step()

                # Discriminator step
                if gan_started:
                    assert engine_D is not None
                    assert self.feed_D is not None

                    engine_D.unfreeze_()
                    losses = self.feed_D(engine=engine_D, batch=batch, fake=fake.detach())
                    del fake

                    assert isinstance(losses, dict)
                    loss_D = sum(losses.values())
                    assert isinstance(loss_D, Tensor)

                    stats |= {f"D/{k}": v.item() for k, v in losses.items()}
                    stats |= {f"D/{k}": v for k, v in engine_D.gather_attribute("stats").items()}
                    del losses

                    if loss_D.isnan().item():
                        logger.error("Discriminator loss is NaN, skipping step")
                        continue

                    engine_D.backward(loss_D)
                    engine_D.step()

                    stats["D/loss"] = loss_D.item()
                    stats["D/lr"] = engine_D.get_lr()[0]
                    d_grad_norm = engine_D.get_grad_norm()

                    if d_grad_norm:
                        stats["D/grad_norm"] = d_grad_norm.cpu().item()
                    else:
                        stats["D/grad_norm"] = 0
                   
                torch.cuda.synchronize()
                stats["elapsed_time"] = time.time() - start_time
                stats = tree_map(lambda x: float(f"{x:.4g}") if isinstance(x, float) else x, stats)
                loss_l1 = stats["G/losses/l1"]
                loss = stats["G/loss"]

                # if step % 1 == 0:
                #     wandb_logger.log({"generator - denoiser/loss": loss, "generator - denoiser/l1": loss_l1})
                # print(stats)
                logger.info(json.dumps(stats, indent=0))

                command = non_blocking_input()

                evaling = step % eval_every == 0 or step in warmup_steps or command.strip() == "eval"
                if eval_fn is not None and is_global_leader() and eval_dir is not None and evaling:
                    engine_G.eval()
                    avg_si_snr_score, avg_eval_loss = eval_fn(engine_G, eval_dir=eval_dir)

                    # Log the si-snr score to wandb
                    wandb_logger.log({"si-snr score": avg_si_snr_score, "eval loss": avg_eval_loss})

                    should_stop = self.early_stopper.should_early_stop(avg_eval_loss)

                    engine_G.train()

                if command.strip() == "quit":
                    logger.info("Training paused")
                    self.save_checkpoint("default")
                    return

                if command.strip() == "backup" or step in self.backup_steps:
                    logger.info("Backing up")
                    self.save_checkpoint(tag=f"backup_{step:07d}")

                if step % update_every == 0 or command.strip() == "save" or should_stop:
                    if should_stop:
                        logger.info("Early stopping")

                    print("Saving checkpoint")
                    self.save_checkpoint(tag="default")

                if step == max_steps:
                    logger.info("Training finished")
                    self.save_checkpoint(tag="default")
                    return

            wandb_logger.log({"Epoch - G/denoiser/loss": loss, "Epoch - G/denoiser/l1": loss_l1})
            epoch_counter += 1

    @classmethod
    def set_running_loop_(cls, loop):
        assert isinstance(loop, cls), f"Expected {cls}, got {type(loop)}"
        cls._running_loop: cls = loop

    @classmethod
    def get_running_loop(cls) -> "TrainLoop | None":
        if hasattr(cls, "_running_loop"):
            assert isinstance(cls._running_loop, cls)
            return cls._running_loop
        return None

    @classmethod
    def get_running_loop_global_step(cls) -> int | None:
        if loop := cls.get_running_loop():
            return loop.global_step
        return None

    @classmethod
    def get_running_loop_viz_path(cls, name: str, suffix: str) -> Path | None:
        if loop := cls.get_running_loop():
            return loop.make_current_step_viz_path(name, suffix)
        return None

class EarlyStopper:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def should_early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False