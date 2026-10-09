import logging
import os
from contextlib import nullcontext
from typing import Any

import torch
from rats import apps
from torch import Tensor
from torch.amp.grad_scaler import GradScaler
from torchmetrics import MeanMetric
from wandb.wandb_run import Run

from trainer.config import BaseConfig
from trainer.datasets import DatasetServices, TrainIteratorContainer, ValDataloaderContainer
from trainer.models import ModelContainer, ModelServices
from trainer.state import StateContainer, StateServices
from trainer.utils.device import distributed_grad_sync, get_peak_flops, to_device
from trainer.utils.randomness import rng_state_from
from trainer.utils.timer import Timer

DeviceLikeType = str | torch.device | int
logger = logging.getLogger(__name__)


def log_on_main_worker(message: str, *args: Any, level: int = logging.INFO):
    """Logs a message only on the worker with number (rank) 0."""
    if int(os.getenv("RANK", "0")) == 0:
        logger.log(level, message, *args)


def parse_loss(args: Tensor | tuple[Tensor, Tensor]) -> Tensor:
    """Parses the loss from model output."""
    if isinstance(args, Tensor):
        return args
    elif isinstance(args, tuple) and len(args) == 2:
        assert isinstance(args[1], Tensor)
        return args[1]
    raise ValueError("Could not parse loss from model output")


class TrainContainer(apps.Container):
    _config: BaseConfig
    _state_container: StateContainer
    _wandb_run: Run | None

    def __init__(
        self,
        config: BaseConfig,
        model_container: ModelContainer,
        state_container: StateContainer,
        train_container: TrainIteratorContainer,
        val_container: ValDataloaderContainer,
        wandb_run: Run | None,
    ) -> None:
        self._config = config
        self._model_container = model_container
        self._state_container = state_container
        self._train_container = train_container
        self._val_container = val_container
        self._wandb_run = wandb_run
        self._val_losses = MeanMetric(sync_on_compute=True, compute_with_cache=True)

    @apps.container()
    def _st_container(self) -> StateContainer:
        return self._state_container

    def train(self):
        rank = int(os.getenv("RANK", "0"))
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(self._config.seed + rank)  # this controls multi-gpu randomness

        sequence_len = self._config.model.sequence_len
        grad_clip = self._config.training.grad_clip
        batch_size = self._config.training.batch_size  # micro batch size
        accelerator = self._config.compute_env.accelerator
        eval_interval = self._config.training.eval_interval
        ckpt_interval = self._config.training.checkpoint_interval
        use_fsdp = self._config.training.strategy in {"fsdp", "fsdp2"}
        non_blocking = self._config.data.pin_memory and torch.cuda.is_available()
        precision = self._config.training.precision
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[precision]
        output_dir = self._config.compute_env.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        ckpt_enabled = ckpt_interval > 0 and (world_size == 1 or rank == 0 or use_fsdp)
        ckpt_path = self._config.training.checkpoint_path
        ckpt_path.mkdir(parents=True, exist_ok=True)

        # modify gradient accumulation depending on world_size
        assert self._config.training.gradient_accumulation_steps % world_size == 0
        gradient_acc_steps = self._config.training.gradient_accumulation_steps  # whole world
        gradient_acc_steps = gradient_acc_steps // world_size  # per device
        assert gradient_acc_steps >= 1

        self._model_container.get(ModelServices.ATTENTION_BACKEND)  # logs SDPA backend info
        num_params = self._model_container.get(ModelServices.NUM_PARAMETERS)
        flops_per_token = self._model_container.get(ModelServices.FLOPS_PER_TOKEN)
        tokens_bs = batch_size * gradient_acc_steps * sequence_len * world_size
        flops_per_fwdbwd = tokens_bs * flops_per_token  # per iteration (all gpus)
        gpu_peak_flops = get_peak_flops(device_name=torch.cuda.get_device_name())  # device flops
        if self._wandb_run is not None:
            d = {
                "experiment": {
                    "num_parameters": num_params,
                    "token_bs": tokens_bs,
                    "flops_per_token": flops_per_token,
                    "flops_per_fwdbwd": flops_per_fwdbwd,
                    "gpu_peak_flops": gpu_peak_flops,
                    "world_size": world_size,
                }
            }
            self._wandb_run.config.update(d)

        # Log training info
        log_on_main_worker(f"Optimizer: {self._config.optimizer.name}")
        log_on_main_worker(f"GPUs: {world_size}")
        log_on_main_worker(f"FLOPs per token: {flops_per_token / 1e9:.2f}G")
        log_on_main_worker(f"Tokens per iteration: {tokens_bs:,}")

        # Initialize training state
        timer = Timer()
        EMA_BETA, tps_ema = 0.9, None  # tokens/sec EMA
        self._val_losses.to(device)
        self._val_losses.update(float("inf"))
        best_val_loss = float("inf")

        with rng_state_from("init_model", self._config.seed, devices=range(world_size)):  # synced
            state = self._state_container.get(StateServices.TRAINING_STATE)
        model, optimizers, schedulers = state.model, list(state.optimizers), list(state.schedulers)
        scaler = GradScaler(accelerator, enabled=precision == "float16")
        train_iterator = self._train_container.get(DatasetServices.TRAIN_ITERATOR)
        val_dataloader = self._val_container.get(DatasetServices.VAL_DATALOADER)

        # Training loop
        print(f"Starting training... (rank={rank}, device={device})")
        while True:
            # Evaluate
            if state.step % eval_interval == 0:
                model.eval()
                self._val_losses.reset()
                for data in val_dataloader:  # eval_iters
                    X_val, Y_val = to_device(data, device, non_blocking=non_blocking)
                    with torch.no_grad():
                        with nullcontext() if use_fsdp else torch.autocast(accelerator, dtype):
                            val_loss = parse_loss(model(X_val, Y_val))
                        self._val_losses.update(val_loss)
                val_loss = self._val_losses.compute()
                log_on_main_worker(f"step {state.step}: val loss {val_loss:.4f}")
                if best_val_loss > val_loss:
                    best_val_loss = val_loss

                model.train()

            # Checkpoint saving
            val_loss = self._val_losses.compute()
            if ckpt_enabled and state.step > 0 and state.step % ckpt_interval == 0:
                fname = f"ckpt-{state.step:09d}-val_{val_loss:.4f}-bestval_{best_val_loss:.4f}.pt"
                state.save_checkpoint(ckpt_path / fname)

            # Forward/backward pass
            with timer("fwdbwd_acc", sync=True):
                for micro_step in range(gradient_acc_steps):
                    is_last = micro_step == gradient_acc_steps - 1
                    with timer("micro_step", sync=True):
                        X, Y = next(train_iterator)
                        X, Y = to_device([X, Y], device, non_blocking=non_blocking)
                        with distributed_grad_sync(model, sync=is_last):
                            with nullcontext() if use_fsdp else torch.autocast(accelerator, dtype):
                                loss = parse_loss(model(X, Y))
                                loss = loss / gradient_acc_steps
                            scaler.scale(loss).backward()

            # Gradient clipping and optimizer step
            if grad_clip != 0.0:
                for opt in optimizers:
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # Optimizer step and zero grads
            with timer("optimizers", sync=True):
                for opt in optimizers:
                    scaler.step(opt)
                scaler.update()

                for opt in optimizers:
                    opt.zero_grad(set_to_none=True)

            for sch in schedulers:
                sch.step()

            if state.step % self._config.training.log_interval == 0:
                lossf = loss.item() * gradient_acc_steps  # type: ignore
                if self._wandb_run is not None:
                    lr = optimizers[0].param_groups[0]["lr"]
                    d: dict[str, Any] = {"step": state.step, "train/loss": lossf, "lr": lr}
                    if state.step % eval_interval == 0:
                        d["val/loss"] = val_loss
                    self._wandb_run.log(d, step=state.step)

                if state.step > 0:
                    df = timer.get_summary()
                    fwdbwd_t = df.loc[df["name"] == "fwdbwd_acc", "mean_time_s"].iloc[0]
                    fwdbwd_frc = df.loc[df["name"] == "fwdbwd_acc", "fraction_of_runtime"].iloc[0]
                    mstep_t = df.loc[df["name"] == "fwdbwd_acc.micro_step", "mean_time_s"].iloc[0]
                    opt_t = df.loc[df["name"] == "optimizers", "mean_time_s"].iloc[0]
                    opt_frc = df.loc[df["name"] == "optimizers", "fraction_of_runtime"].iloc[0]
                    dt = fwdbwd_t + opt_t
                    tokens_per_sec = tokens_bs / max(dt, 1e-9)
                    flops_per_sec = tokens_per_sec * flops_per_token
                    mfu = 100 * flops_per_fwdbwd / (gpu_peak_flops * world_size)
                    time_left = (self._config.training.max_iters - state.step) * dt
                    if tps_ema is None:
                        tps_ema = tokens_per_sec
                    tps_ema = EMA_BETA * tps_ema + (1.0 - EMA_BETA) * tokens_per_sec
                    time_for_1e12_days = 1e12 / tps_ema / 86400.0
                    log_on_main_worker(
                        f"iter {state.step}: "
                        + f"loss {lossf}, "
                        + f"dt {dt * 1000:.2f}ms, "
                        + f"fwdbwd {fwdbwd_t * 1000:.2f}ms ({100 * fwdbwd_frc:.2f}%), "
                        + f"microstep {mstep_t * 1000:.2f}ms, "
                        + f"opt {opt_t * 1000:.2f}ms ({100 * opt_frc:.2f}%), "
                        + f"tokens/s {tokens_per_sec:,.0f}, "
                        + f"TFLOPs/s {flops_per_sec / 1e12:.2f}, "
                        + f"ETA(1T) {time_for_1e12_days:.2f} days, "
                        + f"MFU {mfu:.2f}%, "
                        + f"time left {time_left / 3600:.2f} hrs"
                    )
                    if self._wandb_run is not None:
                        d = dict(zip("time/" + df["name"], df["mean_time_s"], strict=True))
                        d["mfu"] = mfu
                        d["throughput/TFLOPs_per_sec"] = flops_per_sec / 1e12
                        d["throughput/tokens_per_sec"] = tokens_per_sec
                        d["throughput/tokens_per_sec_ema"] = tps_ema
                        d["eta/days_for_1e12_tokens_ema"] = time_for_1e12_days
                        d["eta/hrs_left_for_training"] = time_left / 3600.0
                        self._wandb_run.log(d, step=state.step)

                timer.reset()

            state.step += 1
            if state.step > self._config.training.max_iters:
                break

        log_on_main_worker("Training completed!")
