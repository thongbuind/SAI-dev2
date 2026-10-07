from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
import torch._dynamo
import logging
import json
import gc
import argparse
import math
import time
from src.utils.utils import (
    DEFAULT_MAX_MICROBATCH_TOKENS, VRAM_PROFILE_STEPS,
    get_adaptive_microbatch_shape, cuda_memory_gib, print_vram_profile,
    get_step_lr_lambda, log_progress, load_checkpoint, save_checkpoint,
    estimate_training_flops, print_training_flops_summary,
    TflopsBenchmarker, KernelLogger,
)
from src.utils.chunked_loss import chunked_lm_loss
from src.data.Dataset import Dataset, split_train_val_test, load_data
from src.model.TransformerModel import TransformerModel

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Model size: 100M or 500M")
parser.add_argument(
    "--phase", type=str, required=True,
    choices=["pretrain", "pretrain_resume", "continued_pretrain", "continued_pretrain_resume", "full"],
    help="Training phase: pretrain | pretrain_resume | continued_pretrain | continued_pretrain_resume | full"
)
parser.add_argument(
    "--profile-kernels", action="store_true",
    help="Chỉ chạy 1 lần duy nhất, log tên kernel CPU/CUDA tại 1 step rồi thôi (dùng để debug performance)"
)
parser.add_argument(
    "--limited-max-autotune", action="store_true",
    help="Bật max-autotune giới hạn cho forward_features (chỉ ATEN/TRITON, không exhaustive/CUDA Graph)"
)
parser.add_argument(
    "--profile-vram", action="store_true",
    help="Đo allocated/reserved/peak VRAM theo từng pha trong các batch đầu"
)
args = parser.parse_args()
torch._dynamo.config.cache_size_limit = 24
logging.getLogger("torch._inductor.select_algorithm").setLevel(logging.ERROR)
model_size = args.model

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
config_dir = project_root / "config"
data_dir = project_root / "data"
model_dir = project_root / "model"
src_dir = project_root / "src"
base_config_file = config_dir / "base.json"
model_config_file = config_dir / f"{args.model}.json"
model_dir.mkdir(parents=True, exist_ok=True)
data_processed_dir = project_root / "data" / "processed"
pretrain_tokenized_file = data_processed_dir / "pretrain_manifest.json"
continued_pretrain_tokenized_file = data_processed_dir / "continued_pretrain_data_ids.npz"

pretrained_save_path = model_dir / f"pretrained_{model_size}.pt"
pretrained_ckpt_path = model_dir / f"pretrained_{model_size}.ckpt.pt"
continued_pretrained_save_path = model_dir / f"continued_pretrained_{model_size}.pt"
continued_pretrained_ckpt_path = model_dir / f"continued_pretrained_{model_size}.ckpt.pt"

LM_LOSS_CHUNK_SIZE = 4096
CHECKPOINT_EVERY_STEPS = 5000
EVAL_LOG_EVERY = 100

def train_loop(data_type, tokenized_file, epochs, learning_rate, weight_decay, num_workers, extra_file=None, model_save_path=None, resume_checkpoint_path=None, profile_kernels=False):
    print("╠════════════════════════════════════════════════════════════════════════════════════╣")
    print("║                                BAT ĐAU LOAD DATA                                   ║")
    print("╠════════════════════════════════════════════════════════════════════════════════════╣")

    if extra_file is None:
        X, Y, lengths = load_data(data_type, tokenized_file)
    else:
        X, Y, lengths = load_data(data_type, tokenized_file, extra_file)

    X_train, Y_train, _, lengths_train, X_val, Y_val, _, lengths_val, X_test, Y_test, _, lengths_test = split_train_val_test(X, Y, None, lengths, train_ratio, val_ratio)
    log_progress(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    mean_train_tokens = float(lengths_train.sum()) / max(1, len(lengths_train))
    target_tokens_per_logical_batch = mean_train_tokens * batch_size

    train_ds = Dataset.create_dataloader(
        X_train, Y_train, lengths_train, batch_size, max_seq_len, num_workers,
        shuffle=True, mode="pretrain",
    )
    val_ds = Dataset.create_dataloader(
        X_val, Y_val, lengths_val, batch_size, max_seq_len, num_workers,
        shuffle=False, mode="pretrain",
    )
    test_ds = Dataset.create_dataloader(
        X_test, Y_test, lengths_test, batch_size, max_seq_len, num_workers,
        shuffle=False, mode="pretrain",
    )

    del X_train, Y_train, lengths_train, X_val, Y_val, lengths_val, X_test, Y_test, lengths_test
    gc.collect()
    log_progress(
        f"Adaptive microbatch: max {max_microbatch_tokens:,} padded token-slot, "
        f"logical batch_size={batch_size}"
    )
    log_progress(
        f"Token-normalized gradient: target={target_tokens_per_logical_batch:,.1f} "
        f"valid token/logical batch"
    )

    global optimizer, scaler
    optimizer = optim.AdamW(model_raw.parameters(), lr=learning_rate, weight_decay=weight_decay, fused=True)

    total_steps = math.ceil(len(train_ds) / accumulation_steps) * epochs
    warmup_steps = total_steps // 10

    lr_lambda = get_step_lr_lambda(warmup_steps, total_steps)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_loss = float('inf')
    global_step = 0
    start_epoch = 0
    resume_batch_idx = 0
    sampler_state = None

    checkpoint_path = model_save_path.with_suffix(".ckpt.pt") if model_save_path is not None else None
    if resume_checkpoint_path is not None:
        if resume_checkpoint_path.exists():
            start_epoch, global_step, best_val_loss, resume_batch_idx, sampler_state = load_checkpoint(
                resume_checkpoint_path, model_raw, optimizer, scheduler, device
            )
        else:
            log_progress(f"[WARNING] Checkpoint not found at {resume_checkpoint_path}. Starting from scratch.")

    flops_info = estimate_training_flops(model=model_raw, num_layers=num_layers, max_seq_len=max_seq_len, d_model=d_model, epochs=epochs, batches_per_epoch=len(train_ds), batch_size=batch_size)
    print_training_flops_summary(flops_info, epochs)
    bench = TflopsBenchmarker(flops_per_token=flops_info["flops_per_token"], total_flops_needed=flops_info["total_flops"], total_batches_per_epoch=flops_info["total_batches_per_epoch"], epochs=epochs, device=device, warmup_steps=500, target_steps=1000)
    kernel_logger = KernelLogger(enabled=profile_kernels, log_step=1000)
    if resume_batch_idx > 0:
        bench.reported = True

    for epoch in range(start_epoch, epochs):
        skip_batches = resume_batch_idx if epoch == start_epoch else 0
        if skip_batches > 0:
            train_ds.batch_sampler.seed = sampler_state["seed"]
            train_ds.batch_sampler.epoch = sampler_state["epoch"]
        sampler_epoch = train_ds.batch_sampler.epoch
        model_raw.train()
        train_loss_sum = torch.zeros((), device=device)
        train_valid_tokens = torch.zeros((), device=device)
        total_batches = len(train_ds)
        optimizer.zero_grad()

        for batch_idx, (X_batch, Y_batch, sample_weight, attention_mask, has_padding) in enumerate(train_ds):
            if batch_idx < skip_batches:
                continue
            profile_vram = (
                args.profile_vram
                and device.type == "cuda"
                and epoch == start_epoch
                and batch_idx < VRAM_PROFILE_STEPS
            )
            vram_marks = {}
            if profile_vram:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                vram_marks["start"] = cuda_memory_gib()

            X_batch = X_batch.to(device, non_blocking=True)
            Y_batch = Y_batch.to(device, non_blocking=True)
            sample_weight = sample_weight.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            batch_shape = tuple(X_batch.shape)
            if profile_vram:
                vram_marks["h2d"] = cuda_memory_gib()

            _is_bench_step = bench.is_bench_step(epoch, start_epoch, batch_idx)
            if _is_bench_step:
                bench.on_step_begin(batch_idx, X_batch.shape[0] * X_batch.shape[1])

            _is_kernel_log = kernel_logger.should_log(epoch, start_epoch, batch_idx)
            if _is_kernel_log:
                kernel_logger.start()

            logical_batch_size, padded_length = batch_shape
            microbatch_count, microbatch_size = get_adaptive_microbatch_shape(
                logical_batch_size, padded_length, max_microbatch_tokens
            )

            num_valid_tokens = sample_weight.sum()
            logical_loss_sum = torch.zeros((), device=device, dtype=torch.float32)

            for micro_start in range(0, logical_batch_size, microbatch_size):
                micro_end = min(micro_start + microbatch_size, logical_batch_size)
                X_micro = X_batch[micro_start:micro_end]
                Y_micro = Y_batch[micro_start:micro_end]
                weight_micro = sample_weight[micro_start:micro_end]
                attention_micro = attention_mask[micro_start:micro_end]

                with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    hidden = model_features(
                        X_micro,
                        attention_mask=attention_micro,
                        has_padding=has_padding,
                    )
                    if profile_vram:
                        forward_mark = cuda_memory_gib()
                        if (
                            "forward" not in vram_marks
                            or forward_mark["allocated"] > vram_marks["forward"]["allocated"]
                        ):
                            vram_marks["forward"] = forward_mark

                    hidden_flat = hidden.reshape(-1, hidden.size(-1))
                    targets_flat = Y_micro.reshape(-1)
                    weight_flat = weight_micro.reshape(-1)
                    loss_sum = chunked_lm_loss(
                        hidden_flat,
                        model_raw.lm_head.weight,
                        targets_flat,
                        weight_flat,
                        chunk_size=LM_LOSS_CHUNK_SIZE,
                    )
                    if profile_vram:
                        loss_mark = cuda_memory_gib()
                        if (
                            "loss" not in vram_marks
                            or loss_mark["allocated"] > vram_marks["loss"]["allocated"]
                        ):
                            vram_marks["loss"] = loss_mark

                logical_loss_sum += loss_sum.detach()
                scaled_loss = (
                    loss_sum / target_tokens_per_logical_batch / accumulation_steps
                )
                scaler.scale(scaled_loss).backward()

                del X_micro, Y_micro, weight_micro, attention_micro
                del hidden, hidden_flat, targets_flat, weight_flat
                del loss_sum, scaled_loss

            if profile_vram:
                vram_marks["backward"] = cuda_memory_gib()

            if _is_kernel_log:
                kernel_logger.stop_and_report(batch_idx, epoch, device)

            optimizer_updated = (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == total_batches
            if optimizer_updated:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model_raw.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                if global_step % CHECKPOINT_EVERY_STEPS == 0 and checkpoint_path is not None:
                    save_checkpoint(
                        checkpoint_path, epoch, global_step,
                        model_raw, optimizer, scheduler, best_val_loss,
                        batches_done=batch_idx + 1,
                        sampler_state={"seed": train_ds.batch_sampler.seed, "epoch": sampler_epoch},
                    )
                    print()
                    log_progress(f"Step {global_step}: checkpoint saved → {checkpoint_path}")
            if profile_vram:
                vram_marks["update"] = cuda_memory_gib()

            train_loss_sum += logical_loss_sum
            train_valid_tokens += num_valid_tokens.detach()
            current_lr = optimizer.param_groups[0]['lr']

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                avg_loss = (
                    train_loss_sum / (train_valid_tokens + 1e-8)
                ).item()
                vram_text = ""
                if device.type == "cuda":
                    gib = 1024 ** 3
                    current_vram = torch.cuda.memory_reserved() / gib
                    peak_vram = torch.cuda.max_memory_reserved() / gib
                    vram_text = f" - VRAM: current={current_vram:.2f}GiB peak={peak_vram:.2f}GiB"
                print(
                    f"\rEpoch {epoch+1}/{epochs} - Step {global_step}/{total_steps} "
                    f"- loss: {avg_loss:.4f} - lr: {current_lr:.2e} "
                    f"- micro={microbatch_count}x<={microbatch_size}{vram_text}",
                    end='',
                )

            if _is_bench_step:
                bench.on_step_end(batch_idx)

            del X_batch, Y_batch, sample_weight, attention_mask
            del num_valid_tokens, logical_loss_sum
            if profile_vram:
                vram_marks["released"] = cuda_memory_gib()
                print_vram_profile(
                    batch_idx, batch_shape, has_padding, optimizer_updated,
                    microbatch_count, microbatch_size, vram_marks,
                )

        print()

        train_loss = (
            train_loss_sum / (train_valid_tokens + 1e-8)
        ).item()
        model_raw.eval()
        val_loss_sum = torch.zeros((), device=device)
        val_valid_tokens = torch.zeros((), device=device)
        eval_start = time.time()

        with torch.no_grad():
            for eval_idx, (X_batch, Y_batch, sample_weight, attention_mask, has_padding) in enumerate(val_ds):
                X_batch = X_batch.to(device, non_blocking=True)
                Y_batch = Y_batch.to(device, non_blocking=True)
                sample_weight = sample_weight.to(device, non_blocking=True)
                attention_mask = attention_mask.to(device, non_blocking=True)
                logical_batch_size, padded_length = X_batch.shape
                _, microbatch_size = get_adaptive_microbatch_shape(
                    logical_batch_size, padded_length, max_microbatch_tokens
                )

                for micro_start in range(0, logical_batch_size, microbatch_size):
                    micro_end = micro_start + microbatch_size
                    with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                        hidden = model_features(X_batch[micro_start:micro_end], attention_mask=attention_mask[micro_start:micro_end], has_padding=has_padding)
                        hidden_flat = hidden.reshape(-1, hidden.size(-1))
                        targets_flat = Y_batch[micro_start:micro_end].reshape(-1)
                        weight_flat = sample_weight[micro_start:micro_end].reshape(-1)
                        loss_sum = chunked_lm_loss(
                            hidden_flat, model_raw.lm_head.weight, targets_flat, weight_flat,
                            chunk_size=LM_LOSS_CHUNK_SIZE,
                        )

                    val_loss_sum += loss_sum.detach()
                    val_valid_tokens += weight_flat.sum()

                    del hidden, hidden_flat, targets_flat, weight_flat, loss_sum

                if (eval_idx + 1) % EVAL_LOG_EVERY == 0 or (eval_idx + 1) == len(val_ds):
                    print(f"\rVal {eval_idx+1}/{len(val_ds)} - {time.time() - eval_start:.0f}s", end='', flush=True)

                del X_batch, Y_batch, sample_weight, attention_mask

        print()
        val_loss = (val_loss_sum / (val_valid_tokens + 1e-8)).item()
        log_progress(f"Epoch {epoch+1}/{epochs} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if model_save_path is not None:
                torch.save(model_raw.state_dict(), model_save_path)
                print(f"Epoch {epoch+1}: val_loss improved to {val_loss:.5f}, saving model to {model_save_path}")
            else:
                torch.save(model_raw.state_dict(), pretrained_save_path)
                print(f"Epoch {epoch+1}: val_loss improved to {val_loss:.5f}, saving model to default path")

        if checkpoint_path is not None:
            save_checkpoint(
                checkpoint_path, epoch, global_step,
                model_raw, optimizer, scheduler, best_val_loss
            )
            log_progress(f"Checkpoint saved → {checkpoint_path}")

    print("╠════════════════════════════════════════════════════════════════════════════════════╣")
    print("║                               DANH GIA TREN TEST SET                               ║")
    print("╠════════════════════════════════════════════════════════════════════════════════════╣")

    model_raw.eval()
    test_loss_sum = torch.zeros((), device=device)
    test_valid_tokens = torch.zeros((), device=device)
    eval_start = time.time()

    with torch.no_grad():
        for eval_idx, (X_batch, Y_batch, sample_weight, attention_mask, has_padding) in enumerate(test_ds):
            X_batch = X_batch.to(device, non_blocking=True)
            Y_batch = Y_batch.to(device, non_blocking=True)
            sample_weight = sample_weight.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            logical_batch_size, padded_length = X_batch.shape
            _, microbatch_size = get_adaptive_microbatch_shape(
                logical_batch_size, padded_length, max_microbatch_tokens
            )

            for micro_start in range(0, logical_batch_size, microbatch_size):
                micro_end = micro_start + microbatch_size
                with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    hidden = model_features(X_batch[micro_start:micro_end], attention_mask=attention_mask[micro_start:micro_end], has_padding=has_padding)
                    hidden_flat = hidden.reshape(-1, hidden.size(-1))
                    targets_flat = Y_batch[micro_start:micro_end].reshape(-1)
                    weight_flat = sample_weight[micro_start:micro_end].reshape(-1)
                    loss_sum = chunked_lm_loss(
                        hidden_flat, model_raw.lm_head.weight, targets_flat, weight_flat,
                        chunk_size=LM_LOSS_CHUNK_SIZE,
                    )

                test_loss_sum += loss_sum.detach()
                test_valid_tokens += weight_flat.sum()

                del hidden, hidden_flat, targets_flat, weight_flat, loss_sum

            if (eval_idx + 1) % EVAL_LOG_EVERY == 0 or (eval_idx + 1) == len(test_ds):
                print(f"\rTest {eval_idx+1}/{len(test_ds)} - {time.time() - eval_start:.0f}s", end='', flush=True)

            del X_batch, Y_batch, sample_weight, attention_mask

    print()
    test_loss = (test_loss_sum / (test_valid_tokens + 1e-8)).item()
    log_progress(f"Test Loss: {test_loss:.4f}")
    print("╠════════════════════════════════════════════════════════════════════════════════════╣")

    return test_loss

with open(base_config_file, 'r') as f:
    config = json.load(f)
with open(model_config_file, 'r') as f:
    config.update(json.load(f))

vocab_size = config['vocab_size']
max_seq_len = config['max_seq_len']
d_model = config['d_model']
num_heads = config['num_heads']
num_kv_heads = config['num_kv_heads']
num_layers = config['num_layers']
ff_dim = config['ff_dim']
dropout = config['dropout']
pretrain_epochs = config['pretrain_epochs']
continued_pretrain_epochs = config['continued_pretrain_epochs']
batch_size = config['batch_size']
train_ratio = config['train_ratio']
val_ratio = config['val_ratio']
pretrain_learning_rate = config['pretrain_learning_rate']
continued_pretrain_learning_rate = config['continued_pretrain_learning_rate']
accumulation_steps = config['accumulation_steps']
max_microbatch_tokens = config['max_microbatch_tokens']
pretrain_weight_decay = config['pretrain_weight_decay']
continued_pretrain_weight_decay = config['continued_pretrain_weight_decay']
num_workers = config['num_workers']

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

amp_enabled = device.type == "cuda"
amp_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16
scaler = GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)

if amp_enabled:
    precision_name = "BF16" if amp_dtype == torch.bfloat16 else "FP16 + GradScaler"
    log_progress(f"CUDA mixed precision: {precision_name}")

model_raw = TransformerModel(vocab_size, d_model, num_heads, num_kv_heads, num_layers, ff_dim, max_seq_len, dropout).to(device)
model = torch.compile(model_raw, dynamic=True)
if args.limited_max_autotune:
    compile_options = {
        "max_autotune": True,
        "max_autotune_gemm_backends": "ATEN,TRITON",
        "max_autotune_gemm_search_space": "DEFAULT",
        "epilogue_fusion": True,
        "triton.cudagraphs": False,
    }
    model_features = torch.compile(
        model_raw.forward_features,
        dynamic=True,
        options=compile_options,
    )
    log_progress("torch.compile: limited max-autotune (ATEN,TRITON; DEFAULT; dynamic=True; CUDA Graph off)")
else:
    model_features = torch.compile(model_raw.forward_features, dynamic=True)
optimizer = optim.AdamW(model_raw.parameters(), lr=pretrain_learning_rate, weight_decay=pretrain_weight_decay, fused=True)

print("╠════════════════════════════════════════════════════════════════════════════════════╣")
print("║                                 BAT ĐAU TRAINING                                   ║")
print("╠════════════════════════════════════════════════════════════════════════════════════╣")

phase = args.phase

if phase == "pretrain":
    pretrain_test_loss = train_loop(
        data_type="pretrain",
        tokenized_file=pretrain_tokenized_file,
        epochs=pretrain_epochs,
        learning_rate=pretrain_learning_rate,
        weight_decay=pretrain_weight_decay,
        num_workers=num_workers,
        model_save_path=pretrained_save_path,
        resume_checkpoint_path=None,
        profile_kernels=args.profile_kernels,
    )
    log_progress(f"Pretrain Test Loss: {pretrain_test_loss:.4f}")

elif phase == "pretrain_resume":
    log_progress("Pretrain resume: khởi tạo model skeleton trước khi load checkpoint...")
    pretrain_test_loss = train_loop(
        data_type="pretrain",
        tokenized_file=pretrain_tokenized_file,
        epochs=pretrain_epochs,
        learning_rate=pretrain_learning_rate,
        weight_decay=pretrain_weight_decay,
        num_workers=num_workers,
        model_save_path=pretrained_save_path,
        resume_checkpoint_path=pretrained_ckpt_path,
        profile_kernels=args.profile_kernels,
    )
    log_progress(f"Pretrain Test Loss: {pretrain_test_loss:.4f}")

elif phase == "continued_pretrain":
    log_progress("Load best model từ pretrain để tiếp tục training...")
    model_raw.load_state_dict(torch.load(pretrained_save_path, map_location=device))
    model_raw.to(device)
    optimizer = optim.AdamW(model_raw.parameters(), lr=continued_pretrain_learning_rate, weight_decay=continued_pretrain_weight_decay, fused=True)
    log_progress(f"Reset optimizer với learning rate: {continued_pretrain_learning_rate}")

    continued_pretrain_test_loss = train_loop(
        data_type="continued_pretrain",
        tokenized_file=continued_pretrain_tokenized_file,
        epochs=continued_pretrain_epochs,
        learning_rate=continued_pretrain_learning_rate,
        weight_decay=continued_pretrain_weight_decay,
        num_workers=num_workers,
        extra_file=pretrain_tokenized_file,
        model_save_path=continued_pretrained_save_path,
        resume_checkpoint_path=None,
        profile_kernels=args.profile_kernels,
    )
    log_progress(f"Continued Pretrain Test Loss: {continued_pretrain_test_loss:.4f}")

elif phase == "continued_pretrain_resume":
    log_progress("Continued pretrain resume: khởi tạo model skeleton trước khi load checkpoint...")
    optimizer = optim.AdamW(model_raw.parameters(), lr=continued_pretrain_learning_rate, weight_decay=continued_pretrain_weight_decay, fused=True)

    continued_pretrain_test_loss = train_loop(
        data_type="continued_pretrain",
        tokenized_file=continued_pretrain_tokenized_file,
        epochs=continued_pretrain_epochs,
        learning_rate=continued_pretrain_learning_rate,
        weight_decay=continued_pretrain_weight_decay,
        num_workers=num_workers,
        extra_file=pretrain_tokenized_file,
        model_save_path=continued_pretrained_save_path,
        resume_checkpoint_path=continued_pretrained_ckpt_path,
        profile_kernels=args.profile_kernels,
    )
    log_progress(f"Continued Pretrain Test Loss: {continued_pretrain_test_loss:.4f}")

elif phase == "full":
    pretrain_test_loss = train_loop(
        data_type="pretrain",
        tokenized_file=pretrain_tokenized_file,
        epochs=pretrain_epochs,
        learning_rate=pretrain_learning_rate,
        weight_decay=pretrain_weight_decay,
        num_workers=num_workers,
        model_save_path=pretrained_save_path,
        resume_checkpoint_path=None,
        profile_kernels=args.profile_kernels,
    )

    log_progress("Đang load best model từ pretrain để tiếp tục training...")
    model_raw.load_state_dict(torch.load(pretrained_save_path, map_location=device))
    model_raw.to(device)
    optimizer = optim.AdamW(model_raw.parameters(), lr=continued_pretrain_learning_rate, weight_decay=continued_pretrain_weight_decay, fused=True)
    log_progress(f"Reset optimizer với learning rate: {continued_pretrain_learning_rate}")

    continued_pretrain_test_loss = train_loop(
        data_type="continued_pretrain",
        tokenized_file=continued_pretrain_tokenized_file,
        epochs=continued_pretrain_epochs,
        learning_rate=continued_pretrain_learning_rate,
        weight_decay=continued_pretrain_weight_decay,
        num_workers=num_workers,
        extra_file=pretrain_tokenized_file,
        model_save_path=continued_pretrained_save_path,
        resume_checkpoint_path=None,
        profile_kernels=args.profile_kernels,
    )

    log_progress(f"Hoàn thành training!")
    log_progress(f"Pretrain Test Loss: {pretrain_test_loss:.4f}")
    log_progress(f"Continued Pretrain Test Loss: {continued_pretrain_test_loss:.4f}")
