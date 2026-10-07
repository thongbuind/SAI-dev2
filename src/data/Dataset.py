import torch
import numpy as np
import gc
import json
from pathlib import Path
from src.utils.utils import log_progress

class RandomizedSortishBatchSampler(torch.utils.data.Sampler):
    def __init__(self, lengths, batch_size, pool_batches=50, drop_last=False, seed=None):
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.pool_size = max(batch_size, batch_size * pool_batches)
        self.drop_last = drop_last
        self.seed = seed if seed is not None else int(
            torch.empty((), dtype=torch.int64).random_().item()
        )
        self.epoch = 0

    def __len__(self):
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return int(np.ceil(len(self.lengths) / self.batch_size))

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        shuffled_indices = rng.permutation(len(self.lengths))
        self.epoch += 1

        for pool_start in range(0, len(shuffled_indices), self.pool_size):
            pool = shuffled_indices[pool_start:pool_start + self.pool_size]
            length_order = np.argsort(self.lengths[pool], kind="stable")
            sorted_pool = pool[length_order]

            pool_batches = []
            for batch_start in range(0, len(sorted_pool), self.batch_size):
                batch = sorted_pool[batch_start:batch_start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batch = batch.tolist()
                rng.shuffle(batch)
                pool_batches.append(batch)

            rng.shuffle(pool_batches)
            yield from pool_batches

class TokenBalancedSortishBatchSampler(torch.utils.data.Sampler):
    def __init__(
        self, lengths, loss_token_counts, batch_size, pool_batches=50,
        max_samples=None, seed=None,
    ):
        self.lengths = np.asarray(lengths)
        self.loss_token_counts = np.asarray(loss_token_counts, dtype=np.int64)
        self.pool_size = max(batch_size, batch_size * pool_batches)
        self.max_samples = max_samples or batch_size * 4
        self.target_loss_tokens = max(
            1, int(round(self.loss_token_counts.mean() * batch_size))
        )
        self.seed = seed if seed is not None else int(
            torch.empty((), dtype=torch.int64).random_().item()
        )
        self.epoch = 0
        self._cached_epoch = None
        self._cached_batches = None

    def _build_batches(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        shuffled_indices = rng.permutation(len(self.lengths))
        batches = []
        current_batch = []
        current_loss_tokens = 0

        for pool_start in range(0, len(shuffled_indices), self.pool_size):
            pool = shuffled_indices[pool_start:pool_start + self.pool_size]
            length_order = np.argsort(self.lengths[pool], kind="stable")
            for idx in pool[length_order]:
                if current_batch and (
                    current_loss_tokens >= self.target_loss_tokens
                    or len(current_batch) >= self.max_samples
                ):
                    rng.shuffle(current_batch)
                    batches.append(current_batch)
                    current_batch = []
                    current_loss_tokens = 0

                current_batch.append(int(idx))
                current_loss_tokens += int(self.loss_token_counts[idx])

        if current_batch:
            rng.shuffle(current_batch)
            batches.append(current_batch)

        rng.shuffle(batches)
        return batches

    def _get_batches(self):
        if self._cached_epoch != self.epoch:
            self._cached_batches = self._build_batches()
            self._cached_epoch = self.epoch
        return self._cached_batches

    def __len__(self):
        return len(self._get_batches())

    def __iter__(self):
        batches = self._get_batches()
        self.epoch += 1
        self._cached_epoch = None
        self._cached_batches = None
        yield from batches

class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, lengths, indices, loss_masks=None):
        self.X = X
        self.Y = Y
        self.lengths = lengths
        self.indices = indices
        self.loss_masks = loss_masks

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return (
            self.X[real_idx],
            self.Y[real_idx],
            self.lengths[real_idx],
            self.loss_masks[real_idx] if self.loss_masks is not None else None
        )
    
    @classmethod
    def create_dataloader(
        cls, X, Y, lengths, batch_size, max_seq_len, num_workers, shuffle,
        mode, loss_masks=None,
    ):
        if mode not in {"pretrain", "finetune"}:
            raise ValueError(
                f"mode phải là 'pretrain' hoặc 'finetune', nhận được: {mode!r}"
            )
        if mode == "finetune" and loss_masks is None:
            raise ValueError("Finetune dataloader bắt buộc phải có loss_masks")

        log_progress(f"Đang tạo dataset từ {len(X)} samples...")
        indices = np.arange(len(X))
        dataset = cls(X, Y, lengths, indices, loss_masks)
        bucket_size = 128
        PAD_ID = 0

        def collate_fn(batch):
            X_batch = [item[0] for item in batch]
            Y_batch = [item[1] for item in batch]
            lm_batch = [item[3] for item in batch]

            max_len = max(len(x) for x in X_batch)
            if max_len > (max_seq_len - (max_seq_len%bucket_size)):
                bucket = max_seq_len
            else:
                bucket = int(np.ceil(max_len / bucket_size) * bucket_size)

            bsz = len(X_batch)
            X_padded = torch.zeros((bsz, bucket), dtype=torch.long)
            Y_padded = torch.zeros((bsz, bucket), dtype=torch.long)
            has_padding = any(len(x) < bucket for x in X_batch)

            if loss_masks is not None:
                loss_mask_padded = torch.zeros((bsz, bucket), dtype=torch.float)

            for i, (x, y) in enumerate(zip(X_batch, Y_batch)):
                x_len = len(x)
                y_len = len(y)

                X_padded[i, :x_len] = torch.as_tensor(x, dtype=torch.long)
                Y_padded[i, :y_len] = torch.as_tensor(y, dtype=torch.long)

                if loss_masks is not None:
                    lm = lm_batch[i]
                    lm_len = len(lm)
                    loss_mask_padded[i, :lm_len] = torch.as_tensor(lm, dtype=torch.float)

            attention_mask = (X_padded != PAD_ID).float()

            if loss_masks is None:
                sample_weight = (Y_padded != PAD_ID).float()
            else:
                sample_weight = loss_mask_padded

            return X_padded, Y_padded, sample_weight, attention_mask, has_padding

        loader_kwargs = {
            "dataset": dataset,
            "collate_fn": collate_fn,
            "num_workers": num_workers,
            "pin_memory": torch.cuda.is_available(),
            "persistent_workers": True,
        }

        if shuffle and mode == "pretrain":
            batch_sampler = RandomizedSortishBatchSampler(
                lengths=lengths,
                batch_size=batch_size,
                pool_batches=50,
                drop_last=False,
            )
            dataloader = torch.utils.data.DataLoader(
                batch_sampler=batch_sampler,
                **loader_kwargs,
            )
            log_progress(
                f"Sortish sampler: pool={batch_sampler.pool_size} samples, "
                f"{len(batch_sampler)} batch/epoch"
            )
        elif shuffle and mode == "finetune":
            loss_token_counts = np.fromiter(
                (np.asarray(mask).sum() for mask in loss_masks),
                dtype=np.int64,
                count=len(loss_masks),
            )
            batch_sampler = TokenBalancedSortishBatchSampler(
                lengths=lengths,
                loss_token_counts=loss_token_counts,
                batch_size=batch_size,
                pool_batches=50,
                max_samples=batch_size * 4,
            )
            dataloader = torch.utils.data.DataLoader(
                batch_sampler=batch_sampler,
                **loader_kwargs,
            )
            log_progress(
                f"SFT token-balanced sortish: target={batch_sampler.target_loss_tokens:,} "
                f"loss-token/batch, pool={batch_sampler.pool_size}, "
                f"max_samples={batch_sampler.max_samples}, "
                f"{len(batch_sampler)} batch/epoch"
            )
        else:
            length_order = np.argsort(np.asarray(lengths), kind="stable")
            batch_sampler = [
                length_order[i:i + batch_size].tolist()
                for i in range(0, len(length_order), batch_size)
            ]
            dataloader = torch.utils.data.DataLoader(
                batch_sampler=batch_sampler,
                **loader_kwargs,
            )

        log_progress(f"Dataset mode={mode}, batch_size={batch_size}")
        return dataloader

def split_train_val_test(X, Y, loss_masks, lengths, train_ratio, val_ratio, seed=54):
    total_sample = len(X)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(total_sample)

    train_end = int(total_sample * train_ratio)
    val_end = int(total_sample * (train_ratio + val_ratio))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    X_train, Y_train, lengths_train = X[train_idx], Y[train_idx], lengths[train_idx]
    X_val, Y_val, lengths_val = X[val_idx], Y[val_idx], lengths[val_idx]
    X_test, Y_test, lengths_test = X[test_idx], Y[test_idx], lengths[test_idx]
    
    if loss_masks is not None:
        mask_train = loss_masks[train_idx]
        mask_val = loss_masks[val_idx]
        mask_test = loss_masks[test_idx]
        
        return (X_train, Y_train, mask_train, lengths_train, 
                X_val, Y_val, mask_val, lengths_val, 
                X_test, Y_test, mask_test, lengths_test)
    else:
        return (X_train, Y_train, None, lengths_train,
                X_val, Y_val, None, lengths_val,
                X_test, Y_test, None, lengths_test)

def _load_single_npz(path, load_mask=False):
    f = np.load(path)
    def reconstruct(name):
        flat, offsets = f[f"{name}_flat"], f[f"{name}_offsets"]
        return np.array([flat[offsets[i]:offsets[i+1]] for i in range(len(offsets)-1)], dtype=object)
    X = reconstruct("X")
    Y = reconstruct("Y")
    lengths = f["lengths"]
    if load_mask:
        M = reconstruct("M")
        f.close()
        return X, Y, M, lengths
    f.close()
    return X, Y, lengths

def _load_manifest_shards(manifest_path, load_mask=False):
    """Đọc manifest.json rồi ghép nhiều shard .npz lại thành 1 dataset logic."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    manifest_dir = Path(manifest_path).parent
    shards = manifest["shards"]

    X_list, Y_list, M_list, L_list = [], [], [], []
    for shard in shards:
        shard_path = manifest_dir / shard["file"]
        if load_mask:
            X_s, Y_s, M_s, L_s = _load_single_npz(shard_path, load_mask=True)
            M_list.append(M_s)
        else:
            X_s, Y_s, L_s = _load_single_npz(shard_path, load_mask=False)
        X_list.append(X_s)
        Y_list.append(Y_s)
        L_list.append(L_s)

    X = np.concatenate(X_list)
    Y = np.concatenate(Y_list)
    lengths = np.concatenate(L_list)

    if load_mask:
        M = np.concatenate(M_list)
        return X, Y, M, lengths
    return X, Y, lengths

def load_npz(path, load_mask=False):
    path = Path(path)
    if path.suffix == ".json":
        return _load_manifest_shards(path, load_mask=load_mask)
    return _load_single_npz(path, load_mask=load_mask)

def load_data(data_type, main_data, sub_data=None, seed=54):
    if data_type == "pretrain":
        X, Y, lengths = load_npz(main_data)
        return X, Y, lengths

    elif data_type == "continued_pretrain":
        X_main, Y_main, L_main = load_npz(main_data)

        if sub_data is not None:
            X_sub, Y_sub, L_sub = load_npz(sub_data)

            n_continued = len(X_main)
            n_pretrain_needed = 3 * n_continued
            total_pretrain = len(X_sub)
            
            rng = np.random.default_rng(seed)
            shuffled_indices = rng.permutation(total_pretrain)
            
            n_samples = min(n_pretrain_needed, total_pretrain)
            selected_indices = shuffled_indices[:n_samples]
            
            X_sub_sampled = X_sub[selected_indices]
            Y_sub_sampled = Y_sub[selected_indices]
            L_sub_sampled = L_sub[selected_indices]
            
            del X_sub, Y_sub, L_sub, shuffled_indices, selected_indices
            gc.collect()
            
            X_combined = np.concatenate([X_main, X_sub_sampled])
            Y_combined = np.concatenate([Y_main, Y_sub_sampled])
            L_combined = np.concatenate([L_main, L_sub_sampled])
            
            combined_indices = rng.permutation(len(X_combined))
            X = X_combined[combined_indices]
            Y = Y_combined[combined_indices]
            lengths = L_combined[combined_indices]
            
        else:
            X, Y, lengths = X_main, Y_main, L_main

        return X, Y, lengths

    elif data_type == "sft1":
        X, Y, loss_mask, lengths = load_npz(main_data, load_mask=True)
        return X, Y, loss_mask, lengths
    
    elif data_type == "sft2":
        X_main, Y_main, M_main, L_main = load_npz(main_data, load_mask=True)

        if sub_data is not None:
            X_sub, Y_sub, M_sub, L_sub = load_npz(sub_data, load_mask=True)

            n_sub = len(X_main)
            total_sub = len(X_sub)

            rng = np.random.default_rng(seed)
            shuffled_indices = rng.permutation(total_sub)

            n_samples = min(n_sub, total_sub)
            selected_indices = shuffled_indices[:n_samples]

            X_sub_sampled = X_sub[selected_indices]
            Y_sub_sampled = Y_sub[selected_indices]
            M_sub_sampled = M_sub[selected_indices]
            L_sub_sampled = L_sub[selected_indices]

            del X_sub, Y_sub, M_sub, L_sub, shuffled_indices, selected_indices
            gc.collect()

            X_combined = np.concatenate([X_main, X_sub_sampled])
            Y_combined = np.concatenate([Y_main, Y_sub_sampled])
            M_combined = np.concatenate([M_main, M_sub_sampled])
            L_combined = np.concatenate([L_main, L_sub_sampled])

            combined_indices = rng.permutation(len(X_combined))
            X = X_combined[combined_indices]
            Y = Y_combined[combined_indices]
            loss_mask = M_combined[combined_indices]
            lengths = L_combined[combined_indices]

        else:
            X, Y, loss_mask, lengths = X_main, Y_main, M_main, L_main

        return X, Y, loss_mask, lengths

    else:
        raise ValueError(f"Unknown data_type: {data_type}")
