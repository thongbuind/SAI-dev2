from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import torch

TokenConstraint = Callable[[list[int]], Sequence[int] | None]

@dataclass(slots=True)
class GenerationConfig:
    strategy: str = "auto"
    max_new_tokens: int = 500
    beam_size: int = 5
    stable_steps: int | None = 8
    temperature: float = 0.8
    top_k: int = 0
    top_p: float = 0.95
    repetition_penalty: float = 1.2
    no_repeat_ngram_size: int = 3
    length_penalty: float = 1.0
    lowercase: bool = True
    seed: int | None = None
    stop_sequences: list[list[int]] = field(default_factory=list)

@dataclass(slots=True)
class GenerationResult:
    text: str
    token_ids: list[int]
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    decode_steps: int
    completed_beams: int

def _special_ids(tokenizer):
    ids = {
        "bos": tokenizer.piece_to_id("[BOS]"),
        "eos": tokenizer.piece_to_id("[EOS]"),
        "im_start": tokenizer.piece_to_id("<|im_start|>"),
        "im_end": tokenizer.piece_to_id("<|im_end|>"),
    }
    if any(token_id < 0 for token_id in ids.values()):
        raise ValueError(f"Tokenizer is missing required special tokens: {ids}")
    return ids

def _normalise_messages(user_input, system=None, instruction=None, history=None, messages=None):
    if system is not None and instruction is not None:
        raise ValueError("Use either system or instruction, not both")
    system = system if system is not None else instruction
    if messages is not None:
        if user_input not in (None, "") or system is not None or history:
            raise ValueError("Use either messages or user_input/instruction/history, not both")
        source = list(messages)
    else:
        source = []
        if system:
            source.append({"role": "system", "content": system})
        for turn in history or []:
            if "role" in turn:
                source.append(turn)
                continue
            if turn.get("user"):
                source.append({"role": "user", "content": turn["user"]})
            if turn.get("model"):
                source.append({"role": "model", "content": turn["model"]})
        if user_input is not None:
            source.append({"role": "user", "content": user_input})

    result = []
    role_map = {"assistant": "model", "model": "model", "user": "user", "system": "system"}
    for message in source:
        role = role_map.get(str(message.get("role", "")).lower())
        content = str(message.get("content", "")).strip()
        if role is None or not content:
            raise ValueError(f"Invalid chat message: {message!r}")
        result.append({"role": role, "content": content})
    if not result or result[-1]["role"] != "user":
        raise ValueError("The final input message must have role='user'")
    return result

def _build_input(user_input, tokenizer, *, max_seq_len, config, system=None, instruction=None, history=None, messages=None):
    special = _special_ids(tokenizer)
    normalised = _normalise_messages(user_input, system, instruction, history, messages)

    def encode_message(message):
        text = f"{message['role']}\n{message['content']}"
        if config.lowercase:
            text = text.lower()
        return [special["im_start"]] + tokenizer.encode(text, out_type=int) + [special["im_end"]]

    segments = [(message["role"], encode_message(message)) for message in normalised]
    assistant_prefix = [special["im_start"]] + tokenizer.encode("model\n", out_type=int)
    prompt_budget = max_seq_len - config.max_new_tokens
    required = 1 + len(assistant_prefix)
    if prompt_budget < required:
        raise ValueError("max_new_tokens leaves no room for a prompt")

    system_segments = [segment for role, segment in segments if role == "system"]
    latest = segments[-1][1]
    fixed_len = required + len(latest) + sum(map(len, system_segments))
    if fixed_len > prompt_budget:
        raise ValueError(
            f"System + latest user message need {fixed_len} tokens, "
            f"but prompt budget is {prompt_budget}"
        )

    middle = [(role, segment) for role, segment in segments[:-1] if role != "system"]
    turn_groups = []
    index = 0
    while index < len(middle):
        role, segment = middle[index]
        group = [segment]
        if role == "user" and index + 1 < len(middle) and middle[index + 1][0] == "model":
            group.append(middle[index + 1][1])
            index += 1
        turn_groups.append(group)
        index += 1

    kept_groups = []
    used = fixed_len
    for group in reversed(turn_groups):
        group_len = sum(map(len, group))
        if used + group_len <= prompt_budget:
            kept_groups.append(group)
            used += group_len
    kept_groups.reverse()

    input_ids = [special["bos"]]
    for segment in system_segments:
        input_ids.extend(segment)
    for group in kept_groups:
        for segment in group:
            input_ids.extend(segment)
    input_ids.extend(latest)
    input_ids.extend(assistant_prefix)
    return input_ids, special

def _validate_config(config):
    if config.strategy not in {"auto", "beam", "greedy", "sample"}:
        raise ValueError("strategy must be auto, beam, greedy, or sample")
    if config.max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")
    if config.beam_size < 1:
        raise ValueError("beam_size must be >= 1")
    if config.stable_steps is not None and config.stable_steps < 1:
        raise ValueError("stable_steps must be >= 1 or None")
    if config.temperature <= 0:
        raise ValueError("temperature must be > 0")
    if config.top_k < 0:
        raise ValueError("top_k must be >= 0")
    if not 0 < config.top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if config.repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be > 0")
    if config.no_repeat_ngram_size < 0:
        raise ValueError("no_repeat_ngram_size must be >= 0")
    if config.length_penalty < 0:
        raise ValueError("length_penalty must be >= 0")

def _apply_repetition_penalty(logits, generated, penalty):
    if penalty == 1.0 or not generated:
        return logits
    logits = logits.clone()
    token_ids = torch.tensor(list(set(generated)), dtype=torch.long, device=logits.device)
    selected = logits.index_select(-1, token_ids)
    selected = torch.where(selected < 0, selected * penalty, selected / penalty)
    logits.scatter_(-1, token_ids, selected)
    return logits

def _banned_ngram_tokens(generated, n):
    if n <= 0 or len(generated) < n - 1:
        return set()
    if n == 1:
        return set(generated)
    prefix = tuple(generated[-(n - 1):])
    return {
        generated[index + n - 1]
        for index in range(len(generated) - n + 1)
        if tuple(generated[index:index + n - 1]) == prefix
    }

def _is_stopped(generated, stop_ids, stop_sequences):
    if generated and generated[-1] in stop_ids:
        return True
    return any(
        sequence and len(generated) >= len(sequence) and generated[-len(sequence):] == sequence
        for sequence in stop_sequences
    )

def _decode(generated, tokenizer, stop_ids, stop_sequences):
    output = list(generated)
    for sequence in stop_sequences:
        if sequence and len(output) >= len(sequence) and output[-len(sequence):] == sequence:
            del output[-len(sequence):]
            break
    while output and output[-1] in stop_ids:
        output.pop()
    return tokenizer.decode(output)

def _beam_score(candidate, length_penalty):
    length = max(len(candidate["generated"]), 1)
    return candidate["log_prob"] / (length ** length_penalty)

def _process_logits(logits, generated, config, constraint=None):
    logits = _apply_repetition_penalty(logits, generated, config.repetition_penalty)
    banned = _banned_ngram_tokens(generated, config.no_repeat_ngram_size)
    if banned:
        logits = logits.clone()
        logits[list(banned)] = -torch.inf
    if constraint is not None:
        allowed = constraint(generated)
        if allowed is not None:
            allowed = list(allowed)
            if not allowed:
                raise ValueError("Token constraint returned an empty allowed set")
            constrained = torch.full_like(logits, -torch.inf)
            constrained[allowed] = logits[allowed]
            logits = constrained
    return logits

def _filter_sampling_logits(logits, config):
    logits = logits / config.temperature
    if config.top_k > 0:
        threshold = torch.topk(logits, min(config.top_k, logits.numel())).values[-1]
        logits = logits.masked_fill(logits < threshold, -torch.inf)
    if config.top_p < 1:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probabilities = torch.softmax(sorted_logits, dim=-1)
        remove = probabilities.cumsum(dim=-1) > config.top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        logits = logits.clone()
        logits[sorted_indices[remove]] = -torch.inf
    return logits

def _single_path_generate(model, input_ids, tokenizer, special, config, strategy, constraint):
    device = model.lm_head.weight.device
    stop_ids = {special["eos"], special["im_end"]}
    stop_sequences = [list(sequence) for sequence in config.stop_sequences]
    prompt_len = len(input_ids)
    cache = model.init_cache(1, min(model.max_seq_len, prompt_len + config.max_new_tokens), device)
    prompt = torch.tensor([input_ids], dtype=torch.long, device=device)
    generator = None
    if config.seed is not None:
        generator = torch.Generator(device=device).manual_seed(config.seed)

    with torch.inference_mode():
        logits, _ = model.prefill(prompt, kv_cache=cache)
        generated = []
        finish_reason = "length"
        decode_steps = 0
        for step in range(config.max_new_tokens):
            row = _process_logits(logits[0], generated, config, constraint)
            if strategy == "greedy":
                token = int(torch.argmax(row).item())
            else:
                row = _filter_sampling_logits(row, config)
                token = int(torch.multinomial(torch.softmax(row, dim=-1), 1, generator=generator).item())
            generated.append(token)
            if _is_stopped(generated, stop_ids, stop_sequences):
                finish_reason = "stop"
                break
            if step + 1 < config.max_new_tokens:
                last = torch.tensor([[token]], dtype=torch.long, device=device)
                logits = model.decode_step(last, cache, prompt_len + step)
                decode_steps += 1

    return GenerationResult(
        text=_decode(generated, tokenizer, stop_ids, stop_sequences),
        token_ids=generated,
        finish_reason=finish_reason,
        prompt_tokens=prompt_len,
        completion_tokens=len(generated),
        decode_steps=decode_steps,
        completed_beams=0,
    )

def _beam_generate(model, input_ids, tokenizer, special, config):
    device = model.lm_head.weight.device
    stop_ids = {special["eos"], special["im_end"]}
    stop_sequences = [list(sequence) for sequence in config.stop_sequences]
    prompt_len = len(input_ids)
    cache = model.init_cache(
        config.beam_size,
        min(model.max_seq_len, prompt_len + config.max_new_tokens),
        device,
    )
    prompt = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.inference_mode():
        first_logits, present = model.prefill(prompt, kv_cache=None)
        for layer, (key, value) in enumerate(present):
            cache[layer][0][:, :, :prompt_len].copy_(key.expand(config.beam_size, -1, -1, -1))
            cache[layer][1][:, :, :prompt_len].copy_(value.expand(config.beam_size, -1, -1, -1))

        first_log_probs = torch.log_softmax(first_logits[0], dim=-1)
        top_log_probs, top_tokens = torch.topk(
            first_log_probs,
            min(config.beam_size, first_log_probs.numel()),
        )
        alive, completed = [], []
        for log_prob, token in zip(top_log_probs.tolist(), top_tokens.tolist()):
            candidate = {"generated": [int(token)], "log_prob": float(log_prob), "source": 0}
            target = completed if _is_stopped(candidate["generated"], stop_ids, stop_sequences) else alive
            target.append(candidate)

        cache_len = prompt_len
        decode_steps = 0
        stable_count = 0
        last_best_completed = None
        finish_reason = "length"

        while alive and max(len(item["generated"]) for item in alive) < config.max_new_tokens:
            last_tokens = torch.tensor(
                [[candidate["generated"][-1]] for candidate in alive],
                dtype=torch.long,
                device=device,
            )
            logits = model.decode_step(last_tokens, cache, cache_len)
            cache_len += 1
            decode_steps += 1

            candidates = []
            for beam_index, candidate in enumerate(alive):
                row = _process_logits(logits[beam_index], candidate["generated"], config)
                log_probs = torch.log_softmax(row, dim=-1)
                values, tokens = torch.topk(log_probs, min(config.beam_size, log_probs.numel()))
                for value, token in zip(values.tolist(), tokens.tolist()):
                    candidates.append({
                        "generated": candidate["generated"] + [int(token)],
                        "log_prob": candidate["log_prob"] + float(value),
                        "source": beam_index,
                    })

            candidates.sort(
                key=lambda item: _beam_score(item, config.length_penalty),
                reverse=True,
            )
            next_alive = []
            for candidate in candidates:
                if _is_stopped(candidate["generated"], stop_ids, stop_sequences):
                    completed.append(candidate)
                elif len(next_alive) < config.beam_size:
                    next_alive.append(candidate)
                if len(next_alive) >= config.beam_size and len(completed) >= config.beam_size:
                    break

            if completed:
                best_completed_score = max(
                    _beam_score(item, config.length_penalty) for item in completed
                )
                if last_best_completed is None or best_completed_score > last_best_completed + 1e-12:
                    last_best_completed = best_completed_score
                    stable_count = 0
                else:
                    stable_count += 1

                if (
                    config.stable_steps is not None
                    and len(completed) >= config.beam_size
                    and stable_count >= config.stable_steps
                ):
                    best_alive_score = max(
                        (_beam_score(item, config.length_penalty) for item in next_alive),
                        default=float("-inf"),
                    )
                    if best_completed_score >= best_alive_score:
                        finish_reason = "stable_beam"
                        alive = next_alive
                        break

            if not next_alive:
                finish_reason = "all_beams_finished"
                break

            sources_list = [candidate["source"] for candidate in next_alive]
            if sources_list != list(range(len(sources_list))):
                sources = torch.tensor(sources_list, dtype=torch.long, device=device)
                for key, value in cache:
                    selected_key = key.index_select(0, sources)[:, :, :cache_len].clone()
                    selected_value = value.index_select(0, sources)[:, :, :cache_len].clone()
                    key[:len(next_alive), :, :cache_len].copy_(selected_key)
                    value[:len(next_alive), :, :cache_len].copy_(selected_value)
            alive = next_alive

    pool = completed + alive
    if not pool:
        return GenerationResult("", [], "empty", prompt_len, 0, decode_steps, len(completed))
    best = max(pool, key=lambda item: _beam_score(item, config.length_penalty))
    return GenerationResult(
        text=_decode(best["generated"], tokenizer, stop_ids, stop_sequences),
        token_ids=list(best["generated"]),
        finish_reason=finish_reason,
        prompt_tokens=prompt_len,
        completion_tokens=len(best["generated"]),
        decode_steps=decode_steps,
        completed_beams=len(completed),
    )

def generate(model, user_input=None, tokenizer=None, *, messages=None, system=None, instruction=None, history=None, config=None, constraint: TokenConstraint | None = None, return_result=False):
    """Generate a response with beam search, greedy decoding, or sampling."""
    if tokenizer is None:
        raise ValueError("tokenizer is required")
    config = config or GenerationConfig()
    _validate_config(config)
    strategy = config.strategy
    if strategy == "auto":
        strategy = "greedy" if constraint is not None else "beam"
    if strategy == "beam" and constraint is not None:
        raise ValueError("Beam strategy does not support token constraints")

    input_ids, special = _build_input(
        user_input,
        tokenizer,
        max_seq_len=model.max_seq_len,
        config=config,
        system=system,
        instruction=instruction,
        history=history,
        messages=messages,
    )
    if strategy == "beam":
        result = _beam_generate(model, input_ids, tokenizer, special, config)
    else:
        result = _single_path_generate(
            model,
            input_ids,
            tokenizer,
            special,
            config,
            strategy,
            constraint,
        )
    return result if return_result else result.text
