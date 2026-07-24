from __future__ import annotations

import gc
import math
import os
import statistics
import time
import traceback
from threading import Thread
from pathlib import Path
from typing import Any, Callable

from ..schema import CompressionRequest
from ..utils import utc_now


EventCallback = Callable[[dict[str, Any]], None]


def _event(callback: EventCallback | None, value: dict[str, Any]) -> None:
    if callback:
        callback(value)


def _stream_transformers_generation(
    model: Any,
    tokenizer: Any,
    *,
    prompt: str,
    max_new_tokens: int,
    device: str,
    event_callback: EventCallback | None,
) -> tuple[str, dict[str, Any]]:
    """Generate a sample while forwarding decoded chunks to the Web workbench."""
    import torch
    from transformers import TextIteratorStreamer

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=120,
    )
    holder: dict[str, Any] = {}

    def generate() -> None:
        try:
            with torch.inference_mode():
                holder["output"] = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    streamer=streamer,
                )
        except BaseException as exc:  # Forward worker-thread failures.
            holder["error"] = exc

    _event(
        event_callback,
        {
            "type": "generation_start",
            "backend": "transformers",
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
        },
    )
    started = time.perf_counter()
    thread = Thread(target=generate, daemon=True)
    thread.start()
    generated_text = ""
    generated_tokens = 0
    for chunk in streamer:
        generated_text += chunk
        generated_tokens = len(
            tokenizer.encode(generated_text, add_special_tokens=False)
        )
        elapsed = time.perf_counter() - started
        _event(
            event_callback,
            {
                "type": "generation_token",
                "backend": "transformers",
                "text": generated_text,
                "tokens": generated_tokens,
                "elapsed_seconds": elapsed,
                "tokens_per_second": generated_tokens / max(elapsed, 1e-12),
            },
        )
    thread.join()
    if holder.get("error"):
        raise holder["error"]
    elapsed = time.perf_counter() - started
    output = holder["output"]
    full_text = tokenizer.decode(output[0], skip_special_tokens=True)
    metrics = {
        "tokens": generated_tokens,
        "elapsed_seconds": elapsed,
        "tokens_per_second": generated_tokens / max(elapsed, 1e-12),
    }
    _event(
        event_callback,
        {
            "type": "generation_complete",
            "backend": "transformers",
            "prompt": prompt,
            "text": generated_text,
            **metrics,
        },
    )
    return full_text, metrics


def _sync(torch_module, device: str) -> None:
    if device.startswith("cuda") and torch_module.cuda.is_available():
        torch_module.cuda.synchronize(device)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = round((len(ordered) - 1) * fraction)
    return ordered[max(0, min(index, len(ordered) - 1))]


def _dataset_texts(path: Path, limit: int = 10_000) -> list[str]:
    from datasets import Dataset

    arrows = [path] if path.is_file() else sorted(path.rglob("*test*.arrow"))
    if not arrows and path.is_dir():
        arrows = sorted(path.rglob("*.arrow"))
    if not arrows:
        raise FileNotFoundError(f"no Arrow files below {path}")
    dataset = Dataset.from_file(str(arrows[0]))
    column = next(
        (item for item in ("text", "content", "sentence") if item in dataset.column_names),
        None,
    )
    if not column:
        raise ValueError(f"dataset has no text column: {dataset.column_names}")
    texts = []
    for value in dataset[column]:
        text = str(value).strip()
        if text:
            texts.append(text)
        if len(texts) >= limit:
            break
    return texts


def evaluate_perplexity(
    model: Any,
    tokenizer: Any,
    dataset_path: Path,
    *,
    max_tokens: int,
    device: str,
) -> dict[str, Any]:
    import torch

    texts = _dataset_texts(dataset_path)
    encoded = tokenizer(
        "\n\n".join(texts),
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids[:, :max_tokens]
    if encoded.numel() < 2:
        raise RuntimeError("evaluation dataset produced fewer than two tokens")
    model_max = int(
        min(
            getattr(model.config, "max_position_embeddings", 2048) or 2048,
            2048,
        )
    )
    stride = model_max
    losses: list[float] = []
    scored_tokens = 0
    for start in range(0, encoded.shape[1] - 1, stride):
        end = min(start + model_max, encoded.shape[1])
        batch = encoded[:, start:end].to(device)
        if batch.shape[1] < 2:
            break
        labels = batch.clone()
        with torch.inference_mode():
            output = model(input_ids=batch, labels=labels, use_cache=False)
        count = batch.shape[1] - 1
        losses.append(float(output.loss.detach().float().item()) * count)
        scored_tokens += count
    mean_nll = sum(losses) / max(scored_tokens, 1)
    return {
        "metric": "wikitext2_perplexity",
        "perplexity": float(math.exp(min(mean_nll, 80))),
        "mean_nll": mean_nll,
        "scored_tokens": scored_tokens,
    }


def _exact_length_ids(tokenizer: Any, prompt: str, length: int, device: str, batch: int):
    import torch

    base = tokenizer(prompt, add_special_tokens=True).input_ids
    if not base:
        base = [tokenizer.bos_token_id or tokenizer.eos_token_id or 0]
    repeated = (base * ((length // len(base)) + 1))[:length]
    return torch.tensor([repeated] * batch, dtype=torch.long, device=device)


def _transformers_shape_benchmark(
    model: Any,
    tokenizer: Any,
    *,
    prompt: str,
    batch: int,
    input_length: int,
    output_length: int,
    warmup: int,
    iterations: int,
    device: str,
) -> dict[str, Any]:
    import torch

    input_ids = _exact_length_ids(tokenizer, prompt, input_length, device, batch)
    warmup_count = max(0, warmup)
    iteration_count = max(1, iterations)

    def run_once() -> tuple[float, list[float], float]:
        _sync(torch, device)
        start = time.perf_counter()
        with torch.inference_mode():
            output = model(input_ids=input_ids, use_cache=True)
        _sync(torch, device)
        ttft = (time.perf_counter() - start) * 1000
        past = output.past_key_values
        next_token = output.logits[:, -1:, :].argmax(dim=-1)
        decode_samples: list[float] = []
        for _ in range(output_length):
            _sync(torch, device)
            step_start = time.perf_counter()
            with torch.inference_mode():
                output = model(
                    input_ids=next_token,
                    past_key_values=past,
                    use_cache=True,
                )
            _sync(torch, device)
            decode_samples.append((time.perf_counter() - step_start) * 1000)
            past = output.past_key_values
            next_token = output.logits[:, -1:, :].argmax(dim=-1)
        return ttft, decode_samples, sum(decode_samples)

    for _ in range(warmup_count):
        run_once()
    ttfts: list[float] = []
    decode_steps: list[float] = []
    totals: list[float] = []
    for _ in range(iteration_count):
        ttft, steps, total = run_once()
        ttfts.append(ttft)
        decode_steps.extend(steps)
        totals.append(total)
    ttft_p50 = statistics.median(ttfts)
    tpot_p50 = statistics.median(decode_steps)
    decode_total = statistics.median(totals)
    return {
        "batch_size": batch,
        "input_length": input_length,
        "output_length": output_length,
        "warmup": warmup_count,
        "iterations": iteration_count,
        "ttft_p50_ms": ttft_p50,
        "ttft_p95_ms": _percentile(ttfts, 0.95),
        "prefill_tokens_per_second": batch * input_length / max(ttft_p50 / 1000, 1e-12),
        "tpot_p50_ms": tpot_p50,
        "tpot_p95_ms": _percentile(decode_steps, 0.95),
        "decode_tokens_per_second": batch * output_length / max(decode_total / 1000, 1e-12),
        "request_throughput_per_second": batch / max((ttft_p50 + decode_total) / 1000, 1e-12),
        "end_to_end_p50_ms": ttft_p50 + decode_total,
    }


def benchmark_transformers(
    model_path: Path,
    request: CompressionRequest,
    *,
    mode: str,
    include_quality: bool = True,
    event_callback: EventCallback | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "generated_at": utc_now(),
        "status": "running",
        "backend": "transformers",
        "model_path": str(model_path),
        "scope": "end_to_end",
    }
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = request.execution.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=request.execution.offline,
            trust_remote_code=request.model.trust_remote_code,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            local_files_only=request.execution.offline,
            trust_remote_code=request.model.trust_remote_code,
            low_cpu_mem_usage=True,
        ).to(device)
        model.eval()
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)

        batches = request.workload.batch_sizes
        inputs = request.workload.input_lengths
        outputs = request.workload.output_lengths
        if mode == "smoke":
            batches = batches[:1]
            inputs = inputs[:1]
            outputs = outputs[:1]
        records = []
        warmup = (
            min(request.workload.warmup, 2)
            if mode == "smoke"
            else request.workload.warmup
        )
        iterations = (
            min(request.workload.iterations, 5)
            if mode == "smoke"
            else request.workload.iterations
        )
        for batch in batches:
            for input_length in inputs:
                for output_length in outputs:
                    records.append(
                        _transformers_shape_benchmark(
                            model,
                            tokenizer,
                            prompt=request.evaluation.generation_prompts[0],
                            batch=batch,
                            input_length=input_length,
                            output_length=output_length,
                            warmup=warmup,
                            iterations=iterations,
                            device=device,
                        )
                    )

        prompt = request.evaluation.generation_prompts[0]
        sample, generation_metrics = _stream_transformers_generation(
            model,
            tokenizer,
            prompt=prompt,
            max_new_tokens=min(outputs[0], 32),
            device=device,
            event_callback=event_callback,
        )
        quality = None
        if include_quality:
            quality = evaluate_perplexity(
                model,
                tokenizer,
                Path(request.evaluation.dataset),
                max_tokens=(
                    request.evaluation.max_tokens_smoke
                    if mode == "smoke"
                    else request.evaluation.max_tokens_full
                ),
                device=device,
            )
        peak = (
            int(torch.cuda.max_memory_allocated(device))
            if device.startswith("cuda")
            else None
        )
        result.update(
            {
                "status": "completed",
                "records": records,
                "quality": quality,
                "peak_vram_bytes": peak,
                "generation_sample": sample,
                "generation_metrics": generation_metrics,
                "timing_note": "CUDA synchronized; prefill and token-by-token decode measured separately.",
            }
        )
        del model
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    except Exception as exc:
        result.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    return result


def evaluate_transformers_artifact(
    model_path: Path,
    request: CompressionRequest,
    *,
    mode: str,
    event_callback: EventCallback | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "generated_at": utc_now(),
        "status": "running",
        "backend": "transformers",
        "scope": "quality_and_generation",
        "model_path": str(model_path),
    }
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = request.execution.device
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=request.execution.offline,
            trust_remote_code=request.model.trust_remote_code,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
            local_files_only=request.execution.offline,
            trust_remote_code=request.model.trust_remote_code,
            low_cpu_mem_usage=True,
        ).to(device)
        model.eval()
        quality = evaluate_perplexity(
            model,
            tokenizer,
            Path(request.evaluation.dataset),
            max_tokens=(
                request.evaluation.max_tokens_smoke
                if mode == "smoke"
                else request.evaluation.max_tokens_full
            ),
            device=device,
        )
        prompt = request.evaluation.generation_prompts[0]
        sample, generation_metrics = _stream_transformers_generation(
            model,
            tokenizer,
            prompt=prompt,
            max_new_tokens=16,
            device=device,
            event_callback=event_callback,
        )
        result.update(
            {
                "status": "completed",
                "quality": quality,
                "generation_sample": sample,
                "generation_metrics": generation_metrics,
            }
        )
        del model
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    except Exception as exc:
        result.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    return result


def benchmark_vllm(
    model_path: Path,
    request: CompressionRequest,
    *,
    mode: str,
    quantization: str | None = None,
    event_callback: EventCallback | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "generated_at": utc_now(),
        "status": "running",
        "backend": "vllm",
        "model_path": str(model_path),
        "scope": "end_to_end",
        "quantization": quantization,
    }
    try:
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        from transformers import AutoConfig
        from vllm import LLM, SamplingParams

        batches = request.workload.batch_sizes[:1] if mode == "smoke" else request.workload.batch_sizes
        inputs = request.workload.input_lengths[:1] if mode == "smoke" else request.workload.input_lengths
        outputs = request.workload.output_lengths[:1] if mode == "smoke" else request.workload.output_lengths
        required_model_len = max(inputs) + max(outputs)
        config = AutoConfig.from_pretrained(
            model_path,
            local_files_only=request.execution.offline,
            trust_remote_code=request.model.trust_remote_code,
        )
        configured_limits = [
            value
            for value in (
                getattr(config, "max_position_embeddings", None),
                getattr(config, "model_max_length", None),
            )
            if isinstance(value, int) and 0 < value < 1_000_000_000
        ]
        configured_model_len = min(configured_limits) if configured_limits else None
        max_model_len = (
            min(required_model_len, configured_model_len)
            if configured_model_len
            else required_model_len
        )

        kwargs: dict[str, Any] = {
            "model": str(model_path),
            "trust_remote_code": request.model.trust_remote_code,
            "tensor_parallel_size": request.execution.tensor_parallel_size,
            "gpu_memory_utilization": 0.80,
            "max_model_len": max_model_len,
        }
        if quantization:
            kwargs["quantization"] = quantization
        llm = LLM(**kwargs)
        tokenizer = llm.get_tokenizer()
        records = []
        for batch in batches:
            for input_length in inputs:
                for output_length in outputs:
                    effective_input_length = min(
                        input_length, max_model_len - output_length
                    )
                    if effective_input_length <= 0:
                        continue
                    base = tokenizer.encode(
                        request.evaluation.generation_prompts[0],
                        add_special_tokens=True,
                    )
                    ids = (
                        base
                        * ((effective_input_length // max(len(base), 1)) + 1)
                    )[:effective_input_length]
                    prompts = [{"prompt_token_ids": ids} for _ in range(batch)]
                    params = SamplingParams(
                        temperature=0,
                        max_tokens=output_length,
                    )
                    warmup = (
                        min(request.workload.warmup, 2)
                        if mode == "smoke"
                        else request.workload.warmup
                    )
                    iterations = (
                        min(request.workload.iterations, 5)
                        if mode == "smoke"
                        else request.workload.iterations
                    )
                    for _ in range(warmup):
                        llm.generate(prompts, params, use_tqdm=False)
                    samples = []
                    generated_tokens = 0
                    latest = None
                    for _ in range(max(1, iterations)):
                        start = time.perf_counter()
                        latest = llm.generate(prompts, params, use_tqdm=False)
                        elapsed = time.perf_counter() - start
                        samples.append(elapsed)
                        generated_tokens = sum(
                            len(item.outputs[0].token_ids) for item in latest
                        )
                    elapsed_p50 = statistics.median(samples)
                    records.append(
                        {
                            "batch_size": batch,
                            "input_length": effective_input_length,
                            "requested_input_length": input_length,
                            "output_length": output_length,
                            "warmup": warmup,
                            "iterations": max(1, iterations),
                            "end_to_end_p50_ms": elapsed_p50 * 1000,
                            "end_to_end_p95_ms": _percentile(samples, 0.95) * 1000,
                            "decode_tokens_per_second": generated_tokens
                            / max(elapsed_p50, 1e-12),
                            "request_throughput_per_second": batch
                            / max(elapsed_p50, 1e-12),
                            "ttft_p50_ms": None,
                            "tpot_p50_ms": (elapsed_p50 * 1000)
                            / max(output_length, 1),
                            "timing_note": "vLLM batch wall time; TTFT unavailable in this compatibility adapter.",
                        }
                    )
        sample_params = SamplingParams(temperature=0, max_tokens=16)
        prompt = request.evaluation.generation_prompts[0]
        _event(
            event_callback,
            {
                "type": "generation_start",
                "backend": "vllm",
                "prompt": prompt,
                "max_new_tokens": 16,
            },
        )
        sample_started = time.perf_counter()
        sample_output = llm.generate(
            [prompt],
            sample_params,
            use_tqdm=False,
        )[0].outputs[0].text
        sample_elapsed = time.perf_counter() - sample_started
        sample_tokens = len(tokenizer.encode(sample_output, add_special_tokens=False))
        generation_metrics = {
            "tokens": sample_tokens,
            "elapsed_seconds": sample_elapsed,
            "tokens_per_second": sample_tokens / max(sample_elapsed, 1e-12),
        }
        _event(
            event_callback,
            {
                "type": "generation_complete",
                "backend": "vllm",
                "prompt": prompt,
                "text": sample_output,
                **generation_metrics,
            },
        )
        result.update(
            {
                "status": "completed",
                "records": records,
                "quality": None,
                "generation_sample": sample_output,
                "generation_metrics": generation_metrics,
            }
        )
        del llm
        gc.collect()
    except Exception as exc:
        result.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    return result


def benchmark_cublas_weight(
    weight: Any,
    *,
    token_counts: list[int],
    warmup: int,
    iterations: int,
    device: str,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    target = torch.device(device)
    matrix = weight.detach().to(target, dtype=torch.float16).contiguous()
    records = []
    for tokens in token_counts:
        inputs = torch.randn(tokens, matrix.shape[1], device=target, dtype=torch.float16)
        for _ in range(warmup):
            F.linear(inputs, matrix)
        torch.cuda.synchronize(target)
        samples = []
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            F.linear(inputs, matrix)
            end.record()
            torch.cuda.synchronize(target)
            samples.append(float(start.elapsed_time(end)))
        records.append(
            {
                "tokens": tokens,
                "m": tokens,
                "k": int(matrix.shape[1]),
                "n": int(matrix.shape[0]),
                "cublas_p50_ms": statistics.median(samples),
                "cublas_p95_ms": _percentile(samples, 0.95),
            }
        )
    return {
        "generated_at": utc_now(),
        "status": "completed",
        "backend": "cublas",
        "weight_shape": list(matrix.shape),
        "records": records,
    }
