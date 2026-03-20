from __future__ import annotations
import asyncio
import gc
import logging
import math
import os
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import anthropic
import httpx
import openai
import torch
import transformers
from anthropic import BadRequestError
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError
from vllm import LLM, SamplingParams


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def retry_generate(fn, max_retries=5, backoff_base=0.5, backoff_factor=2.0):
    """Calls fn() up to max_retries times with exponential backoff."""
    last_exc = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            time.sleep(backoff_base * (backoff_factor ** attempt))
    raise last_exc


def _to_blocks(x):
    """Normalize various input types to a list of Anthropic content blocks."""
    if x is None:
        return []
    if isinstance(x, str):
        s = x.strip()
        return [{"type": "text", "text": s}] if s else []
    if isinstance(x, dict):
        if "type" in x:
            return [x] if not (x["type"] == "text" and "text" not in x) else []
        s = str(x).strip()
        return [{"type": "text", "text": s}] if s else []
    if isinstance(x, (list, tuple)):
        if not x:
            return []
        if isinstance(x[0], dict) and "type" in x[0]:
            return [b for b in x if isinstance(b, dict) and "type" in b
                    and not (b["type"] == "text" and "text" not in b)]
        return [{"type": "text", "text": str(s).strip()} for s in x if str(s).strip()]
    s = str(x).strip()
    return [{"type": "text", "text": s}] if s else []


def _is_valid_float(value) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return False
    return True


def _format_prompt(sys_prompt: str, user_prompt: str) -> str:
    return f"<|system|>\n{sys_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n"


# ─────────────────────────────────────────────
# API Key Handler
# ─────────────────────────────────────────────

class APIHandler:
    _ENV_VARS = {"gpt": "OPENAI_API_KEY", "claude": "ANTHROPIC_API_KEY"}

    @staticmethod
    def get_api_key(service: str, manual_key: str = None) -> str:
        if manual_key:
            return manual_key
        env_var = APIHandler._ENV_VARS.get(service.lower())
        if not env_var:
            raise ValueError("Unsupported service. Use 'gpt' or 'claude'.")
        key = os.getenv(env_var)
        if not key:
            raise ValueError(f"Missing API key for {service.upper()}. Set {env_var} or provide a manual key.")
        return key


# ─────────────────────────────────────────────
# Cloud API Base (shared billing + batch logic)
# ─────────────────────────────────────────────

class CloudAPIModel:
    """
    Shared base for GPTModel and ClaudeModel.
    Subclasses must set: model_name, input_cost_per_token, output_cost_per_token,
    and maintain: _input_tokens, _output_tokens (updated in generate_response).
    """
    _input_tokens: int = 0
    _output_tokens: int = 0
    input_cost_per_token: float = 0.0
    output_cost_per_token: float = 0.0

    def get_total_usage(self) -> Dict[str, Any]:
        return {
            "total_input_tokens": self._input_tokens,
            "total_output_tokens": self._output_tokens,
            "total_cost": self._input_tokens * self.input_cost_per_token
                        + self._output_tokens * self.output_cost_per_token,
        }

    async def generate_batch_from_prompts(
        self,
        prompts_to_run: List[Tuple[str, str]],
        temp: float = 0.0,
        concurrency_limit: int = 10,
    ) -> List[str]:
        sem = asyncio.Semaphore(concurrency_limit)

        async def _call(sys_prompt, user_prompt):
            async with sem:
                return await self.generate_response(sys_prompt, user_prompt, temp=temp)

        return list(await asyncio.gather(*[_call(s, u) for s, u in prompts_to_run]))

    async def generate_from_meta_prompts(
        self,
        meta_prompts_with_qid: List[Tuple[Any, str]],
        temp: float = 1.0,
    ):
        sys_prompt = "You are a helpful and unbiased prompt generator, specializing in creating optimized queries."
        texts = await self.generate_batch_from_prompts(
            [(sys_prompt, u) for _, u in meta_prompts_with_qid], temp=temp
        )
        return [(meta_prompts_with_qid[i][0], t.strip()) for i, t in enumerate(texts)]


# ─────────────────────────────────────────────
# GPT Model
# ─────────────────────────────────────────────

class GPTModel(CloudAPIModel):
    MODEL_PRICING = {
        "gpt-4o-2024-08-06":       {"input": 2.5 / 1_000_000,  "output": 10.0 / 1_000_000},
        "gpt-4o-mini-2024-07-18":  {"input": 0.6 / 1_000_000,  "output": 2.4  / 1_000_000},
    }

    def __init__(self, model_name="gpt-4o-mini-2024-07-18", api_key=None):
        self.model_name = model_name
        pricing = self.MODEL_PRICING.get(model_name)
        if not pricing:
            raise ValueError(f"Pricing for model '{model_name}' is not defined.")
        self.input_cost_per_token  = pricing["input"]
        self.output_cost_per_token = pricing["output"]
        self.client = openai.AsyncClient(api_key=APIHandler.get_api_key("gpt", api_key))

    async def generate_response(
        self,
        sys_prompt: str,
        user_prompt: str,
        temp: float = 0.0,
        max_retries: int = 3,
        initial_delay: float = 1.0,
    ) -> str:
        messages = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": user_prompt})

        for attempt in range(max_retries):
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=1024,
                    temperature=temp,
                    timeout=60.0,
                )
                if not resp or not getattr(resp, "choices", None):
                    return ""
                choice = resp.choices[0]
                content = (choice.message.content or "").strip()
                if resp.usage:
                    self._input_tokens  += resp.usage.prompt_tokens
                    self._output_tokens += resp.usage.completion_tokens
                return content

            except (RateLimitError, APITimeoutError, APIConnectionError, APIStatusError) as e:
                if isinstance(e, APIStatusError) and e.status_code < 500:
                    logging.error(f"Non-retryable client error: {e}")
                    return f"ERROR: API Client Error - Status {e.status_code}"
                if attempt == max_retries - 1:
                    logging.error(f"API call failed after {max_retries} attempts: {e}")
                    return "ERROR: API call failed after multiple retries."
                delay = initial_delay * (2 ** attempt)
                logging.warning(f"Retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries}): {e}")
                await asyncio.sleep(delay)

        return "ERROR: An unexpected issue occurred in the retry loop."


# ─────────────────────────────────────────────
# Claude Model
# ─────────────────────────────────────────────

class ClaudeModel(CloudAPIModel):
    MODEL_PRICING = {
        "claude-3-5-haiku-20241022":  {"input": 0.8  / 1_000_000, "output": 4.0  / 1_000_000},
        "claude-sonnet-4-20250514":   {"input": 3.00 / 1_000_000, "output": 15.0 / 1_000_000},
        "claude-3-7-sonnet-20250219": {"input": 3.00 / 1_000_000, "output": 15.0 / 1_000_000},
    }

    def __init__(self, model_name="claude-3-5-haiku-20241022", api_key: str = None):
        self.model_name = model_name
        self.api_key = APIHandler.get_api_key("claude", api_key)
        pricing = self.MODEL_PRICING.get(model_name)
        if not pricing:
            raise ValueError(f"Pricing for model '{model_name}' is not defined.")
        self.input_cost_per_token  = pricing["input"]
        self.output_cost_per_token = pricing["output"]

    async def generate_response(self, sys_prompt, user_prompt, examples=None, temp=0.0) -> str:
        if not user_prompt or not str(user_prompt).strip():
            logging.warning("Skipping API call due to empty user_prompt.")
            return ""

        def call_api():
            client = anthropic.Anthropic(api_key=self.api_key)
            msgs = []

            if examples:
                for pair in examples:
                    try:
                        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                            inp, out = pair[0], pair[1]
                        elif isinstance(pair, dict):
                            inp = pair.get("in") or pair.get("input") or pair.get("user") or ""
                            out = pair.get("out") or pair.get("output") or pair.get("assistant") or ""
                        else:
                            continue
                        u_blocks = _to_blocks(inp)
                        a_blocks = _to_blocks(out)
                        if u_blocks:
                            msgs.append({"role": "user",      "content": u_blocks})
                        if a_blocks:
                            msgs.append({"role": "assistant", "content": a_blocks})
                    except Exception as e:
                        logging.warning(f"[examples] skipped malformed example: {e}")

            user_blocks = _to_blocks(user_prompt)
            if not user_blocks:
                logging.warning("User content became empty after normalization; skipping call.")
                return ""
            msgs.append({"role": "user", "content": user_blocks})

            api_kwargs = {
                "model":       self.model_name,
                "max_tokens":  1024,
                "temperature": float(temp) if temp is not None else 0.0,
                "messages":    msgs,
            }
            if sys_prompt:
                api_kwargs["system"] = str(sys_prompt).strip()

            # Schema validation
            for m in api_kwargs["messages"]:
                if not isinstance(m, dict) or "role" not in m or "content" not in m:
                    raise ValueError("Each message must have role and content.")
                if not isinstance(m["content"], list):
                    raise ValueError("message.content must be a list.")
                for b in m["content"]:
                    if not isinstance(b, dict) or "type" not in b:
                        raise ValueError("Each content block must be a dict with a 'type'.")
                    if b["type"] == "text" and "text" not in b:
                        raise ValueError("Text blocks must include 'text'.")

            try:
                response = client.messages.create(**api_kwargs)
            except BadRequestError as e:
                if "Output blocked by content filtering policy" in str(e):
                    logging.warning(f"Claude response blocked by content filter: {e}")
                    return ""
                raise

            content = ""
            try:
                content = "".join(getattr(block, "text", "") for block in response.content)
            except Exception:
                content = "".join(b.get("text", "") for b in response.content if isinstance(b, dict))

            if response.usage:
                self._input_tokens  += response.usage.input_tokens  or 0
                self._output_tokens += response.usage.output_tokens or 0

            return content

        return await asyncio.to_thread(retry_generate, call_api)


# ─────────────────────────────────────────────
# Local / Self-hosted Models
# ─────────────────────────────────────────────

class BackendType(str, Enum):
    VLLM_LOCAL  = "vllm_local"
    HF_PIPELINE = "hf_pipeline"
    HTTP_VLLM   = "http_vllm"


@dataclass
class SamplingConfig:
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    repetition_penalty: float = 1.0

    def to_vllm(self) -> Dict[str, Any]:
        return {"temperature": self.temperature, "top_p": self.top_p,
                "max_tokens": self.max_tokens, "repetition_penalty": self.repetition_penalty}

    def to_hf(self) -> Dict[str, Any]:
        return {"max_new_tokens": self.max_tokens, "temperature": self.temperature,
                "top_p": self.top_p, "repetition_penalty": self.repetition_penalty,
                "do_sample": self.temperature > 0}


class BaseUnifiedModel:
    """Common async interface for locally-hosted models."""
    backend: BackendType
    model_name: str

    async def generate_response(self, sys_prompt: str, user_prompt: str, temp: float = 0.0) -> str:
        raise NotImplementedError

    async def _generate_texts(self, prompts: List[str], cfg: SamplingConfig) -> List[str]:
        raise NotImplementedError

    async def generate_batch_from_prompts(
        self, prompts_to_run: List[Tuple[str, str]], temp: float = 0.0
    ) -> List[str]:
        cfg = SamplingConfig(temperature=temp, top_p=0.9, max_tokens=1024)
        texts = await self._generate_texts([_format_prompt(s, u) for s, u in prompts_to_run], cfg)
        return [t.strip() for t in texts]

    async def generate_from_meta_prompts(
        self, meta_prompts_with_qid: List[Tuple[Any, str]], temp: float = 1.0
    ):
        sys_prompt = "You are a helpful and unbiased prompt generator, specializing in creating optimized queries."
        prompts = [_format_prompt(sys_prompt, u) for _, u in meta_prompts_with_qid]
        texts = await self._generate_texts(prompts, SamplingConfig(temperature=temp, top_p=0.9, max_tokens=512))
        return [(meta_prompts_with_qid[i][0], t.strip()) for i, t in enumerate(texts)]

    def get_total_usage(self) -> Dict[str, Any]:
        return {"total_prompt_tokens": 0, "total_completion_tokens": 0, "total_cost": 0.0}

    async def close(self):
        pass


class VLLMLocalModel(BaseUnifiedModel):
    def __init__(
        self,
        model_name: str,
        *,
        max_model_len: int = 4096,
        gpu_mem_util: float = 0.9,
        dtype: Optional[str] = None,
        tp_size: Optional[int] = None,
        enforce_eager: bool = True,
    ):
        self.backend = BackendType.VLLM_LOCAL
        self.model_name = model_name
        self.max_model_len = max_model_len
        logging.info(f"[VLLM_LOCAL] Initializing: {model_name}")
        self.llm = LLM(
            model=model_name,
            tokenizer=model_name,
            trust_remote_code=True,
            dtype=dtype or "auto",
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_mem_util,
            tensor_parallel_size=tp_size or 1,
            enforce_eager=enforce_eager,
        )

    def _tok(self):
        return self.llm.get_tokenizer()

    def _tok_len(self, text: str) -> int:
        return len(self._tok().encode(text))

    def _trim_by_tokens(self, text: str, budget: int, keep_head: int = 512) -> str:
        tok = self._tok()
        ids = tok.encode(text)
        if len(ids) <= budget:
            return text
        head = ids[:max(0, min(keep_head, budget))]
        tail = ids[-(budget - len(head)):] if budget > len(head) else []
        return tok.decode(head + tail)

    def _trim_prompts(self, prompts: List[str], soft_margin: int = 96, keep_head: int = 512) -> List[str]:
        budget = max(128, self.max_model_len - soft_margin)
        return [
            self._trim_by_tokens(str(p), budget, keep_head) if self._tok_len(str(p)) > budget else str(p)
            for p in prompts
        ]

    async def _generate_texts(self, prompts: List[str], cfg: SamplingConfig) -> List[str]:
        def _call():
            outs = self.llm.generate(prompts, SamplingParams(**cfg.to_vllm()), use_tqdm=False)
            if random.random() < 0.05:
                gc.collect()
                torch.cuda.empty_cache()
            return [o.outputs[0].text if o.outputs else "" for o in outs]
        return await asyncio.to_thread(_call)

    async def generate_response(self, sys_prompt: str, user_prompt: str, temp: float = 0.0) -> str:
        texts = await self._generate_texts(
            [_format_prompt(sys_prompt, user_prompt)],
            SamplingConfig(temperature=temp, top_p=0.9, max_tokens=1024),
        )
        return texts[0].strip()

    async def close(self):
        logging.info(f"[VLLM_LOCAL] Closing: {self.model_name}")
        try:
            del self.llm
        except Exception:
            pass
        torch.cuda.empty_cache()
        gc.collect()


class HFPipelineModel(BaseUnifiedModel):
    def __init__(self, model_name="meta-llama/Meta-Llama-3.1-8B", dtype: Optional[str] = "bfloat16", device_map="auto"):
        self.backend = BackendType.HF_PIPELINE
        self.model_name = model_name
        _DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        t_dtype = _DTYPE_MAP.get(dtype) if dtype else None
        self.pipe = transformers.pipeline(
            "text-generation",
            model=model_name,
            model_kwargs={"torch_dtype": t_dtype} if t_dtype is not None else {},
            device_map=device_map,
        )

    async def _generate_texts(self, prompts: List[str], cfg: SamplingConfig) -> List[str]:
        def _call():
            results = self.pipe(prompts, **cfg.to_hf())
            if isinstance(results, dict):
                results = [results]
            return [(r[0] if isinstance(r, list) else r).get("generated_text", "") for r in results]
        return await asyncio.to_thread(_call)

    async def generate_response(self, sys_prompt: str, user_prompt: str, temp: float = 0.0) -> str:
        texts = await self._generate_texts(
            [_format_prompt(sys_prompt, user_prompt)],
            SamplingConfig(temperature=temp, top_p=0.9, max_tokens=1024),
        )
        return texts[0]


class HttpVLLMModel(BaseUnifiedModel):
    """Calls a FastAPI/vLLM server at POST {url}/v1/generateText."""

    def __init__(self, model_name: str, endpoint_url: str, *, timeout: Optional[float] = None, max_retries: int = 2):
        self.backend = BackendType.HTTP_VLLM
        self.model_name = model_name
        base = endpoint_url.rstrip("/")
        self.url = base if base.endswith("/v1/generateText") else f"{base}/v1/generateText"
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(timeout=timeout)

    async def _generate_texts(self, prompts: List[str], cfg: SamplingConfig) -> List[str]:
        payload = {"prompts": prompts, "sampling_params": cfg.to_vllm()}
        for attempt in range(self.max_retries + 1):
            try:
                r = await self._client.post(self.url, json=payload)
                r.raise_for_status()
                return [o.get("text", "") for o in r.json().get("result", [])]
            except Exception as e:
                if attempt == self.max_retries:
                    raise RuntimeError(f"HTTP generation failed after {self.max_retries + 1} attempts: {e}") from e
                await asyncio.sleep(min(1.5 ** (attempt + 1), 5.0))

    async def generate_response(self, sys_prompt: str, user_prompt: str, temp: float = 0.0) -> str:
        texts = await self._generate_texts(
            [_format_prompt(sys_prompt, user_prompt)],
            SamplingConfig(temperature=temp, top_p=0.9, max_tokens=1024),
        )
        return texts[0].strip() if texts else ""

    async def close(self):
        await self._client.aclose()


