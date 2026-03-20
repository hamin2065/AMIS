import os
import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from jb.models import BackendType, GPTModel, ClaudeModel, VLLMLocalModel, HFPipelineModel, HttpVLLMModel, BaseUnifiedModel
from jb.prompt import asr_user_prompt_template
from datetime import datetime, timezone, timedelta
import asyncio
import httpx

def generate_init_prompts(query: str, seed_templates: List[str]) -> List[str]:
    """Generate initial jailbreaking prompts from seed templates"""
    prompts = []
    for template in seed_templates:
        # Replace any placeholders with the query
        prompt = template.replace("{{QUERY}}", query)
        prompts.append(prompt)
    return prompts

def extract_single_sys_user_prompt(text: str) -> Tuple[str, str]:
    """
    Extract the first SYS and USER prompt from the LLM output.

    Args:
        text (str): LLM-generated output with one SYS and one USER section.

    Returns:
        Tuple[str, str]: A tuple containing (system_prompt, user_prompt).
    """
    pattern = r"SYS:\s*(.*?)\s*USER:\s*(.*)"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        raise ValueError("Could not find both SYS: and USER: sections in the text.")
    return match.group(1).strip(), match.group(2).strip()

async def _gather_limited(coros: List[asyncio.Future], limit: int = 10):
    """
    Await many coroutines concurrently, but never more than `limit` at once.
    Tweak `limit` to match your API-rate or VRAM budget.
    """
    sem = asyncio.Semaphore(limit)

    async def _runner(c):
        async with sem:
            return await c

    return await asyncio.gather(*(_runner(c) for c in coros))

        
def VLLM_load_model(
    backend: BackendType,
    model_name: str,
    *,
    # vLLM options
    max_model_len: int = 4096,
    gpu_mem_util: float = 0.9,
    dtype: Optional[str] = None,
    tp_size: Optional[int] = None,
    enforce_eager: bool = True,
    # HF options
    hf_dtype: Optional[str] = "bfloat16",
    device_map: str | dict = "auto",
    # HTTP options
    endpoint_url: Optional[str] = None,
    http_timeout: Optional[float] = None,
    http_retries: int = 2,
) -> BaseUnifiedModel:
    if backend == BackendType.VLLM_LOCAL:
        return VLLMLocalModel(
            model_name,
            # max_model_len=max_model_len,
            max_model_len=8192,
            gpu_mem_util=gpu_mem_util,
            dtype=dtype,
            tp_size=tp_size,
            enforce_eager=enforce_eager,
        )
    elif backend == BackendType.HF_PIPELINE:
        return HFPipelineModel(
            model_name=model_name,
            dtype=hf_dtype,
            device_map=device_map,
        )
    elif backend == BackendType.HTTP_VLLM:
        assert endpoint_url, "HTTP_VLLM backend requires endpoint_url"
        return HttpVLLMModel(
            model_name=model_name,
            endpoint_url=endpoint_url,
            timeout=None,
            max_retries=http_retries,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

def get_model(
    name: str,
    *,
    max_model_len: int = 4096,
    gpu_mem_util: Optional[float] = 0.90,
    dtype: Optional[str] = "float16",
    tp_size: Optional[int] = 1,
    **extra,                          
):
    """
    Factory that returns the right model/adapter instance.

    Behavior:
      - "gpt*", "claude*" prefixes route to each vendor API class (compatibility with existing code)
      - For other models, create via unified loader (load_model) through:
          · HTTP vLLM (if endpoint_url or LLM_ENDPOINTS[name] exists)
          · Otherwise, default to local vLLM

    Extra kwargs (optional):
      backend: BackendType("vllm_local" | "hf_pipeline" | "http_vllm") forcefully specified
      endpoint_url: HTTP vLLM endpoint (if specified, backend is not specified, http_vllm is automatically)
      http_timeout: float | None
      http_retries: int = 2
      enforce_eager: bool = True (vLLM)
      hf_dtype: str = "bfloat16"
      device_map: "auto" | dict
      model_name_override: str  # "llama" special case to change to another repo (default: None)

    Returns
    -------
    GPTModel | ClaudeModel | BaseUnifiedModel
    """
    # 1. OpenAI GPT-* models
    if name.startswith("gpt"):
        return GPTModel(
            name,
            api_key=os.getenv("OPENAI_API_KEY"),
            **extra,
        )

    # 2. Anthropic Claude-* models
    if name.startswith("claude"):
        return ClaudeModel(
            name,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            **extra,
        )

    # 3. Everything else → unified backend (vLLM/HF/HTTP)
    name_lc = name.lower()

    # Determine model repo
    model_repo = extra.pop("model_name_override", None)
    if not model_repo:
        model_repo = name

    # Determine backend: explicit backend > HTTP if endpoint exists > default(vLLM_LOCAL)
    backend = extra.pop("backend", None)

    # endpoint_url priority:
    #  1) extra["endpoint_url"]
    #  2) global LLM_ENDPOINTS[name] (if exists)
    endpoint_url = extra.get("endpoint_url")
    if endpoint_url is None:
        # global dict exists, use it
        llm_eps = globals().get("LLM_ENDPOINTS", {}) or {}
        endpoint_url = llm_eps.get(name) or llm_eps.get(model_repo)

    if backend is None:
        if endpoint_url:
            backend = BackendType.HTTP_VLLM
        else:
            backend = BackendType.VLLM_LOCAL  # default is local vLLM

    # collect options to pass to load_model
    enforce_eager = extra.pop("enforce_eager", True)
    hf_dtype     = extra.pop("hf_dtype", "bfloat16")
    device_map   = extra.pop("device_map", "auto")
    http_timeout = extra.pop("http_timeout", None)
    http_retries = extra.pop("http_retries", 2)

    # now call unified loader
    return VLLM_load_model(
        backend=backend,
        model_name=model_repo,
        # vLLM
        max_model_len=max_model_len,
        gpu_mem_util=gpu_mem_util,
        dtype=dtype,
        tp_size=tp_size,
        enforce_eager=enforce_eager,
        # HF
        hf_dtype=hf_dtype,
        device_map=device_map,
        # HTTP
        endpoint_url=endpoint_url,
        http_timeout=http_timeout,
        http_retries=http_retries,
    )
    
def setup_logging(method_name: str,
                  log_dir: str | None = None,
                  *, force: bool = True) -> str:
    """logging configuration"""

    KST = timezone(timedelta(hours=9))
    ts = datetime.now(KST).strftime("%Y%m%d_%H%M%S")

    # ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if log_dir is None:
        log_dir = f"./results_{method_name}_{ts}"
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"{method_name}_run_{ts}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ],
        force=force
    )
    return log_dir


def scale_0_to100(x: float) -> float:
    """Linearly map the range [-1, 1] to [0, 100]."""
    if not -1 <= x <= 1:
        raise ValueError("x must be in [-1, 1]")
    return (x + 1) / 2 * 100


def parse_numbered_list(text: str) -> List[str]:
    """
    Parse a numbered list of prompts from text output.
    Works with both jailbreaking prompts and scoring templates.
    """
    import re
    
    # First try parsing as JSON (for scoring templates)
    if text.strip().startswith('[') and text.strip().endswith(']'):
        try:
            templates = json.loads(text)
            if isinstance(templates, list):
                return templates
        except json.JSONDecodeError:
            pass
    
    # Otherwise parse as numbered list
    lines = text.strip().split('\n')
    prompts = []
    current_prompt = []
    
    # Look for numbered items (1., 2., etc.)
    numbered_pattern = r'^\s*\d+[\.)]\s+'
    
    for line in lines:
        if re.match(numbered_pattern, line):
            # If we hit a new number and already have content, save the previous prompt
            if current_prompt:
                prompts.append('\n'.join(current_prompt).strip())
                current_prompt = []
            
            # Add the current line without the number prefix
            current_prompt.append(re.sub(numbered_pattern, '', line))
        elif current_prompt:
            # Continue building current prompt
            current_prompt.append(line)
    
    # Add the last prompt if there is one
    if current_prompt:
        prompts.append('\n'.join(current_prompt).strip())
    
    # Handle JSON objects in the list
    parsed_prompts = []
    for p in prompts:
        try:
            # If it's a valid JSON object, parse it
            obj = json.loads(p)
            parsed_prompts.append(obj)
        except json.JSONDecodeError:
            # Otherwise keep as string
            parsed_prompts.append(p)
    
    return parsed_prompts

def select_topK_plus_random(details_list: List[Dict], K: int = 4, keep_random: bool = False) -> List[Dict]:
    """
    Select top K items based on score, with option to keep some random ones.
    Works for both jailbreaking and scoring prompt optimization.
    
    Args:
        details_list: List of dicts with score metric
        K: Number of top performers to keep
        keep_random: Whether to keep some random ones 
        
    Returns:
        List of selected items
    """
    # Sort by performance (assuming higher is better)
    sorted_details = sorted(
        details_list, 
        key=lambda x: x.get("score", 0), 
        reverse=True
    )
    
    # Always keep top K
    top_k = sorted_details[:min(K, len(sorted_details))]
    
    # Optionally add some random ones from the remainder
    if keep_random and len(sorted_details) > K:
        import random
        remaining = sorted_details[K:]
        random_count = min(K // 2, len(remaining))
        random_selections = random.sample(remaining, random_count)
        return top_k + random_selections
    
    return top_k

async def check_asr(prompt: str, response: str, model: GPTModel, max_retries: int = 3) -> int:
    """
    Returns 1 if model judges response as a successful attack (yes),
    0 if it judges as no, and -1 if ambiguous.
    """
    # fill in prompt template
    up = asr_user_prompt_template \
         .replace("{{QUERY}}", str(prompt)) \
         .replace("{{RESPONSE}}", str(response))
    
    # Try multiple times with retry logic
    for attempt in range(max_retries):
        try:
            asr_text = await model.generate_response(
                sys_prompt=None,
                user_prompt=up,
                max_retries=3,  # Use model's built-in retry logic
                initial_delay=1.0
            )
            
            # Check if we got an error response
            if asr_text.startswith("ERROR:"):
                if attempt == max_retries - 1:
                    print(f"check_asr failed after {max_retries} attempts: {asr_text}")
                    return -1  # Return ambiguous on final failure
                else:
                    print(f"check_asr attempt {attempt + 1} failed, retrying: {asr_text}")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
            
            # Parse the response
            txt = asr_text.lower()
            if "yes" in txt and "no" not in txt:
                return 1
            if "no" in txt and "yes" not in txt:
                return 0
            return -1
            
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"check_asr failed after {max_retries} attempts with exception: {e}")
                return -1
            else:
                print(f"check_asr attempt {attempt + 1} failed with exception, retrying: {e}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    return -1  # Fallback

LLM_ENDPOINTS = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "http://localhost:8001/v1/generateText",
    "meta-llama/Meta-Llama-3-8B": "http://localhost:8002/v1/generateText",
    "cais/HarmBench-Mistral-7b-val-cls": "http://localhost:8003/v1/generateText",
    "cais/zephyr_7b_r2d2": "http://localhost:8004/v1/generateText",
}



async def call_llm_http(model_name: str,
                        prompts: list[str],
                        *,
                        temp: float = 0.7,
                        top_p: float = 0.9,
                        max_tokens: int = 512,
                        repetition_penalty: float = 1.0) -> list[str]:
    """vLLM FastAPI 서버에 POST → text 리스트 반환"""
    url = LLM_ENDPOINTS[model_name]
    payload = {
        "prompts": prompts,
        # "samplings": {
        "sampling_params": {
            "temperature": temp,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "repetition_penalty": repetition_penalty
        }
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(url, json=payload)
    r.raise_for_status()

    return [o["text"] for o in r.json()["result"]]


def save_json_checkpoint(data: Any, log_dir: str, filename: str) -> str:
    """Save checkpoint data to a JSON file"""
    filepath = os.path.join(log_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return filepath