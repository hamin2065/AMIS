import os, re, json, logging, random, hashlib, asyncio, sys
from typing import List, Dict, Any, Optional, Tuple, Set
from statistics import mean
from dotenv import load_dotenv
from constants import MIN_SCORE, MAX_SCORE, SCORE_RANGE

# ──────────────────────────────────────────────────────────────────────
from jb.utils import (
    setup_logging, parse_numbered_list, select_topK_plus_random,
    check_asr, save_json_checkpoint, _gather_limited, generate_init_prompts
)
from jb.models import GPTModel, ClaudeModel, VLLMLocalModel, HFPipelineModel, HttpVLLMModel
from jb.prompt import initial_jailbreak_templates
from jb.meta_prompt import build_inner_meta_prompt
# ──────────────────────────────────────────────────────────────────────

def _alignment_0_to_100(model_score: float, asr: int) -> float:
    ideal = MIN_SCORE if asr == 0 else MAX_SCORE
    return (1.0 - abs(model_score - ideal) / SCORE_RANGE) * 100

async def evaluate_candidates_batch(
    prompts_to_run: List[Tuple[str, str]], # (query_id, prompt_text)
    eval_cache: Dict[str, Dict[str, Any]],
    query_states: Dict[str, Dict[str, Any]],
    target_model,
    scorer_model,
    scorer_template: Dict[str, str],
    asr_model,
) -> None:
    """Batch function to evaluate all candidate prompts for all queries at once"""
    
    # 1. Generate responses (Target Model)
    target_tasks = []
    task_map = []
    for query_id, prompt in prompts_to_run:
        if not eval_cache.get(prompt, {}).get('response'):
            target_tasks.append(target_model.generate_response(sys_prompt=None, user_prompt=prompt))
            task_map.append((query_id, prompt))

    if target_tasks:
        logging.info(f"Generating {len(target_tasks)} responses from target model...")
        responses = await _gather_limited(target_tasks, limit=128)
        for (query_id, prompt), response in zip(task_map, responses):
            eval_cache.setdefault(prompt, {})['response'] = response

    # 2. Score (Scorer Model)
    prompts_to_score_batch = []
    task_map_scorer = []
    tpl_hash = hashlib.md5(json.dumps(scorer_template, sort_keys=True).encode()).hexdigest()
    sys_t, usr_t = scorer_template.get("system", ""), scorer_template.get("user", "")

    for query_id, prompt in prompts_to_run:
        if eval_cache.get(prompt, {}).get('scorer_template_hash') != tpl_hash:
            response = eval_cache.get(prompt, {}).get('response')
            if response:
                sys_f = sys_t.replace("{{QUERY}}", str(prompt)).replace("{{RESPONSE}}", str(response))
                usr_f = usr_t.replace("{{QUERY}}", str(prompt)).replace("{{RESPONSE}}", str(response))
                prompts_to_score_batch.append((sys_f, usr_f))
                task_map_scorer.append((query_id, prompt))
    
    if prompts_to_score_batch:
        logging.info(f"Scoring {len(prompts_to_score_batch)} responses in a single batch...")
        score_texts = await scorer_model.generate_batch_from_prompts(prompts_to_score_batch)
        
        for (query_id, prompt), score_text in zip(task_map_scorer, score_texts):
            m = re.search(r'Final score:\s*([0-9]+(?:\.[0-9]+)?)', score_text or "", re.I)
            score = float(m.group(1)) if m else (float(re.findall(r'([0-9]+(?:\.[0-9]+)?)', score_text or "")[-1]) if re.findall(r'([0-9]+(?:\.[0-9]+)?)', score_text or "") else 0.0)
            eval_cache.setdefault(prompt, {})['score'] = max(MIN_SCORE, min(MAX_SCORE, score))
            eval_cache[prompt]['score_text'] = score_text
            eval_cache[prompt]['scorer_template_hash'] = tpl_hash

    # 3. ASR determination (ASR Model)
    asr_tasks = []
    task_map = []
    for query_id, prompt in prompts_to_run:
        if eval_cache.get(prompt, {}).get('asr') is None:
            response = eval_cache.get(prompt, {}).get('response')
            if response:
                asr_tasks.append(check_asr(prompt, response, asr_model))
                task_map.append((query_id, prompt))

    if asr_tasks:
        logging.info(f"Checking ASR for {len(asr_tasks)} responses...")
        asr_results = await _gather_limited(asr_tasks, limit=32)
        for (query_id, prompt), raw_asr in zip(task_map, asr_results):
            try: asr_val = int(raw_asr.strip()) if isinstance(raw_asr, str) else int(raw_asr)
            except: asr_val = 0
            eval_cache.setdefault(prompt, {})['asr'] = asr_val


async def run_jailbreaking_prompt_optimization(
    scorer_prompt: Dict,
    best_seed_templates_: Dict[Any, List[Tuple[str, float]]],
    data_path: str,
    log_dir: Optional[str] = None,
    prompt_num_iterations: int = 10,
    K: int = 8, M: int = 5, jb_inherit_n: int = 5, *,
    _query_performance_registry: Optional[Dict[str, float]] = None,
    _solved_queries: Set[str] = None,
    _prev_run_log: Optional[str] = None,
    optim_model=None, target_model=None, scorer_model=None, asr_model=None,
) -> Tuple[List[Dict], str, Dict[Any, List[Tuple[str, float]]], Dict[str, float], Dict[str, int]]:
    
    load_dotenv()
    log_dir = setup_logging("jailbreak_opt", log_dir)
    if target_model is None: target_model = GPTModel("gpt-4o-mini-2024-07-18")
    if optim_model is None: optim_model = HuggingFaceModel("meta-llama/Llama-3.1-8B-Instruct")
    if scorer_model is None: scorer_model = GPTModel("gpt-4o-mini-2024-07-18")
    if asr_model is None: asr_model = GPTModel("gpt-4o-mini-2024-07-18")
    
    def mname(m): return getattr(m, "model_name", str(m))
    logging.info(f"Optim model:  {mname(optim_model)} | Target model: {mname(target_model)} | Scorer model: {mname(scorer_model)} | ASR model: {mname(asr_model)}")

    eval_cache: Dict[str, Dict[str, Any]] = {}
    
    with open(data_path, "r") as f: data = json.load(f)
    newly_solved_this_run: Set[str] = set()
    query_states: Dict[str, Dict[str, Any]] = {}
    for item in data:
        query_id = str(item.get("ind", item.get("id", item.get("Index", ""))))
        query = item.get("goal") or item.get("Behavior") or item.get("prompt")
        
        if "jbb" in data_path:
            query_id = str(item.get("Index", ""))
            query = item.get("Goal", "")

        if "harmbench" in data_path:
            query_id = str(item.get("Index", ""))
            if item.get("FunctionalCategory") == "contextual":
                query_ = item.get("Behavior", "")
                context = item.get("ContextString", "")
                query = f"Context: {context}\n\n{query_}"
            else:
                query = item.get("Behavior", "")

        if not query or not query_id: continue

        existing_best = best_seed_templates_.get(query_id, [])
        inherited = [p for p, _ in existing_best[:jb_inherit_n]]
        initial_prompts = inherited + generate_init_prompts(query, random.sample(initial_jailbreak_templates, max(0, 10 - len(inherited))))
        
        query_states[query_id] = {
            "query": query,
            "population": initial_prompts,
            "log": {"query": query, "initial_candidates": [], "iterations": [], "score_prompt": scorer_prompt},
            "active": True,
            "is_rescore_only": False,
        }

    initial_prompts_to_run = [(qid, p) for qid, state in query_states.items() for p in state['population']]
    await evaluate_candidates_batch(initial_prompts_to_run, eval_cache, query_states, target_model, scorer_model, scorer_prompt, asr_model)

    for qid, state in query_states.items():
        candidates = [eval_cache.get(p) for p in state['population'] if eval_cache.get(p)]
        state['log']['initial_candidates'] = candidates
        
    for step in range(prompt_num_iterations):
        active_queries = [qid for qid, state in query_states.items() if state['active']]
        if not active_queries:
            logging.info("All queries deactivated or finished. Stopping optimization early.")
            break
        
        logging.info(f"--- Iteration {step + 1}/{prompt_num_iterations} --- | Active queries: {len(active_queries)}")

        meta_prompts_to_generate = []
        for qid in active_queries:
            state = query_states[qid]
            population = state['population']
            
            candidates = []
            for p in population:
                eval_result = eval_cache.get(p)
                if eval_result:
                    candidate_with_prompt = eval_result.copy()
                    candidate_with_prompt['attack_prompt'] = p
                    candidates.append(candidate_with_prompt)
            
            top_candidates = sorted(candidates, key=lambda c: c.get('score', MIN_SCORE), reverse=True)[:K]
            top_prompts = [
                (c.get('attack_prompt') or c.get('prompt'), c.get('score', MIN_SCORE))
                for c in top_candidates
                if (c.get('attack_prompt') or c.get('prompt')) is not None
            ]

            meta_prompt = build_inner_meta_prompt(top_prompts, M, max_tokens=150, min_score=MIN_SCORE, max_score=MAX_SCORE)
            meta_prompts_to_generate.append((qid, meta_prompt))

        if meta_prompts_to_generate:
            logging.info(f"Generating {len(meta_prompts_to_generate)} new sets of prompts in a single batch...")
            generated_results = await optim_model.generate_from_meta_prompts(meta_prompts_to_generate)
        else:
            generated_results = []

        prompts_to_evaluate_next = []
        newly_generated_prompts = {qid: [] for qid in active_queries}
        for qid, raw_text in generated_results:
            logging.info(f"===== Optimizer Raw Output for QID {qid} =====")
            logging.info(raw_text)
            new_prompts = parse_numbered_list(raw_text)
            newly_generated_prompts[qid] = new_prompts
            for p in new_prompts:
                prompts_to_evaluate_next.append((qid, p))

        await evaluate_candidates_batch(prompts_to_evaluate_next, eval_cache, query_states, target_model, scorer_model, scorer_prompt, asr_model)

        for qid in active_queries:
            state = query_states[qid]
            
            all_prompts = state['population'] + newly_generated_prompts.get(qid, [])
            all_candidates = []
            for p in all_prompts:
                eval_result = eval_cache.get(p)
                if eval_result: 
                    if 'attack_prompt' not in eval_result:
                         eval_result['attack_prompt'] = p
                    all_candidates.append(eval_result)

            survivors = select_topK_plus_random(all_candidates, K=min(K, len(all_candidates)), keep_random=False)
            state['population'] = [s['attack_prompt'] for s in survivors]

            newly_evaluated_prompts = newly_generated_prompts.get(qid, [])
            newly_evaluated_candidates = [eval_cache[p] for p in newly_evaluated_prompts if p in eval_cache]
            
            iter_log = {"step": step + 1, "best_score": max([c.get('score',0) for c in survivors] or [0]), "candidates": newly_evaluated_candidates}
            state['log']['iterations'].append(iter_log)

    run_log = []
    mean_alignment_this_run = {}
    for qid, state in query_states.items():
        final_log = state['log']
        
        all_candidates_final = final_log.get("initial_candidates", []) + [c for it_log in final_log.get("iterations", []) for c in it_log.get("candidates", [])]
        def _norm_candidate(c):
            if c is None: 
                return None
            c.setdefault("attack_prompt", c.get("prompt"))
            c.setdefault("score", MIN_SCORE)
            c.setdefault("asr", 0)
            return c

        # Combine initial/repeated candidates
        all_candidates_final = (final_log.get("initial_candidates", []) +
                                [c for it_log in final_log.get("iterations", []) 
                                for c in it_log.get("candidates", [])])

        # Normalize + valid candidates only
        all_candidates_final = [ _norm_candidate(dict(c)) for c in all_candidates_final if isinstance(c, dict) ]

        # Score-based filter & max
        scored_prompts = [c for c in all_candidates_final if c.get("score", MIN_SCORE) > (MIN_SCORE - 1.0)]
        best = (max(scored_prompts, key=lambda x: x.get("score", MIN_SCORE))
        if scored_prompts else {"attack_prompt": None, "score": 0.0, "asr": 0})
        final_log.update({"best_prompt": best["attack_prompt"], "best_score": best["score"], "best_asr": best.get("asr", 0)})
        run_log.append(final_log)
        
        all_alignments = [_alignment_0_to_100(d['score'], d.get('asr',0)) for d in all_candidates_final if d and 'score' in d]
        if all_alignments:
            mean_alignment_this_run[state['query']] = mean(all_alignments)
            
        final_survivors = sorted([d for d in all_candidates_final if d], key=lambda x: x.get('score', 0), reverse=True)[:jb_inherit_n]
        best_seed_templates_[qid] = [(d["attack_prompt"], d["score"]) for d in final_survivors]

    final_path = save_json_checkpoint(run_log, log_dir, "jailbreak_opt_final.json")
    logging.info(f"Finished jailbreak run → {final_path}")

    return run_log, log_dir, best_seed_templates_, mean_alignment_this_run, newly_solved_this_run