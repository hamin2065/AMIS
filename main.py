#!/usr/bin/env python3
import os, re, argparse, json, logging, hashlib, asyncio, random, sys

from typing import List, Dict, Any, Optional, Tuple, Set
from statistics import mean
from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────────────────
from jb.models import GPTModel, ClaudeModel, VLLMLocalModel, HFPipelineModel, HttpVLLMModel
from jb.utils import (
    setup_logging, parse_numbered_list, save_json_checkpoint, check_asr,
    get_model, extract_single_sys_user_prompt
)
from jb.meta_prompt import build_scorer_meta_prompt
from jb.prompt import initial_scorer_templates
from inner import run_jailbreaking_prompt_optimization
from constants import MIN_SCORE, MAX_SCORE, SCORE_RANGE
# ──────────────────────────────────────────────────────────────────────

async def evaluate_scoring_template(
    template: Dict,
    json_data_path: str,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Using the run_log created by inner, calculate the alignment.
    Calculate the 'mean' alignment for each query and return it.
    """
    def _alignment_0_to_100(model_score: float, asr: int) -> float:
        ideal = MIN_SCORE if asr == 0 else MAX_SCORE
        return (1.0 - abs(model_score - ideal) / SCORE_RANGE) * 100

    with open(json_data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    per_query_results = []
    align_asr0, align_asr1 = [], []
    per_query_mean_alignment: Dict[str, float] = {}

    for entry in data:
        query = entry.get("query", "")
        cand_all = []
        for d in entry.get("initial_candidates", []) or []:
            if d and d.get("attack_prompt") and d.get("response") is not None and d.get("score") is not None and d.get("asr") is not None:
                cand_all.append(d)
        for it in entry.get("iterations", []) or []:
            for c in it.get("candidates", []) or []:
                if c and c.get("attack_prompt") and c.get("response") is not None and c.get("score") is not None and c.get("asr") is not None:
                    cand_all.append(c)

        if not cand_all:
            continue

        q_snaps = []
        query_alignments = []
        for c in cand_all:
            sc = float(c["score"])
            asr_val = int(c["asr"])
            sc = max(MIN_SCORE, min(MAX_SCORE, sc))
            align = _alignment_0_to_100(sc, asr_val)
            query_alignments.append(align)
            if asr_val == 1:
                align_asr1.append(align)
            else:
                align_asr0.append(align)
            q_snaps.append({
                "prompt": c["attack_prompt"], "response": c.get("response", ""),
                "score": sc, "asr": asr_val, "alignment": align
            })

        if query_alignments:
            per_query_mean_alignment[query] = mean(query_alignments)

        per_query_results.append({
            "query": query,
            "sampled_candidates": q_snaps
        })

    avg_asr0 = mean(align_asr0) if align_asr0 else 0.0
    avg_asr1 = mean(align_asr1) if align_asr1 else 0.0
    num_broken_queries = len({c['query'] for c in per_query_results for s in c['sampled_candidates'] if s['asr'] == 1})

    num_asr0 = len(align_asr0)
    num_asr1 = len(align_asr1)
    total_samples = num_asr0 + num_asr1

    all_alignments = align_asr0 + align_asr1
    overall_alignment = mean(all_alignments) if all_alignments else 0.0

    result = {
        "template": template, "per_query": per_query_results,
        "overall_alignment": overall_alignment, 
        "avg_alignment_asr0": avg_asr0,
        "avg_alignment_asr1": avg_asr1,
        "total_prompts_checked": sum(len(q['sampled_candidates']) for q in per_query_results),
        "num_broken_queries": num_broken_queries,
    }
    return result, per_query_mean_alignment


async def run_scoring_prompt_optimization(
    initial_templates: List[Dict],
    jailbreak_data_path: str,
    best_seed_templates: Dict[Any, List[Tuple[str, float]]],
    scoring_iterations: int = 5,
    scoring_topK: int = 4,
    scoring_M: int = 1,
    jailbreaking_iterations: int = 10,
    jailbreaking_K: int = 8,
    jailbreaking_M: int = 5,
    jb_inherit_n: int = 5,
    log_dir: Optional[str] = None,
    score_optim_model=None,
    optim_model=None,
    target_model=None,
    scorer_model=None,
    asr_model=None,
    initial_templates_path: Optional[str] = None,
) -> Tuple[List[Dict], str, Dict[Any,List[Tuple[str,float]]], Dict[str, float], Dict[str, int]]:

    load_dotenv()
    log_dir = setup_logging("scoring_opt", log_dir)

    if score_optim_model is None: score_optim_model = get_model("gpt-4o-mini-2024-07-18")
    if optim_model is None: optim_model = get_model("meta-llama/Llama-3.1-8B-Instruct")
    if target_model is None: target_model = get_model("claude-3-5-haiku-20241022")
    if scorer_model is None: scorer_model = get_model("gpt-4o-mini-2024-07-18")
    if asr_model is None: asr_model = get_model("gpt-4o-mini-2024-07-18")

    population = list(initial_templates)
    run_history = []
    template_registry: Dict[str, Any] = {}
    
    query_performance_registry: Dict[str, float] = {}
    solved_queries: Set[str] = set() 
    total_cost_data = {
        "scorer_model": 0, "target_model": 0, "score_optim_model": 0, "asr_model": 0
    }

    if initial_templates_path:
        with open(initial_templates_path, "r") as f:
            data_ = json.load(f)
        if isinstance(data_, dict) and "template_registry" in data_:
            template_registry = data_["template_registry"]
            query_performance_registry = data_.get("query_performance_registry", {})
            solved_queries = set(data_.get("solved_queries", []))
            total_cost_data = data_.get("total_cost_data", total_cost_data)
            logging.info(f"Loaded checkpoint registry={len(template_registry)}, known_queries={len(query_performance_registry)}")
        elif isinstance(data_, list) and data_:
            template = data_[0].get("score_prompt")
            if template:
                tpl_str = json.dumps(template, sort_keys=True)
                tpl_id = hashlib.md5(tpl_str.encode()).hexdigest()[:8]
                perf, mean_alignments = await evaluate_scoring_template(template, initial_templates_path)
                template_registry[tpl_id] = {
                    "template": template, "overall_alignment": perf["overall_alignment"], "details": perf,
                    "jailbreak_results": data_, "jb_dir": os.path.dirname(initial_templates_path)
                }
                for query, align in mean_alignments.items():
                    query_performance_registry[query] = max(query_performance_registry.get(query, 0.0), align)
                logging.info(f"Imported one run from {initial_templates_path}; known_queries={len(query_performance_registry)}")

    else:
        logging.info("Evaluating initial scoring templates...")
        for template in population:
            tpl_str = json.dumps(template, sort_keys=True)
            tpl_id  = hashlib.md5(tpl_str.encode()).hexdigest()[:8]
            jb_dir = os.path.join(log_dir, f"jailbreak_{tpl_id}")

            run_log, _, best_seed_templates, mean_alignments_this_run, newly_solved = await run_jailbreaking_prompt_optimization(
                scorer_prompt=template, best_seed_templates_=best_seed_templates,
                data_path=jailbreak_data_path, log_dir=jb_dir, prompt_num_iterations=jailbreaking_iterations,
                K=jailbreaking_K, M=jailbreaking_M, jb_inherit_n=jb_inherit_n,
                _query_performance_registry=query_performance_registry,
                _prev_run_log=None,
                optim_model=optim_model, target_model=target_model,
                scorer_model=scorer_model, asr_model=asr_model,
                _solved_queries=solved_queries
            )
            solved_queries.update(newly_solved)

            for query, align in mean_alignments_this_run.items():
                query_performance_registry[query] = max(query_performance_registry.get(query, 0.0), align)

            temp_data_path = os.path.join(jb_dir, "jailbreak_opt_final.json")
            perf, mean_alignments = await evaluate_scoring_template(template, temp_data_path)
            for query, align in mean_alignments.items():
                query_performance_registry[query] = max(query_performance_registry.get(query, 0.0), align)

            template_registry[tpl_id] = {
                "template": template, "overall_alignment": perf["overall_alignment"], "details": perf,
                "jailbreak_results": run_log, "jb_dir": jb_dir,
            }
        save_json_checkpoint(best_seed_templates, log_dir, f"best_seed_templates_0.json")
        current_total_cost_data = {
            "scorer_model": scorer_model.get_total_usage(),
            "target_model": target_model.get_total_usage(),
            "score_optim_model": score_optim_model.get_total_usage(),
            "asr_model": asr_model.get_total_usage(),
        }
        checkpoint_path = save_json_checkpoint(
            {
                "run_history": run_history, "template_registry": template_registry,
                "query_performance_registry": query_performance_registry,
                "solved_queries": list(solved_queries), 
                "total_cost_data": current_total_cost_data
            },
            log_dir, f"scoring_opt_checkpoint_0.json"
        )

    for it in range(scoring_iterations):
        logging.info(f"--- Scoring Optimization Iteration {it+1}/{scoring_iterations} ---")
        if not template_registry:
            logging.warning("No templates in registry; skipping iteration.")
            break
        top_ids = sorted(template_registry, key=lambda tid: template_registry[tid]["overall_alignment"], reverse=True)[:scoring_topK]
        top_pairs = [(template_registry[tid]["template"], template_registry[tid]["overall_alignment"]) for tid in top_ids]
        meta_prompt = build_scorer_meta_prompt(top_pairs, scoring_topK, MIN_SCORE, MAX_SCORE)
        try:
            raw_out = await score_optim_model.generate_response(
                sys_prompt="You are an expert prompt architect.", user_prompt=meta_prompt, temp=1.0
            )
            new_scorer_templates = []
            prompts = re.findall(r"SYS: (.*?)\nUSER: (.*?)(?=\nSYS:|\Z)", raw_out, re.DOTALL)
            for sys_prompt, user_prompt in prompts:
                new_scorer_templates.append({"system": sys_prompt.strip(), "user": user_prompt.strip()})
            if not new_scorer_templates and "SYS:" in raw_out and "USER:" in raw_out:
                 sys_p, user_p = extract_single_sys_user_prompt(raw_out)
                 new_scorer_templates.append({"system": sys_p, "user": user_p})

        except Exception as e:
            logging.error(f"Failed to generate new scorer template: {e}")
            new_scorer_templates = []
        
        prev_best_id = top_ids[0] if top_ids else None
        prev_run_log_path = None
        if prev_best_id and template_registry[prev_best_id].get("jb_dir"):
            candidate = os.path.join(template_registry[prev_best_id]["jb_dir"], "jailbreak_opt_final.json")
            if os.path.exists(candidate):
                prev_run_log_path = candidate
        
        iteration_template_ids = []
        for tpl in new_scorer_templates:
            tpl_str = json.dumps(tpl, sort_keys=True)
            tpl_id  = hashlib.md5(tpl_str.encode()).hexdigest()[:8]
            if tpl_id in template_registry:
                logging.info(f"Template {tpl_id} already exists, skip.")
                continue

            jb_dir = os.path.join(log_dir, f"jailbreak_{tpl_id}")
            run_log, _, best_seed_templates, mean_alignments_this_run, newly_solved = await run_jailbreaking_prompt_optimization(
                scorer_prompt=tpl, best_seed_templates_=best_seed_templates,
                data_path=jailbreak_data_path, log_dir=jb_dir, prompt_num_iterations=jailbreaking_iterations,
                K=jailbreaking_K, M=jailbreaking_M, jb_inherit_n=jb_inherit_n,
                _query_performance_registry=query_performance_registry,
                _prev_run_log=prev_run_log_path,
                optim_model=optim_model, target_model=target_model,
                scorer_model=scorer_model, asr_model=asr_model,
                _solved_queries=solved_queries
            )
            solved_queries.update(newly_solved) 
            for query, align in mean_alignments_this_run.items():
                query_performance_registry[query] = max(query_performance_registry.get(query, 0.0), align)

            temp_data_path = os.path.join(jb_dir, "jailbreak_opt_final.json")
            perf, mean_alignments = await evaluate_scoring_template(tpl, temp_data_path)
            for query, align in mean_alignments.items():
                query_performance_registry[query] = max(query_performance_registry.get(query, 0.0), align)

            template_registry[tpl_id] = {
                "template": tpl, "overall_alignment": perf["overall_alignment"], "details": perf,
                "jailbreak_results": run_log, "iteration": it + 1, "jb_dir": jb_dir,
            }
            iteration_template_ids.append(tpl_id)

        if iteration_template_ids:
            perfs = [template_registry[tid]["overall_alignment"] for tid in iteration_template_ids]
            avg_perf, best_perf = mean(perfs), max(perfs)
            best_template_id = max(iteration_template_ids, key=lambda x: template_registry[x]["overall_alignment"])
        else:
            avg_perf = best_perf = 0.0
            best_template_id = None

        run_history.append({
            "iteration": it + 1, "avg_performance": avg_perf, "best_performance": best_perf,
            "best_template_id": best_template_id, "new_template_ids": iteration_template_ids,
            "known_queries_count": len(query_performance_registry),
        })
        current_total_cost_data = {
            "scorer_model": scorer_model.get_total_usage(),
            "target_model": target_model.get_total_usage(),
            "score_optim_model": score_optim_model.get_total_usage(),
            "asr_model": asr_model.get_total_usage(),
            # "optim_model": optim_model.get_total_usage()
        }
        checkpoint_path = save_json_checkpoint(
            {
                "run_history": run_history, "template_registry": template_registry,
                "query_performance_registry": query_performance_registry,
                "solved_queries": list(solved_queries), 
                "total_cost_data": current_total_cost_data
            },
            log_dir, f"scoring_opt_checkpoint_{it+1}.json"
        )
        save_json_checkpoint(best_seed_templates, log_dir, f"best_seed_templates_{it+1}.json")
        logging.info(f"Checkpoint saved → {checkpoint_path}")
        await asyncio.sleep(0.1)

    best_ids = sorted(template_registry, key=lambda tid: template_registry[tid]["overall_alignment"], reverse=True)[:scoring_topK]
    best_templates = [template_registry[tid]["template"] for tid in best_ids]

    total_cost_data = {
        "scorer_model": scorer_model.get_total_usage(),
        "target_model": target_model.get_total_usage(),
        "score_optim_model": score_optim_model.get_total_usage(),
        "asr_model": asr_model.get_total_usage(),
        # "optim_model": optim_model.get_total_usage()
    }

    final_results = {
        "best_template_ids": best_ids, "best_templates": best_templates,
        "template_registry": template_registry, "query_performance_registry": query_performance_registry,
        "total_known_queries": len(query_performance_registry), "total_cost_data": total_cost_data,
        "solved_queries": list(solved_queries), 
        "total_solved_queries": len(solved_queries) 
    }
    save_json_checkpoint(final_results, log_dir, "scoring_opt_final_results.json")
    save_json_checkpoint(best_templates, log_dir, "final_best_seed_templates.json")
    logging.info(f"Finished scoring optimization. Known queries={len(query_performance_registry)}")

    return best_templates, log_dir, best_seed_templates, query_performance_registry, total_cost_data

async def main():
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--optim_model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--scorer_model", type=str, default="gpt-4o-mini-2024-07-18")
    p.add_argument("--target_model", type=str, default="gpt-4o-mini-2024-07-18")
    p.add_argument("--score_optim_model", type=str, default="gpt-4o-mini-2024-07-18")
    p.add_argument("--asr_model", type=str, default="gpt-4o-mini-2024-07-18")
    p.add_argument("--jailbreak_data_path", type=str, default="./data/advbench.json")
    p.add_argument("--scoring_iterations", type=int, default=5)
    p.add_argument("--scoring_topK", type=int, default=5, help="top-K templates for scoring meta-prompt")
    p.add_argument("--scoring_M", type=int, default=1, help="scoring M")
    p.add_argument("--jailbreaking_iterations", type=int, default=5, help="jailbreaking prompt iterations")
    p.add_argument("--jailbreaking_K", type=int, default=5, help="jailbreaking K")
    p.add_argument("--jailbreaking_M", type=int, default=5, help="jailbreaking M")
    p.add_argument("--jb_inherit_n", type=int, default=5, help="jb inherit n")
    p.add_argument("--initial_templates_path", type=str, default=None)
    p.add_argument("--log_dir", type=str, default=None, help="log_dir")
    p.add_argument("--best_seed_templates_path", type=str, default=None, help="best_seed_templates")
    p.add_argument("--http_vllm_url", type=str, default="http://127.0.0.1:8001", help="HTTP vLLM server URL")
    p.add_argument("--http_timeout", type=int, default=120, help="HTTP timeout seconds")
    args = p.parse_args()

    from jb.models import BackendType

    if (args.optim_model == args.target_model) and ("llama" in args.optim_model):
        optim_model = None
        target_model = get_model(
            args.optim_model,
            backend=BackendType.HTTP_VLLM,
            http_vllm_url=args.http_vllm_url,
            http_timeout=None,
        )
    elif "llama" in args.optim_model:
        optim_model = get_model(args.optim_model, backend=BackendType.VLLM_LOCAL)
        target_model = get_model(args.target_model)
    else:
        optim_model = get_model(args.optim_model)
        target_model = get_model(args.target_model)

    scorer_model = get_model(args.scorer_model)
    score_optim_model = get_model(args.score_optim_model)
    asr_model = get_model(args.asr_model)

    def mname(m): return getattr(m, "model_name", str(m))
    print(f"Optim model:       {mname(optim_model)}")
    print(f"Target model:      {mname(target_model)}")
    print(f"Scorer model:      {mname(scorer_model)}")
    print(f"Score Optim model: {mname(score_optim_model)}")
    print(f"ASR model:         {mname(asr_model)}")

    if args.best_seed_templates_path:
        with open(args.best_seed_templates_path, "r") as f:
            best_seed_templates = json.load(f)
    else:
        best_seed_templates: Dict[Any, List[Tuple[str, float]]] = {}
    
    best_templates, log_dir, _, registry, costs = await run_scoring_prompt_optimization(
        initial_templates=initial_scorer_templates,
        jailbreak_data_path=args.jailbreak_data_path,
        best_seed_templates=best_seed_templates,
        scoring_iterations=args.scoring_iterations,
        scoring_topK=args.scoring_topK, scoring_M=args.scoring_M,
        jailbreaking_iterations=args.jailbreaking_iterations, jailbreaking_K=args.jailbreaking_K,
        jailbreaking_M=args.jailbreaking_M, jb_inherit_n=args.jb_inherit_n,
        log_dir=args.log_dir,
        score_optim_model=score_optim_model, 
        optim_model=optim_model,
        target_model=target_model, scorer_model=scorer_model, asr_model=asr_model,
        initial_templates_path=args.initial_templates_path,
    )

    print("\n--- Cost Analysis ---")
    
    model_map = {
        "score_optim_model": score_optim_model, "target_model": target_model,
        "scorer_model": scorer_model, "asr_model": asr_model
    }
    
    total_script_cost = 0.0
    
    for model_key, usage_data in costs.items():
        model_obj = model_map.get(model_key)
        # local models may not have a cost calculation method
        if model_obj and hasattr(model_obj, 'get_total_usage'):
            input_cost = usage_data.get('total_input_tokens', 0) * getattr(model_obj, 'input_cost_per_token', 0)
            output_cost = usage_data.get('total_output_tokens', 0) * getattr(model_obj, 'output_cost_per_token', 0)
            cost = input_cost + output_cost
            total_script_cost += cost
            
            print(f"  - {model_key} ({mname(model_obj)}):")
            print(f"    Total Tokens: Input={usage_data.get('total_input_tokens', 0):,}, Output={usage_data.get('total_output_tokens', 0):,}")
            print(f"    Estimated Total Cost: ${cost:.6f}")

    print("------------------------------------")
    print(f"  Total Estimated Script Cost: ${total_script_cost:.6f}")
    print("------------------------------------\n")


    # final_results_path = os.path.join(log_dir, "scoring_opt_final_results.json")
    # if os.path.exists(final_results_path):
    #     with open(final_results_path, 'r') as f:
    #         final_data = json.load(f)
    #         total_solved = final_data.get("total_solved_queries", 0)
            # print(f"Total permanently solved queries (Query ASR == 1): {total_solved}")


if __name__ == "__main__":
    asyncio.run(main())