    import argparse, gc, os, random, socket, torch, numpy as np
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from vllm import LLM, SamplingParams
    import uvicorn
    import math

    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

    def is_valid_float(value):
        """Check if the value is a valid float for JSON compatibility"""
        if value is None:
            return False
        if math.isnan(value) or math.isinf(value):
            return False
        return True

    def safe_float(value, default=-float("inf")):
        """Return a safe float value"""
        if value is None:
            return default
        if math.isnan(value) or math.isinf(value):
            return default
        return value

    def build_app(args): 
        app = FastAPI()

        # Additional parameters for controlling batch size
        vllm_kwargs = {
            "model": args.model,
            "tensor_parallel_size": args.tp,
            "gpu_memory_utilization": args.gpu_mem,
            "max_model_len": args.max_len,
            "dtype": args.dtype,
            "enforce_eager": (args.compile == "none"),
        }
        
        # Additional options for controlling batch size
        if hasattr(args, 'max_num_seqs') and args.max_num_seqs:
            vllm_kwargs["max_num_seqs"] = args.max_num_seqs  # Maximum number of sequences to process simultaneously
        
        if hasattr(args, 'max_num_batched_tokens') and args.max_num_batched_tokens:
            vllm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens  # Maximum number of tokens per batch
        
        if hasattr(args, 'max_paddings') and args.max_paddings:
            vllm_kwargs["max_paddings"] = args.max_paddings  # Maximum padding limit
            
        if hasattr(args, 'block_size') and args.block_size:
            vllm_kwargs["block_size"] = args.block_size  # Block size
        
        # Additional options for stability
        if hasattr(args, 'disable_custom_all_reduce') and args.disable_custom_all_reduce:
            vllm_kwargs["disable_custom_all_reduce"] = True
        
        print(f"vLLM initialization options: {vllm_kwargs}")
        llm = LLM(**vllm_kwargs)

        @app.post("/generateText")
        @app.post("/v1/generateText")
        async def generate(req: Request):
            try:
                body = await req.json()
                prompts = body["prompts"]               # list[str]
                sampling_params = body.get("sampling_params", {})
                
                # Client-side batch size limit (optional)
                if hasattr(args, 'client_batch_limit') and args.client_batch_limit:
                    if len(prompts) > args.client_batch_limit:
                        # Process larger batches into smaller batches
                        all_outs = []
                        batch_size = args.client_batch_limit
                        
                        for i in range(0, len(prompts), batch_size):
                            batch_prompts = prompts[i:i + batch_size]
                            sampl = SamplingParams(**sampling_params)
                            batch_outs_raw = llm.generate(batch_prompts, sampl, use_tqdm=False)
                            
                            batch_outs = process_outputs(batch_outs_raw)
                            all_outs.extend(batch_outs)
                        
                        outs = all_outs
                    else:
                        sampl = SamplingParams(**sampling_params)
                        outs_raw = llm.generate(prompts, sampl, use_tqdm=False)
                        outs = process_outputs(outs_raw)
                else:
                    sampl = SamplingParams(**sampling_params)
                    outs_raw = llm.generate(prompts, sampl, use_tqdm=False)
                    outs = process_outputs(outs_raw)

                # Memory cleanup
                if random.random() < 0.1:
                    gc.collect()
                    torch.cuda.empty_cache()
                return JSONResponse({"result": outs})
                
            except Exception as e:
                print(f"Error in generate: {e}")
                return JSONResponse({"error": str(e)}, status_code=500)
        
        def process_outputs(outs_raw):
            """Separate output processing logic"""
            outs = []
            for o in outs_raw:
                if not o.outputs:
                    continue  # Skip if no outputs
                out = o.outputs[0]
                token_ids = out.token_ids or []
                
                # Safe log probability calculation
                if out.cumulative_logprob is None or not token_ids:
                    nll = -100.0  # Safe value representing very low probability
                else:
                    raw_nll = out.cumulative_logprob / len(token_ids)
                    nll = safe_float(raw_nll, -100.0)
                
                # Debug log
                if not is_valid_float(nll):
                    print(f"Warning: Invalid log_prob value: {nll}, using -100.0")
                    nll = -100.0
                
                outs.append({
                    "text": out.text,
                    "stop_reason": out.stop_reason,
                    "log_prob": nll,
                    "token_ids": token_ids,
                })
            return outs
        
        return app

    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--model")
        parser.add_argument("--port", type=int, default=8001)
        parser.add_argument("--gpu_mem", type=float, default=0.9)
        parser.add_argument("--tp", type=int, default=1)
        parser.add_argument("--max_len", type=int, default=4096)
        parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16"])
        parser.add_argument("--compile", default="none", choices=["none", "max-autotune", "full"])
        
        # Additional parameters for controlling batch size
        parser.add_argument("--max_num_seqs", type=int, default=None, 
                        help="Maximum number of sequences the vLLM engine can process simultaneously")
        parser.add_argument("--max_num_batched_tokens", type=int, default=None,
                        help="Maximum number of tokens per batch")
        parser.add_argument("--max_paddings", type=int, default=None,
                        help="Maximum number of paddings per batch")
        parser.add_argument("--block_size", type=int, default=None,
                        help="KV cache block size")
        parser.add_argument("--client_batch_limit", type=int, default=None,
                        help="Standard size for dividing batches between client and server")
        parser.add_argument("--disable_custom_all_reduce", action="store_true",
                        help="Disable custom all-reduce (stability improvement)")

        args = parser.parse_args()

        app = build_app(args)
        uvicorn.run(app, host="0.0.0.0", port=args.port)  

    if __name__ == "__main__":
        main()