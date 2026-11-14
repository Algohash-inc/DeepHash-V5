# Serving DeepHash V5 with vLLM

This guide sets up DeepHash V5 as an OpenAI-compatible API server on your own GPU.

## Requirements

| | |
|---|---|
| GPU | NVIDIA Blackwell (B200, RTX 5090 or newer). The MoE experts are NVFP4, which needs Blackwell tensor cores. |
| VRAM | 32 GB or more (tested on RTX 5090 32 GB and B200 180 GB). Weights use ~18 GB; the rest is KV cache. |
| Disk | ~20 GB |
| OS | Linux, CUDA 12.8+ driver |
| Python | 3.12 |
| vLLM | Nightly build, `0.30.1rc1.dev58` or later (tested on 0.30.1 nightly and 0.34.0 nightly). Stable releases do not support this model yet. |

## 1. Install vLLM

```bash
uv venv ~/deephash-venv --python 3.12
source ~/deephash-venv/bin/activate
uv pip install -U --pre vllm --torch-backend=auto \
  --extra-index-url https://wheels.vllm.ai/nightly --index-strategy unsafe-best-match
uv pip install "huggingface_hub[hf_transfer]"
```

## 2. Download the weights

Access is gated. Open the [model page](https://huggingface.co/Algohash-inc/DeepHash-V5), accept the DeepHash V5 terms of use, then log in with the same Hugging Face account.

```bash
hf auth login
HF_HUB_ENABLE_HF_TRANSFER=1 hf download Algohash-inc/DeepHash-V5 \
  --exclude "adapter/*" --local-dir ~/models/DeepHash-v5
```

`adapter/` holds the LoRA adapter used for further fine-tuning. It is not needed for serving.

## 3. Start the server

The repository ships ready-made vLLM settings in [`vllm/`](vllm). Pick the profile for your GPU:

| Profile | GPU | Context | Concurrent requests |
|---|---|---|---|
| `rtx5090` | RTX 5090 / any 32 GB Blackwell card | 256K | 2 |
| `b200` | B200 (numbers on the model card) | 16K | 64 |

```bash
cd ~/models/DeepHash-v5
bash vllm/serve.sh rtx5090        # or: bash vllm/serve.sh b200
```

`serve.sh` sets the two required environment variables and runs `vllm serve` with the profile file. To run it by hand:

```bash
export VLLM_USE_V2_MODEL_RUNNER=1                       # the diffusion decoder runs on the v2 model runner
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # avoids OOM from fragmentation
vllm serve ~/models/DeepHash-v5 --config ~/models/DeepHash-v5/vllm/rtx5090.yaml
```

Extra arguments override the profile, for example `bash vllm/serve.sh rtx5090 --port 9000`.
Do not change the parser settings in the profile files: tool calls and thinking mode depend on them.

### What the profile settings do

| Flag | Reason |
|---|---|
| `--moe-backend marlin` | The default NVFP4 MoE kernel cannot pad this model's expert size (2112). Marlin handles it. |
| `--attention-backend TRITON_ATTN` | Block diffusion uses bidirectional attention inside each block. Only the Triton backend supports it. |
| `reasoning-parser` | Splits thinking-mode output into `reasoning_content`. |
| `enable-auto-tool-choice`, `tool-call-parser` | Returns OpenAI-style `tool_calls`. |
| `--gpu-memory-utilization 0.80` (32 GB) | vLLM counts CUDA graphs in the budget. Lower values leave too little KV cache for 256K. |
| `--max-num-batched-tokens 8192` (32 GB) | Caps the warm-up memory peak. |

The KV cache is stored in FP8 automatically (set by the checkpoint). No flag is needed.

## 4. Test it

```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "deephash-v5",
  "messages": [{"role": "user", "content": "Hi! Please introduce yourself."}]
}'
```

Or with the examples in [`examples/`](examples):

```bash
export ALGOHASH_BASE_URL=http://localhost:8000/v1
export ALGOHASH_API_KEY=none
python examples/quickstart.py
```

## Request parameters

DeepHash V5 is a diffusion model, so some sampling parameters do not apply.

- **Do not send** `temperature`, `top_p`, `min_p`, `seed`, `min_tokens` or `logit_bias`. The server returns HTTP 400.
- Output is deterministic for the same prompt.
- `max_tokens`, `tools`, `tool_choice`, `stream` and `stop` work as usual.
- Thinking mode: `extra_body={"chat_template_kwargs": {"enable_thinking": True}}`.

## Speed vs quality

The diffusion sampler commits more tokens per step when the entropy bound is higher. This is faster, at some quality cost. The default is 0.1. To change it, add:

```bash
  --override-generation-config '{"sampler_config": {"_cls_name": "EntropyBoundSamplerConfig", "entropy_bound": 0.2}}'
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `FLASHINFER_CUTLASS` padding error at startup | Add `--moe-backend marlin`. |
| Startup error about the model runner or attention backend | Set `VLLM_USE_V2_MODEL_RUNNER=1` and use `--attention-backend TRITON_ATTN`. |
| CUDA OOM during warm-up | Lower `--max-num-batched-tokens` or `--max-model-len`; set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. |
| HTTP 400 on requests | Remove `temperature`, `seed` and other sampling parameters. |
| `--enable-lora` makes startup fail | LoRA serving is not supported for this architecture. Use the merged weights. |
| Tensor parallel (TP > 1) errors | Use one GPU (`--tensor-parallel-size 1`). The NVFP4 weights fit on a single 32 GB card. |
