#!/usr/bin/env python3
"""A tiny random Gemma 4 for the atlas's checks, and for driving the atlas page without the GPU.

The instrument's own checks (scripts/check_causal_check.py and the others) read the real tokenizer from the local
Hugging Face cache. The atlas's checks must also run where that cache cannot be filled (a machine with no access to the
gated checkpoint), so this stand-in brings its own: a byte-level BPE trained here on a few lines, with the special pieces
a Gemma 4 chat needs (<bos>, <eos>, <|turn>, <turn|>, <|think|>, <|channel>, <channel|>) and a chat template of the same
shape, which splits a multi-byte character over several pieces as the real one can. The model is transformers'
`Gemma4ForCausalLM`, built as check_causal_check.py builds its stand-in (sliding layers with a small window, a last
layer with full attention, `attention_k_eq_v`, partial rotary on the full layer, the final softcap of 30, every norm and
layer scalar given some spread), at the stand-in vocabulary's size. `stand_in_runtime` wraps it in a TheodoliteRuntime
whose own methods are unchanged, with a random bf16 "lens" at each layer (the identity at the last) and the shipped
lens's provenance, so the lens card and the label gate behave as on the real server.

    .venv/bin/python -I scripts/atlas_stand_in.py --serve 8765     # the instrument's server on the stand-in

serves the whole instrument (frontend/dist when built) on the stand-in: what the pages show is random, and the server
says so (`/api/atlas/arch` -> "stand_in": true). Nothing here downloads anything.
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"

import torch

N_LAYERS = 6
WINDOW = 8
HIDDEN = 64
FFN = 128
VOCAB_TARGET = 600

SPECIALS = ["<pad>", "<eos>", "<bos>", "<unk>", "<|turn>", "<turn|>", "<|think|>", "<|channel>", "<channel|>"]

# The chat template's shape (a stand-in for Gemma 4's, not a copy): a start mark, turns between <|turn>{role} and
# <turn|>, the model's turn opened for the answer, thinking switched on by <|think|> at the head of a system turn and
# switched off by an empty thought channel.
CHAT_TEMPLATE = (
    "{{ bos_token }}{% for m in messages %}<|turn>{{ 'model' if m['role'] == 'assistant' else m['role'] }}\n"
    "{% if loop.first and m['role'] == 'system' and enable_thinking %}<|think|>{% endif %}{{ m['content'] }}<turn|>\n"
    "{% endfor %}{% if add_generation_prompt %}<|turn>model\n{% if not enable_thinking %}<|channel>thought\n<channel|>"
    "{% endif %}{% endif %}"
)

CORPUS = [
    "Who are you? I am a large language model, trained to help.",
    "Write a snake game in Python. Here is a simple version using pygame.",
    "def merge(a, b):\n    out = []\n    while a and b:\n        out.append(a.pop(0))\n    return out + a + b",
    "Describe the color orange. Orange is warm, like a sunset over the sea.",
    "Alice called Bob because she wanted to borrow his car. He said yes.",
    "The committee rejected the proposal; members said it was expensive and rushed.",
    "我昨天去了北京，但是天气很冷，所以我们在酒店里待了一整天。",
    "Ingredients: 2 cups flour, 1 tsp salt, 3 eggs. Mix the flour and salt, then add the eggs.",
    "The mitochondria produce ATP through oxidative phosphorylation, which requires oxygen.",
    "Write whatever you like. The quick brown fox jumps over the lazy dog. 🙂 café naïve",
]


def stand_in_tokenizer() -> Any:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tk = Tokenizer(models.BPE(unk_token=None))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_TARGET,
        special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tk.train_from_iterator(CORPUS * 20, trainer)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tk,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
        chat_template=CHAT_TEMPLATE,
    )
    fast.add_special_tokens({"additional_special_tokens": SPECIALS[4:]})
    return fast


def stand_in_hf(tokenizer: Any, dtype: torch.dtype = torch.float32, attn: str = "sdpa", seed: int = 0) -> Any:
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

    torch.manual_seed(seed)
    V = len(tokenizer)
    cfg = Gemma4TextConfig(
        vocab_size=V,
        vocab_size_per_layer_input=V,
        hidden_size_per_layer_input=0,
        hidden_size=HIDDEN,
        intermediate_size=FFN,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_global_key_value_heads=1,
        head_dim=16,
        global_head_dim=32,
        attention_k_eq_v=True,
        sliding_window=WINDOW,
        layer_types=["sliding_attention"] * (N_LAYERS - 1) + ["full_attention"],
        final_logit_softcapping=30.0,
        max_position_embeddings=4096,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        rope_parameters={
            "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
        },
    )
    cfg._attn_implementation = attn
    hf = Gemma4ForCausalLM(cfg).to(dtype).eval()
    with torch.no_grad():
        for name, p in hf.named_parameters():
            if p.dim() == 1:
                p.copy_((torch.rand_like(p, dtype=torch.float32) * 0.8 + 0.6).to(p.dtype))
        for layer in hf.model.layers:
            layer.layer_scalar.copy_((torch.rand(1) * 0.8 + 0.6).to(layer.layer_scalar.dtype))
    hf.generation_config.eos_token_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<turn|>")]
    return hf


def stand_in_runtime(hf: Any, tokenizer: Any) -> Any:
    """A TheodoliteRuntime holding the stand-in, its own methods unchanged (as check_causal_check.py builds one)."""
    from interp_engine import EagerModel

    from backend.causal_check import language_model
    from backend.engine_runtime import JPP_FITTED_MODEL, TheodoliteRuntime

    rt = TheodoliteRuntime(device="cpu", load_jlens=False)
    rt.eager_model = EagerModel("stand-in", device="cpu", dtype=hf.dtype, hf_model=hf, tokenizer=tokenizer)
    rt.hf_model = hf
    rt.tokenizer = tokenizer
    lm = language_model(hf)
    rt._decoder_layers = lm.layers
    rt._final_norm_weight = lm.norm.weight.detach()
    rt._lm_head_weight = hf.lm_head.weight.detach()
    g = torch.Generator().manual_seed(5)
    D = lm.norm.weight.shape[0]
    for l, block in enumerate(lm.layers):
        rt._pre_ffn_norm_weights[l] = block.pre_feedforward_layernorm.weight.detach()
        rt._post_ffn_norm_weights[l] = block.post_feedforward_layernorm.weight.detach()
        rt.jpp_lenses[l] = (
            torch.eye(D, dtype=torch.bfloat16)
            if l == len(lm.layers) - 1
            else (torch.eye(D) + 0.3 * torch.randn(D, D, generator=g)).to(torch.bfloat16)
        )
    rt.jpp_provenance = {"source": "stand-in", "method": "jpp", "lrp_mode": "rlens", "hf_model_name": JPP_FITTED_MODEL,
                         "n_prompts": 64, "max_seq_len": 128}
    setattr(rt, "atlas_stand_in", True)
    return rt


def stand_in(dtype: torch.dtype = torch.float32, attn: str = "sdpa") -> Any:
    """(runtime, vault) on a fresh stand-in."""
    from backend.larql_vault import LarqlVaultEngine

    tok = stand_in_tokenizer()
    rt = stand_in_runtime(stand_in_hf(tok, dtype, attn), tok)
    return rt, LarqlVaultEngine(rt)


def bind_server(rt: Any, vault: Any) -> Any:
    """The instrument's FastAPI app with the stand-in bound in place of the model-loading lifespan."""
    import backend.server as server
    from backend.mcp_scientist import bind_runtime

    server._RUNTIME, server._VAULT = rt, vault  # type: ignore[assignment]
    bind_runtime(rt, vault)

    @asynccontextmanager
    async def no_lifespan(_app: Any):  # the real one loads the model
        yield

    server.app.router.lifespan_context = no_lifespan
    return server.app


def main(argv: Optional[list] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--serve", type=int, metavar="PORT", help="serve the instrument on the stand-in, on 127.0.0.1:PORT")
    a = ap.parse_args(argv)
    if not a.serve:
        ap.print_help()
        return 0
    import uvicorn

    rt, vault = stand_in()
    app = bind_server(rt, vault)
    uvicorn.run(app, host="127.0.0.1", port=int(a.serve), log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
