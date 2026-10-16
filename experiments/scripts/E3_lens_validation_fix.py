# E3_lens_validation: proposed FIX for Theodolite Module 1 (layer_scalar-aware causal split).
# Drop-in replacement for backend/theodolite_math.compute_causal_vector_split (adds `layer_scalar` and a
# `delta_rescale_J` term). Gemma-4 decoder layer (transformers modeling_gemma4.Gemma4TextDecoderLayer.forward):
#     resid_mid  = resid_pre + post_attention_layernorm(attn(input_layernorm(resid_pre)))      (UNscaled)
#     resid_post = layer_scalar * (resid_mid + post_feedforward_layernorm(mlp(pre_ffn_norm(resid_mid))))
# so the exact additive split of resid_post is
#     resid_post = resid_pre + s*(mid-pre) [attention] + (post - s*mid) [FFN] + (s-1)*pre [layer_scalar rescale].
# The current tool uses delta_ffn = post - mid = s*ffn_out + (s-1)*mid, i.e. it folds the rescale of BOTH the
# incoming residual and the attention write into the "FFN vault" vector, and reports delta_attn unscaled (x 1/s).
import torch
from backend.theodolite_math import CausalSplitResult

@torch.no_grad()
def compute_causal_vector_split_fixed(resid_pre, resid_mid, resid_post, J_l, layer_scalar=1.0):
    dtype = J_l.dtype
    JT = J_l.T
    pre, mid, post = resid_pre.float(), resid_mid.float(), resid_post.float()
    s = torch.as_tensor(layer_scalar, dtype=torch.float32, device=pre.device).reshape(-1)[0]
    delta_attn_raw = (s * (mid - pre)).to(dtype)        # attention write as it lands in resid_post
    delta_ffn_raw = (post - s * mid).to(dtype)          # = s * post_ffn_norm(mlp(...)): true FFN write
    delta_rescale_raw = ((s - 1.0) * pre).to(dtype)     # layer_scalar shrinkage of the incoming residual
    delta_attn_J = delta_attn_raw @ JT
    delta_ffn_J = delta_ffn_raw @ JT
    resid_pre_J = resid_pre.to(dtype=dtype) @ JT
    resid_mid_J = resid_mid.to(dtype=dtype) @ JT
    resid_post_J = resid_post.to(dtype=dtype) @ JT
    attn_norm = torch.linalg.vector_norm(delta_attn_J.float(), dim=-1)
    ffn_norm = torch.linalg.vector_norm(delta_ffn_J.float(), dim=-1)
    out = CausalSplitResult(delta_attn_J=delta_attn_J, delta_ffn_J=delta_ffn_J, resid_pre_J=resid_pre_J,
                            resid_mid_J=resid_mid_J, resid_post_J=resid_post_J, attn_norm=attn_norm,
                            ffn_norm=ffn_norm, ffn_dominance_ratio=ffn_norm / (attn_norm + ffn_norm + 1e-8))
    out.delta_rescale_J = delta_rescale_raw @ JT       # (add `delta_rescale_J: Optional[torch.Tensor] = None` to the dataclass)
    return out

# engine_runtime.run_theodolite_sweep call site (line ~660) becomes:
#     s_l = self._decoder_layers[l].layer_scalar.detach().float()
#     split_l = compute_causal_vector_split(r_pre_t, r_mid_t, r_post_t, J_l, layer_scalar=s_l)
