# V7_tool_bugs_08_diag: are unsteered raw-prompt generations degenerate because (i) the IT model is run without a chat
# template (tool default) or (ii) steering state leaks between capture_generation calls? Also counts module hooks.
# PRE-REGISTERED: (i) -- HF generate (no eager steering) reproduces the degenerate raw continuations; chat-templated
# versions of the same requests are fluent; hook counts identical before/after a steered capture_generation and the
# unsteered generation is identical before/after (no leak).
exec(open("./experiments/scripts/V7_tool_bugs_common.py").read())
from interp_engine import Address
from interp_engine.steer_specs import AddSpec, LayerSteeringSpec, SteeringSpec
def hook_count():
    n = 0
    for m in rt.hf_model.modules():
        n += len(m._forward_hooks) + len(m._forward_pre_hooks)
    return n
out = {"hooks_start": hook_count()}
raws = RAW_PROMPTS
gens = {}
for p in raws:
    ids = lab.ids(p)
    hf = lab.generate(ids, max_new_tokens=12)[0]
    comp, _ = await rt.eager_model.capture_generation(ids, [Address("resid_post", 59)], max_tokens=12, temperature=0.0)
    gens[p] = {"hf_greedy": hf, "eager_greedy": comp.text}
# steered call then re-check
g = torch.Generator().manual_seed(1)
spec = SteeringSpec(layers={l: LayerSteeringSpec(operations=[AddSpec(vector=torch.randn(5376, generator=g), scale=650.0, normalize=True)])
                            for l in [29, 35, 41, 47, 53]}, point="resid_post")
ids = lab.ids(raws[0])
comp, _ = await rt.eager_model.capture_generation(ids, [Address("resid_post", 59)], max_tokens=6, temperature=0.0, steering_spec=spec)
out["steered_text"] = comp.text
out["hooks_after_steer"] = hook_count()
for p in raws:
    ids = lab.ids(p)
    comp, _ = await rt.eager_model.capture_generation(ids, [Address("resid_post", 59)], max_tokens=12, temperature=0.0)
    gens[p]["eager_greedy_after_steer"] = comp.text
    gens[p]["identical_after_steer"] = comp.text == gens[p]["eager_greedy"]
# same requests through the chat template (assistant turn continues the raw text)
for p in raws[:4]:
    c = lab.chat("Continue this text.", assistant=p)
    gens[p]["chat_template_hf_greedy"] = lab.generate(c, max_new_tokens=12)[0]
out["gens"] = gens
out["sampling_settings"] = repr(rt.eager_model.sampling_settings(temperature=0.0, top_k=None, top_p=None, presence_penalty=None))
print(json.dumps(out, indent=1))
result = out
