# V6_quiet_dims_00_inspect: module structure of one decoder layer, param names/shapes, memory headroom.
import inspect
L = lab.layers[30]
print(type(L).__name__)
for n, m in L.named_modules():
    if n.count('.') <= 1:
        attrs = [a for a in ("weight", "weight_packed", "weight_scale") if hasattr(m, a)]
        print(" ", n, type(m).__name__, {a: tuple(getattr(m, a).shape) for a in attrs})
src = inspect.getsource(type(L).forward)
print(src[:4000])
print("layer_scalar", L.layer_scalar.shape if hasattr(L, "layer_scalar") else None)
print("free MiB", torch.cuda.mem_get_info()[0] / 2**20)
fn = rt.final_norm_weight.float()
print("fnw stats", float(fn.min()), float(fn.max()), float(fn.abs().median()))
result = {"ok": True}
