# E4_causal_00: inspect decoder layer structure (no GPU compute)
import inspect
L = lab.layers[10]
print(type(L))
print([n for n,_ in L.named_children()])
src = inspect.getsource(type(L).forward)
print(src[:4000])
print("scalar", L.layer_scalar.shape, float(L.layer_scalar.float().mean()))
print(type(lab.hf))
import sys
print(sys.executable)
