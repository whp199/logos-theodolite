# E2_behavior: read-only memory check (STORE GPU tensor sizes by key prefix). Does not modify anything.
def sz(x):
    if isinstance(x, torch.Tensor): return (x.numel() * x.element_size(), x.is_cuda)
    if isinstance(x, dict):
        a = [sz(v) for v in x.values()]; return (sum(t[0] for t in a), any(t[1] for t in a))
    if isinstance(x, (list, tuple)):
        a = [sz(v) for v in x]; return (sum(t[0] for t in a), any(t[1] for t in a))
    return (0, False)
print("alloc GB", torch.cuda.memory_allocated() / 2**30, "reserved", torch.cuda.memory_reserved() / 2**30)
for k, v in STORE.items():
    b, c = sz(v)
    if b > 50e6: print(k, round(b / 2**30, 3), "GB", "cuda" if c else "cpu")
result = None
