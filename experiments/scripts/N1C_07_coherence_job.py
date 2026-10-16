# N1C_07_coherence_job -- (daemon job, ~1 min) top-10 J-VINDEX / naive tokens for NULL directions, to test whether the
# "neatness" (semantic tightness) of J-VINDEX labels is a property of single neurons or of the J++ lens itself.
# Directions per layer (32 each): single = 32 other random neurons (W_down col * w_post, + sign); pair = sum of 2 random
# neurons; mix512 = Gaussian mixture of 512 random neurons (a non-basis direction in the FFN write space);
# iso = isotropic Gaussian. Ranking = N1B zfinal (RMSNorm*fnw @ W_U^T; identical top-k to the tool's unembed).
# PRE-STATED PREDICTION: at L20-30, J-VINDEX labels of mix512 directions are about as semantically tight as those of
# single neurons (tightness is a lens property), while at L45-57 single neurons are tighter than mix512.
import json
R = "./experiments/results/"
neu = json.load(open(R + "N1_neurons.json"))
out = {}
with torch.no_grad():
    WU = lab.WU
    fnw = lab.fnw.float()

    def top10(v):
        v = v.float()
        v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-12) * fnw
        z = (v.to(WU.dtype) @ WU.T).float()
        return z.topk(10, dim=-1).indices.cpu().tolist()

    g = torch.Generator(device=lab.dev).manual_seed(4242)
    gc = torch.Generator().manual_seed(4242)
    for l in neu["layers"]:
        Jf = lab.J[l].float()
        wpost = rt._post_ffn_norm_weights[l].to(lab.dev).float()
        sample = set(int(i) for i in neu["neurons"][str(l)])
        pool = [i for i in torch.randperm(21504, generator=gc).tolist() if i not in sample]
        Wm = rt.slice_w_down(l, torch.tensor(pool[:512], device=lab.dev)).float() * wpost
        vmix = torch.randn(32, 512, device=lab.dev, generator=g) @ Wm
        vsing = rt.slice_w_down(l, torch.tensor(pool[512:544], device=lab.dev)).float() * wpost
        vpair = (rt.slice_w_down(l, torch.tensor(pool[544:608], device=lab.dev)).float() * wpost).view(32, 2, -1).sum(1)
        viso = torch.randn(32, Wm.shape[1], device=lab.dev, generator=g)
        rec = {"single_ids": pool[512:544]}
        for nm, v in (("mix512", vmix), ("single", vsing), ("pair", vpair), ("iso", viso)):
            rec[nm + "_jpp"] = top10(v @ Jf.T)
            rec[nm + "_naive"] = top10(v)
        out[str(l)] = rec
        del Jf, Wm
        print(l, flush=True)
result = out
