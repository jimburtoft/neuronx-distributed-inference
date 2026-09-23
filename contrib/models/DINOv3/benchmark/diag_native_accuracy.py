"""Diagnose the PyTorch Native accuracy failure on inf2.

Symptom: compiled ViT-L BF16 gives cos_sim ~-0.05 vs CPU FP32 (should be
~0.999+). Timing is stable and plausible, so the graph runs -- it just produces
wrong values. Narrow down WHERE.

Ladder, cheapest first:
  1. Is a trivial op correct on device?            (matmul)
  2. Is EAGER (uncompiled) model output correct?   -> isolates compile vs device
  3. Is FP32 correct?                              -> isolates BF16 cast
  4. Does a random-weight model differ run-to-run? -> isolates nondeterminism
  5. Are the two CPU references self-consistent?   -> isolates MY harness
"""

import os
import sys

REPO = os.environ.get("DINOV3_REPO_DIR", "/mnt/models/dinov3")
sys.path.insert(0, REPO)

import torch
import torch.nn.functional as F

DEV = "privateuseone:0"


def sync():
    torch.neuron.synchronize()


def cos(a, b):
    return F.cosine_similarity(a.flatten().unsqueeze(0).float(),
                               b.flatten().unsqueeze(0).float()).item()


def load(key="dinov3_vitl16", seed=0):
    # Deterministic init so CPU and device models have IDENTICAL weights.
    torch.manual_seed(seed)
    import dinov3.hub.backbones as B
    m = getattr(B, key)(pretrained=False)
    m.eval()
    return m


print("=" * 66)
print("1. trivial op on device")
print("=" * 66)
t = torch.ones(64, 64)
td = t.to(DEV)
r = (td @ td)
sync()
got = float(r.sum().cpu())
print(f"   matmul sum: got {got}, expect {64.0 * 64 * 64}  "
      f"{'OK' if abs(got - 64.0**3) < 1 else 'WRONG'}")

print("=" * 66)
print("2. weights identical between CPU and device model?")
print("=" * 66)
m_cpu = load()
m_dev = load()                      # same seed -> same weights
sd_c, sd_d = m_cpu.state_dict(), m_dev.state_dict()
maxw = max((sd_c[k].float() - sd_d[k].float()).abs().max().item()
           for k in sd_c if sd_c[k].is_floating_point())
print(f"   max weight diff CPU vs to-be-device model: {maxw:.3e} "
      f"{'OK' if maxw == 0 else 'MISMATCH -- harness bug'}")

x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    ref_fp32 = m_cpu(x).float()
print(f"   CPU FP32 ref: shape {tuple(ref_fp32.shape)} "
      f"norm {ref_fp32.norm().item():.4f}")

print("=" * 66)
print("3. EAGER on device (no torch.compile), FP32")
print("=" * 66)
me = load().to(torch.float32).to(DEV)
with torch.no_grad():
    o = me(x.to(torch.float32).to(DEV))
    sync()
    o = o.cpu()
print(f"   eager fp32 cos={cos(ref_fp32, o):.6f}  norm {o.float().norm().item():.4f}")
del me

print("=" * 66)
print("4. EAGER on device, BF16")
print("=" * 66)
me = load().to(torch.bfloat16).to(DEV)
with torch.no_grad():
    o = me(x.to(torch.bfloat16).to(DEV))
    sync()
    o = o.cpu()
print(f"   eager bf16 cos={cos(ref_fp32, o):.6f}  norm {o.float().norm().item():.4f}")
del me

print("=" * 66)
print("5. COMPILED on device, FP32")
print("=" * 66)
mc = load().to(torch.float32).to(DEV)
cm = torch.compile(mc, backend="neuron", fullgraph=False, dynamic=False)
with torch.no_grad():
    o = cm(x.to(torch.float32).to(DEV))
    sync()
    o = o.cpu()
print(f"   compiled fp32 cos={cos(ref_fp32, o):.6f}  norm {o.float().norm().item():.4f}")
del mc, cm

print("=" * 66)
print("6. COMPILED on device, BF16 (the failing config)")
print("=" * 66)
mc = load().to(torch.bfloat16).to(DEV)
cm = torch.compile(mc, backend="neuron", fullgraph=False, dynamic=False)
with torch.no_grad():
    o1 = cm(x.to(torch.bfloat16).to(DEV)); sync(); o1 = o1.cpu()
    o2 = cm(x.to(torch.bfloat16).to(DEV)); sync(); o2 = o2.cpu()
print(f"   compiled bf16 cos={cos(ref_fp32, o1):.6f}  norm {o1.float().norm().item():.4f}")
print(f"   run-to-run self-consistency cos={cos(o1, o2):.6f} "
      f"({'deterministic' if cos(o1, o2) > 0.9999 else 'NONDETERMINISTIC'})")

print("=" * 66)
print("7. CPU BF16 reference (is BF16 itself the problem, independent of Neuron?)")
print("=" * 66)
mb = load().to(torch.bfloat16)
with torch.no_grad():
    ob = mb(x.to(torch.bfloat16)).float()
print(f"   CPU bf16 vs CPU fp32 cos={cos(ref_fp32, ob):.6f}")
