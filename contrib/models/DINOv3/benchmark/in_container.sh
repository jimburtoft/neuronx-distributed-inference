#!/bin/bash
# Runs INSIDE the Beta 6 container. Kept as its own file because nesting quotes
# through `ssh -> bash -c -> docker run --entrypoint bash -c` is unreliable.
set -e

# No pip installs needed: the benchmark imports dinov3.hub.backbones directly
# rather than going through hubconf.py, which is what dragged in torchvision
# (unusable here -- Beta 6's Python lacks _lzma) and transformers.
python - <<'PYIN'
import torch, torch_neuronx
print("container | torch", torch.__version__, "| torch_neuronx", torch_neuronx.__version__)
PYIN

cd /host
exec python bench_native_inf2.py "$@"
