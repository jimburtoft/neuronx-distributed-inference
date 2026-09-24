#!/bin/bash
# Multi-core PyTorch Native sweep inside the Beta 6 container.
#
# --device for all 6 Inferentia2 devices (12 cores). Workers pin themselves with
# NEURON_RT_VISIBLE_CORES, so the container needs visibility of all of them.
# --entrypoint bash overrides the SageMaker wrapper entrypoint, which eats args.
IMG=421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:2.13.0-neuronx-py312-sdk2.32.0-ubuntu24.04-neurondlcbuilder-development-6516808413-0

# Map whatever Neuron devices this instance actually has. Hardcoding 6 breaks on
# inf2.xlarge/8xlarge (1 device) with:
#   error gathering device information while adding custom device "/dev/neuron1"
DEV_ARGS=""
for d in /dev/neuron*; do
  [ -e "$d" ] && DEV_ARGS="$DEV_ARGS --device $d"
done

sudo docker run --rm \
  $DEV_ARGS \
  -v /mnt/models:/mnt/models \
  -v /home/ubuntu:/host \
  -e DINOV3_REPO_DIR=/mnt/models/dinov3 \
  --entrypoint bash "$IMG" -c "cd /host && exec python bench_native_mp_inf2.py $*"
echo MP_RUN_DONE
