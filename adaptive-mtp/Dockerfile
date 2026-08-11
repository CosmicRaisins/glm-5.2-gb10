# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 CosmicRaisins and contributors
ARG BASE_IMAGE=vllm-node-tf5-eldritch-dcp:20260705-e232d26-mtpfix-pr72
FROM ${BASE_IMAGE}

COPY adaptive-mtp/LICENSE adaptive-mtp/NOTICE /usr/share/licenses/glm-5.2-gb10/

COPY patches/adaptive-mtp-vllm-hooks.patch /tmp/adaptive-mtp-vllm-hooks.patch
COPY adaptive-mtp/overlay/vllm/v1/spec_decode/dynamic/acceptance_length.py \
  /usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/dynamic/acceptance_length.py

RUN cd /usr/local/lib/python3.12/dist-packages \
 && git apply --unidiff-zero -p1 /tmp/adaptive-mtp-vllm-hooks.patch \
 && /usr/bin/python3 -m compileall -q \
      /usr/local/lib/python3.12/dist-packages/vllm/config/speculative.py \
      /usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py \
      /usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/dynamic/acceptance_length.py \
      /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/cudagraph_utils.py \
 && rm /tmp/adaptive-mtp-vllm-hooks.patch

LABEL org.opencontainers.image.title="QuantTrio adaptive MTP overlay"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.source="https://github.com/CosmicRaisins/glm-5.2-gb10"
LABEL org.opencontainers.image.description="Aiden Le adaptive speculative depth forward-port with production 2/4/5 controller"
LABEL local.quanttrio.adaptive-spec-depths="2,4,5"
LABEL local.quanttrio.adaptive-spec-window="32@k2,16@k4/k5"
LABEL local.quanttrio.adaptive-spec-policy="head>=0.85 probes k4; tail23>=0.70 probes k5; tail23>=0.35 holds k4; p4>=0.15 holds k5"
