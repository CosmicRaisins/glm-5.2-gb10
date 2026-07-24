ARG BASE_IMAGE=vllm-node-tf5-eldritch-dcp:20260723-adaptive-mtp-k245-greedy-v1
FROM ${BASE_IMAGE}

COPY overlay/vllm/transformers_utils/configs/glm5v.py \
  /usr/local/lib/python3.12/dist-packages/vllm/transformers_utils/configs/glm5v.py
COPY overlay/vllm/model_executor/models/glm5v.py \
  /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/glm5v.py
COPY scripts/apply_vllm_registry_overlay.py /tmp/apply_vllm_registry_overlay.py

RUN /usr/bin/python3 /tmp/apply_vllm_registry_overlay.py \
 && /usr/bin/python3 -m compileall -q \
      /usr/local/lib/python3.12/dist-packages/vllm/transformers_utils/configs/glm5v.py \
      /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/glm5v.py \
 && rm /tmp/apply_vllm_registry_overlay.py

LABEL org.opencontainers.image.title="QuantTrio GLM-5.2 Vision production overlay"
LABEL org.opencontainers.image.description="Adaptive MTP 2/4/5 production image plus MoonViT/PatchMerger GLM-5V overlay"
LABEL local.quanttrio.vision-source="baseten/GLM-5.2-Vision-NVFP4@f6eab6117386a0c69152fdf272dc65bfd0254f9f"
LABEL local.quanttrio.vision-text-backbone="QuantTrio/GLM-5.2-Int4-Int8Mix"
