# QuantTrio GLM-5.2 Vision overlay

This production path keeps the unmodified
[QuantTrio/GLM-5.2-Int4-Int8Mix](https://huggingface.co/QuantTrio/GLM-5.2-Int4-Int8Mix)
text backbone and attaches only the vision components from
[Baseten's GLM-5.2-Vision-NVFP4](https://huggingface.co/baseten/GLM-5.2-Vision-NVFP4)
at revision `f6eab6117386a0c69152fdf272dc65bfd0254f9f`:

- MoonViT-3d vision tower: `833,769,904` bytes
- trained 49.5M-parameter PatchMerger projector: `99,117,136` bytes
- total added weight files: about 0.87 GiB per checkpoint copy

Baseten built the adapter from the frozen MoonViT tower in Moonshot AI's
Kimi-K2.6 and a trained `1152 -> 4608 -> 6144` projector. The projector is the
bridge that makes the vision features compatible with GLM-5.2; this repository
did not train or modify those weights.

## Composite checkpoint

`scripts/assemble_quanttrio_glm5v.py` creates a zero-copy composite directory:

- text shards and support files are relative symlinks to QuantTrio;
- `vision_tower.safetensors` and `mm_projector.safetensors` are relative
  symlinks to the pinned Baseten snapshot;
- the merged safetensors index contains the QuantTrio text tensors plus exactly
  335 vision/projector tensors;
- the wrapper config embeds the original QuantTrio `glm_moe_dsa` text config;
- the Baseten chat template is retained, with the local default reasoning effort
  changed to `medium-high` to match the text production profile;
- `GLM5V_COMPOSITE.json` records sources, revision, sizes, and SHA-256 hashes.

```bash
python3 vision/scripts/assemble_quanttrio_glm5v.py \
  --text-dir /path/to/QuantTrio-GLM-5.2-Int4-Int8Mix \
  --vision-dir /path/to/baseten-GLM-5.2-Vision-NVFP4 \
  --output-dir /cache/huggingface/hub/glm52-quanttrio-vision
```

## vLLM overlay and MTP compatibility

The model wrapper specializes vLLM's Kimi-K2.5 multimodal implementation for a
`GlmMoeDsaForCausalLM` language model. The production fixes are:

- register `Glm5vConfig` and `Glm5vForConditionalGeneration` locally;
- preserve QuantTrio tensor names and quantization mappings while nesting the
  text model under `language_model`;
- expose the nested `lm_head` on the multimodal wrapper so the MTP loader shares
  the target output projection. Without this property, target text is coherent
  but speculative acceptance is exactly zero;
- propagate `index_topk_pattern` to/from the nested text config so the sparse
  indexer uses the trained GLM layer pattern;
- force the GLM MTP proposer down the text-token path. The draft head is
  text-only; treating its permissive signature as multimodal sends embeddings
  down the wrong proposer path and also collapses acceptance;
- reuse MoonViT/PatchMerger processing while accepting GLM's
  `<|begin_of_image|><|image|><|end_of_image|>` placeholder.

Build the vision image after building the adaptive-MTP parent:

```bash
docker build -f vision/Dockerfile \
  --build-arg BASE_IMAGE=glm52-adaptive-mtp:k245 \
  -t glm52-quanttrio-vision:v6 vision
```

## Validated production profile

`recipes/glm52-quanttrio-vision-dcp2-290816.yaml` is the tested profile:

- TP4 + DCP2, one request, 4,096-token batches
- 290,816-token configured context (`71 x 4096`)
- explicit 8.2 GiB/rank `fp8_ds_mla` KV allocation
- 299,648 tokens of measured GPU KV capacity, leaving 8,832 tokens of headroom
- adaptive MTP depths 2/4/5, maximum k=5
- one image and no video per request

The vision-v6 llama-benchy run (pp2048/tg512, five runs, depth 0/8K/32K,
concurrency 1) measured flat prefill around 629/643/640 tok/s and decode means
21.9/19.4/21.9 tok/s. A code canary produced 512 tokens at 32.7 tok/s while
holding high speculative depth; prose dropped to k2. A standard bus-image test
correctly identified and described the image. Raw llama-benchy output is in
`../benchmarks/vision-v6-20260724/`.

The 290,816 context limit is a production headroom choice, not the raw KV
capacity. The vision weights themselves add under 1 GiB, but multimodal runtime
state, model-wrapper overhead, CUDA graphs, and unified-memory launch headroom
also compete with KV on GB10.
