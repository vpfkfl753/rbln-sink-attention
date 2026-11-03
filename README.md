# rbln-sink-attention

Triton attention kernels for Rebellions ATOM (RBLN-CA22) with GPT-OSS attention sink support.

In `vllm-rbln` 0.10.2, the Triton attention path drops the sink term ([vllm-rbln#1248](https://github.com/RBLN-SW/vllm-rbln/issues/1248)). This repo modifies the two kernels to keep it:

- Sliding window (window 128): packs Q to head dim 128 (64 Q values, the sink logit, 63 `-inf`) while K, V, KV cache, and output stay at head dim 64.
- Full causal: adds the sink to the softmax denominator once at logical partition 0; later partitions merge unchanged.

The Python package handles host-side query packing and input validation, and includes an FP64 CPU reference.

## Status

Research code, tested only on:

- Device: ATOM RBLN-CA22, 1 card
- SDK: `rebel-compiler` 0.10.2, `vllm-rbln` 0.10.2.post2
- PyTorch: `torch` 2.9.1+cpu

Untested on SDK 0.12 and the current `vllm-rbln` `dev` branch. A batch-2 decode variant was proposed upstream as [vllm-rbln#1314](https://github.com/RBLN-SW/vllm-rbln/pull/1314) and closed because ATOM non-optimum paths are not on the current milestone and the Triton kernels are planned for deprecation.

## Results

On captured GPT-OSS-20B layer-0 inputs, compared with the SDK's sink-supporting built-in attention op:

- Accuracy vs. built-in: 272/272 within tolerance (max abs ≤ 0.03125), 232/272 bitwise equal.
- Accuracy vs. strict FP64 reference: 116/272 pass, the same count as the built-in op.
- Runtime call latency vs. built-in: full-causal decode about the same, prefill +0.3 to +2.7%, sliding-window decode +8%.
- 2-layer GPT-OSS slice without MoE (weight sharing off): first-token logits match built-in bitwise in 4/4 prompts, with the same 16 generated tokens in 4/4. With MoE layers in the same graph, export fails for both original and modified Triton kernels.

Details: [docs/MEASUREMENTS.md](docs/MEASUREMENTS.md), [docs/SUPPORT.md](docs/SUPPORT.md).

## CPU tests

Runs without the SDK or NPU (15 tests covering query packing, input rejection, stub-op export, and reference math):

```sh
pip install "torch>=2.9,<2.11"
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Device usage

Use a fresh process (`import rbln_sinks` has no registration side effects):

```python
import os
os.environ["RBLN_USE_CUSTOM_KERNEL"] = "1"  # before anything imports rebel

from rbln_sinks.device import register_kernels
register_kernels()  # registers rbln_triton_ops; refuses on version mismatch or name clash
```

Shapes, KV cache layout, and host checks: [docs/API.md](docs/API.md).

## Files

- `src/rbln_sinks/vendor/`: the two modified kernels and wrappers
- `src/rbln_sinks/`: host API, input checks, FP64 reference
- `docs/patches/`: diffs against `vllm-rbln` 0.10.2.post2
- `fixtures/`: captured GPT-OSS-20B layer-0 inputs and two recorded device outputs used by tests

## Acknowledgments

This research was supported by the Ministry of Science and ICT (MSIT) through the 2026 High-Performance Computing (HPC) Support Program (RQT-26-041563).

## License

Apache-2.0. Original kernels © Rebellions Inc.; see [NOTICE](NOTICE).
