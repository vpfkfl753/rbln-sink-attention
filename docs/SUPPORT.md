# Tested configurations and known failures

`Contract` accepts only the shapes below; passing validation does not mean every input passes the accuracy checks (see MEASUREMENTS.md).

Common across all cases: batch 1, 64 query heads, 8 KV heads, head dim 64, finite Q/K/V, flat FP32 sinks (both learned and `-inf` disabled), and fixed block tables (`[0]` for SWA, `[1, 0]` for full).

| Path | QKV / cache dtype | Prefill T | Decode T | Window / partitions |
|---|---|---|---|---|
| SWA | FP32 / FP32 | 64, 128 | 1 | 128 |
| SWA | BF16 / FP32 | 64 | 1 | 128 |
| SWA | FP16 / FP32 | 64 | 1 | 128 |
| SWA | BF16 / BF16 | 64, 128 | 1 | 128 |
| Full causal | FP32 / FP32 | 64 | 1 | 2 × 128 |
| Full causal | BF16 / BF16 | 128 | 1 | 2 × 1024 |

Rejected: batch ≠ 1, other head counts or head dims, H32 with T32, unlisted dtype pairs, FP16 cache, full P128 with BF16, skipped or extra partitions, other block tables, non-contiguous or CUDA tensors, NaN or `+inf` sinks, non-finite Q/K/V, negative or over-capacity `seq`, and SWA `end != seq + T`.

## Known failures

| Case | Result |
|---|---|
| Sink as a separate kernel operand | Registration or export fails; packing the sink into Q works around it (compiler cause unknown). |
| FP32 disabled-sink decode at position 63/64 | Max abs 0.058 (> 0.05 limit); built-in op also fails. |
| BF16 learned-sink decode at position 64/127 | Max abs 0.052 / 0.061 (> 0.05 limit). |
| FP32 T128 learned-sink prefill from 128/256 | Max abs 0.33 / 0.40, RMS ~0.009; same on built-in op. |
| FP32 H32 with T32 | Built-in op: `DEVICE_GRAPH_CONVERSION`; wrapper: `UNEXPECTED_GRAPH`. Rejected by API. |
| Sink repeated in every partition | Did not compile; wrong math verified on CPU only. |
| GPT-OSS 2 layers with MoE | Export fails for both original and modified Triton kernels (even with weight sharing off); a dense control with the same interface compiles. |

Not covered: full 24-layer GPT-OSS, TP4, other batch or head shapes, sparse masks, gradients, concurrent requests, end-to-end quality, and system power.

The FP64 reference rounds Q/K/V to the input dtype before computing in FP64. Device internal precision is unknown, and sharing a failure with the built-in op does not prove it is harmless rounding.
