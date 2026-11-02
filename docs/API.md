# API

```python
import torch
from rbln_sinks import Contract, SinksModule, pack_query

# (kind, phase, query_len, qkv_dtype, cache_dtype, window_or_partition_size)
c = Contract("swa", "prefill", 128, torch.bfloat16, torch.bfloat16, 128)

# Run on the host for every real input, outside the compiled graph:
c.validate_runtime(q=q, k=k, v=v, cache=cache, seq=seq, bt=bt, sinks=sinks, end=end)

model = SinksModule(c)  # forward(q, k, v, cache, seq, bt, sinks, end=None)
```

`SinksModule.forward` checks only shape, dtype, and contiguity (value checks cannot run inside a compiled graph). Call `validate_runtime` on the host to validate tensor values; skipping it leaves bad `seq`, `end`, or block-table values uncaught.

`pack_query(q, sinks)` performs checked host-side packing. `attention_reference` and `partition_reference` compute the FP64 reference over the full logical history without modeling the device cache or its rounding.

## Layout and math

All host tensors must be contiguous CPU tensors.

- `Q`: `[1, 8, 8, T, 64]` (batch, KV heads, query heads per KV head, tokens, head dim); query head `h` uses KV head `h // 8`.
- `K`, `V`: `[1, 8, 1, T, 64]`, same dtype as `Q`.
- `sinks`: flat FP32 `[64]`, one per query head, finite or `-inf` (`-inf` disables the sink; `0` does not).
- Packed `Q`: FP32 `[1, 8, 8, T, 128]` (`Q` in `[..., :64]`, sink in `[..., 64]`, 63 × `-inf` in `[..., 65:]`).
- Output: grouped head-major; to convert `[B, H, Q, D]` to `[B, Q, H, D]`, use `reshape`, not `view`.
- Scores: `QK^T × 0.125`. The sink is an unscaled logit that contributes only to the softmax denominator (no value row).
- Masking: `query_pos - 128 < key_pos <= query_pos` (SWA). Full causal adds the sink at logical partition 0 only, even with physical block order `[1, 0]`.

## Cache and metadata

| Path | cache | seq | block table | end |
|---|---|---|---|---|
| SWA 128 | `[2, 1, 8, 1, 128, 64]` | int16 `[1]`, `min(cached, 128)` | int32, prefill `[1]` / decode `[1, 1]`, value 0 | int16 `[1]`, `seq + T` |
| Full | `[2, 2, 8, 1, P, 64]` | int16 `[1, 2]`, cached count per partition | P=128: int32, P=1024: int16; prefill `[2]` / decode `[1, 2]`, value `[1, 0]` | none |

For full attention, `seq = [min(s, P), max(0, min(s - P, P))]` (capacity `2P`). Prefill must start at a multiple of `T`.

The kernels update the static KV cache in place while declaring `mutates_args=()` (matching upstream). Bind the cache with `CompileContext.mark_static_address(cache)` and verify it appears as a runtime input in compiled metadata. Compilation example caches do not carry over to runtime: start each runtime with a zeroed cache and populate it sequentially. Concurrent calls and shared or aliased caches were not tested.

## Registration

`register_kernels()` loads both kernels under the existing `rbln_triton_ops` names. It refuses if `RBLN_USE_CUSTOM_KERNEL` is not `"1"`, `TRITON_CACHE_DIR` is set, the SDK or `torch` version differs from the tested stack, or the op names are already registered. Run it once in a fresh process; if registration fails partway, start a new process.
