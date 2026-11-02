# Measurements

Measured on ATOM RBLN-CA22 (`rebel-compiler` 0.10.2, `vllm-rbln` 0.10.2.post2) with captured GPT-OSS-20B layer-0 inputs. "Built-in" is the SDK's sink-supporting attention op; "Modified" is this repo. Case counts cover both learned and disabled sinks across four fresh processes.

## vs. FP64 reference

Pass criteria: finite, max abs < 0.20 (prefill) or < 0.05 (decode), RMS < 0.01.

| Path | Built-in pass | Modified pass | Max abs | Max RMS |
|---|---:|---:|---:|---:|
| SWA prefill | 8/24 | 8/24 | 0.374 | 0.0096 |
| SWA decode | 52/72 | 52/72 | 0.118 | 0.0076 |
| Full prefill | 8/80 | 8/80 | 0.374 | 0.0119 |
| Full decode | 48/96 | 48/96 | 0.374 | 0.0231 |

## Modified vs. built-in

Within tolerance: finite, max abs ≤ 0.03125, RMS ≤ 0.01.

| Path | Bitwise equal | Within tolerance | Max abs |
|---|---:|---:|---:|
| SWA prefill | 24/24 | 24/24 | 0 |
| SWA decode | 32/72 | 72/72 | 0.03125 |
| Full prefill | 80/80 | 80/80 | 0 |
| Full decode | 96/96 | 96/96 | 0 |

## Latency

Median of four per-process medians across steady sequential runtime calls (prefix ≥ 128). Measures host-side call latency (including packing, casts, and transfers; excluding input preparation and first calls), rather than device-only time or generation speed.

| Path | Sink | Built-in (ms) | Modified (ms) | Ratio |
|---|---|---:|---:|---:|
| SWA prefill | learned | 0.592 | 0.608 | 1.027 |
| SWA prefill | disabled | 0.655 | 0.657 | 1.003 |
| SWA decode | learned | 0.117 | 0.127 | 1.085 |
| SWA decode | disabled | 0.119 | 0.128 | 1.081 |
| Full prefill | learned | 0.727 | 0.743 | 1.023 |
| Full prefill | disabled | 0.744 | 0.765 | 1.027 |
| Full decode | learned | 0.150 | 0.150 | 1.000 |
| Full decode | disabled | 0.147 | 0.147 | 1.002 |

## GPT-OSS slice

2 layers without MoE (weight sharing off, 4 prompts): bitwise-equal first-token logits and the same 16 generated token IDs vs. the built-in op in 4/4 prompts; 49/64 logit vectors bitwise equal (max abs 0.125).
