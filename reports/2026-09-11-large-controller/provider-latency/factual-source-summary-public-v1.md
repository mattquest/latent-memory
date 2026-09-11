# Provider and hardware performance sources

These are external calibration scenarios, not measurements of this experiment's prompts. Sources were retrieved September 11, 2026, 17:43–17:53 UTC. No hosted inference was requested.

## Cached OpenRouter provider pairs

| Model | Provider | Published output tok/s | Published latency s | Currently declared precision |
| --- | --- | ---: | ---: | --- |
| Qwen3-8B | Alibaba (inferred) | 43 | 0.73 | Unspecified |
| Qwen3-14B | Alibaba | 57 | 0.38 | Unspecified |
| Qwen3-14B | NextBit | 50 | 0.75 | int4 |
| Qwen3-14B | DeepInfra | 43 | 0.83 | fp8 |
| Qwen3.5-122B-A10B | SiliconFlow | 30 | 1.49 | fp8 |
| Qwen3.5-122B-A10B | Alibaba | 99 | 0.96 | Unspecified |
| Qwen3.5-122B-A10B | AtlasCloud | 84 | 1.13 | fp8 |
| Qwen3.5-122B-A10B | Novita | 84 | 0.80 | bf16 |

The [8B pricing cache](https://openrouter.ai/qwen/qwen3-8b/pricing) and [14B uptime cache](https://openrouter.ai/qwen/qwen3-14b-04-28/uptime) report averages of one-week P50 series. Both were marked crawled yesterday; availability displays end September 10. Exact performance-week boundaries and timezone are unspecified. The [122B provider table](https://openrouter.ai/qwen/qwen3.5-122b-a10b/benchmarks) was marked crawled last month; its aggregation window is unavailable. These cached observations retain their own provider pairs. The 8B performance series is unlabeled: attribution to Alibaba is inferred from its current sole endpoint and separate availability/error series, not established for the historical performance window. Precision comes from current endpoint declarations and is not proof of historical precision. The current 122B DeepInfra endpoint declares fp4, with no paired public performance measurement available.

All nine current OpenRouter endpoints returned null performance fields. The [official API schema](https://openrouter.ai/docs/api/api-reference/endpoints/list-all-endpoints-for-a-model) says authentication is required to expose its 30-minute latency and throughput percentiles. Null is not zero. The schema defines latency as TTFT in milliseconds. Cached page prose instead calls latency round-trip time despite a separate E2E series, so the cached label remains ambiguous.

The [provider integration guide](https://openrouter.ai/docs/guides/community/for-providers) includes TTFT in throughput's denominator; the [latency cookbook](https://openrouter.ai/docs/guides/best-practices/latency-and-performance) treats decode speed separately. Any extrapolation must expose this disagreement. A scenario range exploring both conventions is not a confidence interval or guarantee.

## Fresh Vercel observation

At 17:45:39 UTC, the [14B endpoint API](https://ai-gateway.vercel.sh/v1/models/alibaba/qwen-3-14b/endpoints) reported DeepInfra TTFT p50 278 ms and p95 1034.3 ms; throughput and precision were null. The [metrics definition](https://vercel.com/docs/ai-gateway/models-and-providers/metrics) uses a rolling hour and starts timing when the provider receives the request. It excludes BYOK traffic. Do not combine this TTFT with an older OpenRouter TPS value or treat it as laptop-to-provider latency.

## Lambda hardware calibration

[Lambda's published 122B benchmark](https://lambda.ai/inference-models/qwen/qwen3.5-122b-a10b) provides separate mean TTFT and TPOT:

| Engine | Hardware | Mean TTFT ms | Mean TPOT ms |
| --- | --- | ---: | ---: |
| SGLang | 4× B200 | 1156 | 13 |
| SGLang | 8× H100 | 2613 | 18 |
| SGLang | 8× A100 | 4602 | 30 |
| vLLM | 4× B200 | 4904 | 13 |
| vLLM | 8× H100 | 1060 | 16 |
| vLLM | 8× A100 | 7612 | 36 |

The workload uses 8192 input tokens, 1024 output tokens, 512 prompts and concurrency 32. Aggregate output throughput is not single-request decode speed. Published versions use `latest`; precision, reasoning mode and execution date are unspecified. The page's September 1 HTTP modification date is not a benchmark date. These hardware observations support explicit extrapolation scenarios, not measured short-prompt provider performance.

## Evidence handling

Timestamped URL, retrieval time, byte count and SHA-256 receipts are retained locally alongside source bodies. Search-rendered numerical excerpts are identified separately from direct HTTP responses. Full third-party HTML and documentation are internal evidence and should not be redistributed with the report.
