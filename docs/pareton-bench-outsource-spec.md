# Pareton Bench — LLM 推理引擎基准测试系统 开发规格书 v1

> 状态：草案（待商务条款确认后生效）
> 日期：2026-07-14
> 发包方：Pareton（下称"我方"）
> 承包方：待定（下称"开发方"）

---

## 0. 文档目的

本文档定义一个**独立的 LLM 推理引擎基准测试系统**（代号 **pareton-bench**）的完整开发规格，用于外包开发。内容包括：系统边界、接口契约、三个核心模块（正确性门 / 性能初筛 / SLA 压测）的技术规格、工程要求、交付物、里程碑与验收标准。

本文档是自包含的：开发方**不需要**、也**不会**获得我方上游系统的任何信息。所有输入以本文档定义的 JSON 契约为准。

---

## 1. 项目背景（一段话）

我方运营一个 LLM 推理优化平台：外部贡献者针对主流开源推理引擎（当前以 vLLM 为主）提交优化，我方将每个优化构建成一个独立的 Docker 引擎镜像。**pareton-bench 的职责是回答一个问题：给定一个候选引擎镜像和一个基线引擎镜像，候选镜像是否在不损失输出正确性的前提下，真实地提升了推理性能？**

测试结果直接决定贡献者的收益分配，因此本系统是"裁判"角色，对**可复现性、抗作弊、审计留痕**的要求高于一般的 benchmark 工具。

---

## 2. 系统定位与边界

### 2.1 一句话定位

一个**命令行工具 + Python 库**：输入一个 `bench_request.json`（含基线镜像、候选镜像、模型、负载 trace、SLA 阈值、硬件配置），输出一个 `bench_report.json`（含各阶段指标与 pass/fail 判定）以及完整的证据包（evidence bundle）。

### 2.2 In scope（开发方负责）

| 模块 | 内容 |
|---|---|
| 引擎生命周期管理 | 拉起 / 健康检查 / 销毁 Docker 引擎容器（基线与候选） |
| 模型权重管理 | 从 HuggingFace 下载任意指定模型（支持 revision 固定与 gated model token），本地缓存、哈希校验、只读挂载进容器 |
| 模块 A：正确性门 | Greedy teacher-forced logprob 对比（见 §5） |
| 模块 B：性能初筛 | 低成本 smoke 测试，快速淘汰明显不达标的候选（见 §6） |
| 模块 C：SLA 压测 | 完整 workload trace 回放 + TTFT/ITL/吞吐等指标 + SLA 判定（见 §7） |
| 报告与证据包 | `bench_report.json` + 每请求原始记录 + 环境指纹 |
| Mock 引擎模式 | 无 GPU 环境下可跑通全流程的 CI 测试模式 |

### 2.3 Out of scope（明确不做，我方自行负责）

- 引擎镜像的**构建**（镜像作为输入给出，一律以 digest 引用）
- 候选优化的来源、审核、收益分配逻辑
- 多机/多环境的调度编排（我方会在不同硬件上分别调用本工具）
- 任何数据库、对象存储、区块链相关逻辑
- Web UI / Dashboard

### 2.4 关键设计约束

1. **引擎即黑盒 HTTP 服务。** 本系统只通过 OpenAI 兼容 API（`/v1/completions`）与引擎交互，不依赖引擎内部实现，因此对 vLLM 版本无硬编码依赖（版本由镜像输入决定，不同客户的 profile 会 pin 不同版本）。
2. **模型可任意指定。** 只要 HuggingFace 上可下载（含需要 token 的 gated 模型），系统必须能跑。不允许硬编码任何模型名。
3. **网络隔离。** 引擎容器运行期间必须无外网访问（Docker internal network），只允许 harness 与引擎之间的本地通信。模型权重由 harness 预先下载并挂载，不允许引擎在运行期自行下载任何东西。
4. **一切可复现。** 相同的 `bench_request.json` + 相同硬件 → 相同的判定结论（性能数字允许在声明的方差范围内波动，见 §7.5）。
5. **一切留痕。** 镜像 digest、模型 revision 与权重哈希、GPU/驱动/CUDA 指纹、随机种子、每个请求的原始时间戳与 logprob 全部落盘。

---

## 3. 总体架构与运行流程

```
bench_request.json
      │
      ▼
┌─────────────────────────────────────────────┐
│                pareton-bench                │
│                                             │
│  1. 校验请求 & 环境指纹采集                    │
│  2. HF 模型下载/缓存/校验                     │
│  3. 拉起基线引擎容器 ──┐                      │
│  4. 拉起候选引擎容器 ──┤ (internal network)   │
│                       ▼                     │
│  5. 模块 A 正确性门（fail → 直接出报告）        │
│  6. 模块 B 性能初筛（fail → 直接出报告）        │
│  7. 模块 C SLA 压测                          │
│  8. 汇总 → bench_report.json + evidence/     │
└─────────────────────────────────────────────┘
```

- 三个模块**串行、fail-fast**：正确性不过不跑性能，初筛不过不跑全量压测（省 GPU 时）。
- `mode` 参数可单独指定跑某一个模块（调试用）。
- 基线与候选**不同时压测**：性能测试时同一时刻只有一个引擎占用 GPU，避免互相干扰（正确性门允许两个引擎共存，因为不测时延）。

---

## 4. 接口契约

### 4.1 CLI

```bash
pareton-bench run \
  --request /path/to/bench_request.json \
  --output-dir /path/to/output/

# 退出码约定：
#   0  = harness 正常完成（无论 pass 还是 fail，判定看报告）
#   1  = 请求文件非法 / schema 校验失败
#   2  = 环境错误（GPU 不足、Docker 不可用、模型下载失败等）
#   3  = 引擎异常（容器启动失败、健康检查超时、压测中途崩溃）
```

**重要：判定结论（pass/fail）只体现在报告里，不体现在退出码里。** 退出码只反映 harness 本身是否正常运行。

### 4.2 `bench_request.json`（输入）

```jsonc
{
  "schema_version": 1,
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "mode": "all",                        // all | correctness | perf_screen | sla_bench

  "model": {
    "hf_repo": "Qwen/Qwen2.5-7B-Instruct",
    "hf_revision": "bb46c15ee4bb56c5b63245ef50fd7637234d6f75",  // 必须 pin commit
    "dtype": "bfloat16",
    "quantization": null,               // null | "fp8" | "awq" | ...（透传给 serve args）
    "max_model_len": 8192
  },

  "hardware": {
    "gpu_count": 1,                     // tensor_parallel = gpu_count
    "gpu_sku_expected": "NVIDIA-H200"   // 仅校验用：与实际探测不符时 warning 并记录
  },

  "engines": {
    "baseline": {
      "image": "ghcr.io/example/engine@sha256:aaaa...",   // 一律 digest 引用
      "serve_args": ["--enable-prefix-caching"],           // 追加的引擎启动参数
      "env": {}
    },
    "candidate": {
      "image": "ghcr.io/example/engine@sha256:bbbb...",
      "serve_args": ["--enable-prefix-caching"],
      "env": {}
    }
  },

  "workload_trace": {
    "path": "./trace.json",
    "sha256": "sha256:cccc..."          // harness 必须校验后才使用
  },

  "correctness": {
    "num_prompts": 256,                 // 从 trace 中取前 N 条（确定性选取）
    "max_new_tokens": 128,
    "thresholds": {
      "mean_abs_logprob_diff": 0.005,   // 默认值，我方按 campaign 配置可调
      "max_abs_logprob_diff": 0.05,
      "argmax_mismatch_rate": 0.001
    }
  },

  "perf_screen": {
    "num_requests": 64,
    "concurrency": 8,
    "min_throughput_ratio": 1.0         // 候选吞吐 / 基线吞吐 的最低要求
  },

  "sla_bench": {
    "repetitions": 3,
    "warmup_requests": 32,
    "thresholds": {
      "p99_ttft_ms": 2000,
      "p99_itl_ms": 50
    }
  },

  "hf_token_env": "HF_TOKEN"            // gated 模型的 token 从该环境变量读取
}
```

### 4.3 Workload trace schema（输入，我方提供样例）

```jsonc
{
  "schema_version": 1,
  "meta": {
    "name": "synthetic_v0",
    "description": "..."
  },
  "requests": [
    {
      "id": "r-000001",
      "arrival_offset_ms": 0,           // 相对压测开始的到达时刻（open-loop 回放用）
      "prompt": "...",                  // 或 "prompt_token_ids": [...]
      "max_tokens": 256,
      "sampling": {                     // SLA 压测按此执行；正确性门强制 greedy
        "temperature": 0.7,
        "top_p": 0.9
      }
    }
  ]
}
```

### 4.4 `bench_report.json`（输出）

```jsonc
{
  "schema_version": 1,
  "task_id": "550e8400-...",
  "verdict": "pass",                    // pass | fail_correctness | fail_perf_screen | fail_sla | error
  "started_at": "2026-07-14T18:00:00Z",
  "finished_at": "2026-07-14T19:12:34Z",

  "environment": {
    "gpu": [{"index": 0, "name": "NVIDIA H200", "vbios": "...", "memory_mb": 143771}],
    "driver_version": "560.xx",
    "cuda_version": "12.x",
    "docker_version": "...",
    "harness_version": "0.3.1",         // pareton-bench 自身版本（git describe）
    "hostname_hash": "sha256:..."       // 不落明文主机名
  },

  "inputs_fingerprint": {
    "baseline_image_digest": "sha256:aaaa...",
    "candidate_image_digest": "sha256:bbbb...",
    "model_repo": "Qwen/Qwen2.5-7B-Instruct",
    "model_revision": "bb46c15e...",
    "model_weights_sha256": "sha256:dddd...",   // 权重文件清单哈希
    "trace_sha256": "sha256:cccc...",
    "request_sha256": "sha256:eeee..."          // bench_request.json 本身的哈希
  },

  "correctness": {
    "verdict": "pass",
    "num_prompts": 256,
    "num_positions_compared": 31872,
    "mean_abs_logprob_diff": 0.0011,
    "max_abs_logprob_diff": 0.021,
    "argmax_mismatch_rate": 0.0,
    "evidence": "evidence/correctness/"
  },

  "perf_screen": {
    "verdict": "pass",
    "baseline_output_tokens_per_s": 1834.2,
    "candidate_output_tokens_per_s": 2101.7,
    "throughput_ratio": 1.146,
    "evidence": "evidence/perf_screen/"
  },

  "sla_bench": {
    "verdict": "pass",
    "repetitions": 3,
    "candidate": {
      "ttft_ms":   {"p50": 213.0, "p95": 890.1, "p99": 1544.8},
      "itl_ms":    {"p50": 21.2, "p95": 35.7, "p99": 44.1},
      "e2e_ms":    {"p50": 5120.0, "p95": 10233.5, "p99": 12890.2},
      "output_tokens_per_s": 2088.4,
      "requests_per_s": 8.1,
      "sla_goodput_ratio": 0.994       // SLA 内完成的请求占比
    },
    "baseline": { /* 同结构 */ },
    "speedup": {
      "output_tokens_per_s_ratio": 1.139,
      "p99_ttft_ratio": 0.87           // 候选/基线，<1 为改善
    },
    "cross_rep_variance": {
      "p99_ttft_ms_rel_range": 0.06    // 3 次重复间的相对极差，见 §7.5
    },
    "evidence": "evidence/sla_bench/"
  }
}
```

### 4.5 证据包（evidence bundle）目录结构

```
output/
├── bench_report.json
├── harness.log                        # 全量结构化日志（JSON lines）
└── evidence/
    ├── env/                           # nvidia-smi -q、docker inspect 全文
    ├── correctness/
    │   ├── forced_sequences.jsonl     # 每条 prompt 的 greedy 输出 token 序列
    │   └── logprob_diffs.jsonl        # 每 token 位置的双引擎 logprob 与差值
    ├── perf_screen/
    │   └── requests.jsonl             # 每请求时间戳明细
    └── sla_bench/
        ├── rep_1/requests.jsonl       # 每请求：到达/首token/完成时间戳、token数
        ├── rep_2/requests.jsonl
        └── rep_3/requests.jsonl
```

---

## 5. 模块 A：正确性门（Correctness Gate）

**方法：greedy teacher-forced logprob 对比。** 核心思想：候选引擎的优化不允许改变模型的数学行为，用基线引擎的 greedy 输出作为"标准答案序列"，强制两个引擎在同一序列上打分并逐位对比。

### 5.1 流程

1. **生成阶段**：对选取的 N 条 prompt（`correctness.num_prompts`，从 trace 头部确定性选取），用**基线引擎**做 greedy 解码（`temperature=0`，固定 `seed`，`max_new_tokens` 按配置），得到每条 prompt 的输出 token 序列。
2. **Teacher-forcing 打分阶段**：把 `prompt + 基线输出序列` 拼成完整序列，分别发给**基线**和**候选**引擎，要求返回该序列上每个位置的 logprob（OpenAI 兼容 API 的 `echo=true` + `logprobs` / `prompt_logprobs` 能力）。两个引擎打分的是**完全相同的 token 序列**，不存在采样分歧。
3. **对比阶段**：只对比**输出部分**（不含 prompt 部分）每个位置的 logprob：
   - `mean_abs_logprob_diff`：所有位置 |Δlogprob| 的均值
   - `max_abs_logprob_diff`：所有位置 |Δlogprob| 的最大值
   - `argmax_mismatch_rate`：两引擎在该位置 top-1 token 不一致的比例（需要 `logprobs=k` 返回 top-k）
4. 三项指标全部低于阈值 → pass，否则 fail。

### 5.2 实现要求与注意事项

- **串行请求**：正确性门阶段请求逐条串行发送（并发=1），排除动态 batching 引起的数值抖动。
- **数值抖动是真实存在的**（CUDA 非确定性、batch 组合差异），阈值就是为此设计的容差。默认阈值（见 §4.2）为初始值，M1 验收时双方基于"同一镜像 vs 自身"的空跑数据共同校准。
- **对照组自检**：harness 必须内置 `baseline vs baseline` 自检模式（同一镜像跑两遍走完整流程），作为阈值校准和 CI 用例。
- 若引擎不支持所需的 logprob 返回能力，harness 应在健康检查阶段探测并以退出码 3 报错，错误信息明确指出缺失的 API 能力。

### 5.3 反作弊要求

- 打分请求的下发顺序在同一 `task_id` 下确定但**不可被引擎预测**（以 `task_id` 为种子做确定性 shuffle）。
- 记录每一位置的原始 logprob 到证据包，我方可独立复核。

---

## 6. 模块 B：性能初筛（Perf Screen）

**目的：** 用几分钟的低成本测试淘汰"正确但不快"的候选，避免浪费完整 SLA 压测的 GPU 时。

### 6.1 流程

1. 取 trace 前 `num_requests` 条请求（确定性选取）。
2. **闭环（closed-loop）压测**：固定并发数 `concurrency`，请求完成即发下一条，忽略 trace 中的到达时间。
3. 基线、候选各跑一遍（先基线后候选，各自独占 GPU，之间完整销毁容器并等待显存释放）。
4. 计算 `throughput_ratio = 候选输出吞吐 / 基线输出吞吐`，低于 `min_throughput_ratio` → fail。

### 6.2 说明

- 该阶段的采样参数按 trace 执行（非 greedy），但 `seed` 固定。
- 吞吐统计只计 output tokens，从第一条请求发出到最后一条完成的墙钟时间为分母，含 warmup 排除逻辑（前 10% 请求不计入统计窗口，但计入负载）。

---

## 7. 模块 C：SLA 压测（SLA Benchmark）

**目的：** 在贴近真实负载形态下测量候选引擎的时延/吞吐指标，并与 SLA 阈值和基线对比。

### 7.1 回放语义

- **开环（open-loop）回放**：严格按 trace 中每条请求的 `arrival_offset_ms` 发出请求，**不管前序请求是否完成**。这是与真实流量最接近的模式，能暴露排队与 KV cache 压力。
- 请求超时上限：单请求超过 `max(10 × p99_e2e 预估, 120s)` 未完成则标记 timeout 并计入失败（timeout 请求出现即整轮 fail，写明原因）。

### 7.2 指标定义（必须严格按此实现）

| 指标 | 定义 |
|---|---|
| TTFT | 请求发出（HTTP 写完）到收到第一个 output token 的时间 |
| ITL | 同一请求内相邻两个 output token 到达间隔（全部间隔进入分布统计） |
| E2E latency | 请求发出到最后一个 token 收到 |
| output_tokens_per_s | 统计窗口内所有请求 output token 总数 / 窗口墙钟时长 |
| requests_per_s | 完成请求数 / 窗口墙钟时长 |
| sla_goodput_ratio | TTFT 与 ITL 均满足 SLA 阈值的请求占比 |

- 所有时延指标报 p50 / p95 / p99。
- 流式（streaming）接收是硬性要求，否则 TTFT/ITL 无法测量。

### 7.3 测试流程

1. Warmup：先发 `warmup_requests` 条请求（不计入统计），确保权重加载、CUDA graph 捕获、prefix cache 等完成。
2. 正式回放整条 trace，重复 `repetitions` 次（默认 3），每次重复之间不重启引擎。
3. 基线与候选各自独立执行上述全流程（独占 GPU）。
4. 聚合：每个指标取**各次重复的中位数**作为报告值。

### 7.4 判定

- 候选的 `p99_ttft_ms`、`p99_itl_ms` 中位数值必须满足 `sla_bench.thresholds` → 否则 `fail_sla`。
- 同时报告候选相对基线的 speedup（不参与 pass/fail，供我方后续使用）。

### 7.5 可复现性要求（验收硬指标）

- 同一硬件、同一请求文件连续跑 3 次完整 SLA 压测，**p99 TTFT 的相对极差（max-min)/median ≤ 10%**，吞吐相对极差 ≤ 5%。达不到时开发方需定位并消除噪声源（CPU 绑核、容器资源限制、日志 IO 等属于开发方职责范围内的调优）。
- 报告中必须包含 `cross_rep_variance` 字段暴露实测方差。

---

## 8. 工程要求

### 8.1 技术栈与代码

- Python ≥ 3.11，类型注解完整，`ruff` + `mypy` 干净。
- 依赖全部 pin 精确版本（`requirements.lock` 或 `uv.lock`）。
- **不允许任何二进制 blob 进仓库**；不允许引入闭源依赖。
- 结构化日志（JSON lines），日志里不允许出现 HF token 等敏感值。
- 不允许任何遥测/上报/外呼；harness 的外网访问仅限：HuggingFace 下载、镜像 registry 拉取。

### 8.2 仓库与协作流程

- 仓库：我方新建 `Pareton-ai/pareton-bench`（私有），初始只含本规格、trace 样例与 CI 骨架。
- 开发方通过 **Pull Request** 提交，我方 review 后合并；不开放直接 push 权限。
- 每个 PR 必须过 CI（lint + mypy + 单元测试 + mock 引擎 e2e）。
- Commit 历史清晰，禁止 force push。

### 8.3 Mock 引擎模式（CI 关键）

- 提供 `--mock-engine` 模式：用一个进程内的假 OpenAI 兼容 server 替代真实引擎（可配置固定 logprob、固定 token 延迟），使全流程（含三个模块与报告生成）能在**无 GPU 的 CI 机器**上跑通。
- 必须附带对抗性测试用例：
  - 一个"logprob 被篡改"的 mock 引擎 → 正确性门必须 fail
  - 一个"吞吐低于基线"的 mock 引擎 → 初筛必须 fail
  - 一个"TTFT 超标"的 mock 引擎 → SLA 必须 fail
  - `baseline vs baseline` → 三个模块必须全 pass

### 8.4 GPU 与开发环境（成本分工）

- **开发方自备开发用 GPU**（单卡消费级即可，如 4090/A10；用小模型如 Qwen2.5-0.5B/7B 开发调试）。
- **我方负责验收环境**：里程碑验收时由我方在我方租用的目标硬件（多卡 H200 级别、70B+ 模型）上运行，开发方远程配合调试。开发方**不会**获得我方云账号或生产凭据。

---

## 9. 交付物清单

1. `pareton-bench` Python 包 + CLI（源码，含完整类型注解）
2. Harness 自身的 Dockerfile（可容器化运行 harness 本体）
3. 全部 JSON schema 的机器可读定义（JSON Schema 文件）与校验代码
4. 测试套件：单元测试 + mock 引擎 e2e + §8.3 对抗性用例
5. 文档：README（安装/运行/排障）、每个模块的设计说明、指标定义文档（与 §7.2 一致）
6. M3 验收时的一份完整样例报告（真实 GPU 上 baseline vs baseline + baseline vs 候选样例镜像）

---

## 10. 里程碑与验收标准

> 工期为建议值，最终以商务约定为准。每个里程碑的验收由我方在我方硬件上独立执行——**验收标准全部是可机器判定的**。

### M1（约 2 周）：骨架 + 正确性门

**范围：** CLI 骨架、请求 schema 校验、HF 模型下载/缓存/哈希、Docker 引擎生命周期管理（internal network）、环境指纹采集、模块 A 完整实现、mock 引擎模式 + CI。

**验收：**
- [ ] 无 GPU CI 机器上 `--mock-engine` 全流程绿灯，含 §8.3 全部对抗用例
- [ ] 我方单卡机器上：任意指定一个 HF 小模型（我方现场任选，如 Qwen2.5-7B-Instruct），`baseline vs baseline` 正确性门 pass；"篡改 logprob"样例镜像 fail
- [ ] 证据包完整：logprob 明细可独立复核
- [ ] 双方基于空跑数据确认正确性阈值默认值

### M2（约 2 周）：性能初筛 + SLA 压测

**范围：** 模块 B、模块 C 完整实现（开环回放、流式指标采集、重复与聚合、方差报告）、完整 `bench_report.json`。

**验收：**
- [ ] 我方单卡机器上跑通 `mode=all` 全流程，报告与 schema 一致
- [ ] "慢引擎"样例镜像在初筛被拒；"TTFT 超标"样例在 SLA 被拒
- [ ] §7.5 复现性达标：3 次连跑 p99 TTFT 相对极差 ≤ 10%、吞吐 ≤ 5%
- [ ] 指标定义抽查：我方用独立脚本从 `requests.jsonl` 重算 p99 TTFT，与报告一致（±1ms）

### M3（约 2 周）：多卡 + 硬化 + 交接

**范围：** tensor parallel 多卡支持（`gpu_count > 1`）、任意 HF 模型的配置矩阵测试、异常路径硬化（容器崩溃/显存不足/下载中断的干净错误与退出码）、文档完善、交接。

**验收：**
- [ ] 我方多卡 H200 环境上：70B 级模型（FP8、TP=8）全流程跑通，产出 §9-6 的样例报告
- [ ] 异常注入测试：压测中途 `docker kill` 引擎 → 退出码 3 + 报告含明确错误；显存不足 → 退出码 2
- [ ] 文档验收：我方工程师仅凭文档在干净机器上完成安装与运行
- [ ] 代码走查：无未 pin 依赖、无外呼、无敏感信息落日志

### 付款结构（占位，商务另议）

| 里程碑 | 比例 | 条件 |
|---|---|---|
| 合同签订 | ___% | 【待商务确认】 |
| M1 验收 | ___% | 全部验收项通过 |
| M2 验收 | ___% | 全部验收项通过 |
| M3 验收 | ___% | 全部验收项通过 + 交接完成 |

---

## 11. 双方提供物

### 我方提供

- 本规格书 + JSON schema 文件
- Workload trace 样例（synthetic）与 trace 生成说明
- 基线引擎镜像引用（digest）+ 上述对抗性样例镜像（M1 前提供"篡改 logprob"镜像，M2 前提供"慢引擎"与"TTFT 超标"镜像）
- `Pareton-ai/pareton-bench` 仓库与 CI 环境
- 里程碑验收硬件与验收执行
- 一名对接工程师（接口问题 24h 内答复）

### 开发方提供

- 开发人力与开发用 GPU
- 每周一次书面进度同步（PR 列表 + 阻塞项）
- 里程碑验收时的远程配合

---

## 12. 保密与知识产权（占位）

- 【待商务确认】NDA：本规格书及仓库内容在协议签署前均属保密信息。
- 【待商务确认】知识产权归属：交付代码的著作权归属（建议：work-for-hire，全部归我方）。
- 【待商务确认】开发方不得将本项目代码或衍生物用于其他用途或开源。

---

## 13. 术语表

| 术语 | 含义 |
|---|---|
| 基线 / baseline | 未经优化的参考引擎镜像 |
| 候选 / candidate | 待评测的优化后引擎镜像 |
| Teacher forcing | 强制模型在给定 token 序列上打分而非自由生成 |
| TTFT | Time To First Token，首 token 时延 |
| ITL | Inter-Token Latency，token 间隔时延（也称 TPOT） |
| Open-loop | 按预定时刻发请求，不等前序完成 |
| Closed-loop | 固定并发，完成一条再发一条 |
| Evidence bundle | 支撑报告结论的全部原始数据 |
