# PAVA

用于长视频问答的持久化局部检索实现：global reasoning 生成检索目标，local retrieval process 在自己的执行上下文中持续获取、审查信息，结束后向上层传递观察及来源。

本仓库提供独立部署的主方法、80 题试跑与 597 题冻结题单、视频编号清单、可选下载工具、预处理、运行和开销汇总。**当前先准备 80 题对应的视频并试跑；597 题只做好准备，是否运行由用户后续决定。** 接手的 AI 先读 [AGENTS.md](AGENTS.md)。

两套范围共用同一份代码与配置：80 题是 597 题内每个数据集的前 20 题，共 75 个独立视频；完整范围是 597 题、465 个独立视频。扩大范围时可复用同一工作目录的媒体材料，但新运行不复用旧模型答案。

## 当前版本

- `core/` 来自已运行的 observe-only PAVA 主方法，保留原提示词、局部 Audit、持续执行和观察投影。
- 新入口把模型地址、模型名和初始帧数参数化，并增加独立的数据准备和用量报告。
- 原主方法运行配置为 Flash / 250 初始帧；示例配置为 **27B / 128 帧的待测配置**。离线验证不等于已在新模型、新机器上复现实验成绩。
- 本仓库暂只提供主方法。Need-Locked、AVP-style、Codex/OpenCode 和信息边界消融没有作为可运行实验移植进来。

本交接版已在原生 Ubuntu 通过六类离线检查，以及一个合成短视频的实际 FFmpeg 抽帧、CPU OpenCLIP 建索引、字幕绑定、缓存复用和运行 dry-run。没有发出新模型请求。依赖验证复用了已有环境，尚未在空白 H200 机器上全新安装验收。

## 1. 安装与模型服务

使用 Ubuntu、Python 3.12、FFmpeg。客户端环境与 vLLM 服务环境分开；不要把 vLLM 的依赖覆盖到本客户端环境。

```bash
git clone https://github.com/project-nju-learner/PAVA.git && cd PAVA
python3.12 -m venv .venv && source .venv/bin/activate
sudo apt-get update && sudo apt-get install -y ffmpeg tmux
pip install torch==2.13.0 torchvision==0.28.0
pip install -r requirements.txt
python -B check.py
cp config.example.json config.local.json
```

Torch/CUDA 的安装方式由部署机器决定；上面给出验证环境的版本，不指定代理、镜像或加速渠道。使用本地 ASR 时再安装 `faster-whisper`，其模型目录须为可被 CTranslate2 读取的本地模型。

准备本地 OpenCLIP `ViT-B-32 / laion2b_s34b_b79k` 权重文件，以及实际使用的多模态 Qwen 权重。索引和文本检索必须使用同一份 OpenCLIP 权重。模型权重不包含在仓库中。

已有 OpenAI Chat Completions 兼容服务时，直接配置地址即可。自建服务可在**单独的服务环境**使用 vLLM；以下是 4 卡、串行测速的启动示例，GPU 由管理员分配后再运行：

```bash
vllm serve /models/Qwen3.8-27B --served-model-name qwen3.8-27b --host 127.0.0.1 --port 8000 --tensor-parallel-size 4 --dtype bfloat16 --max-model-len 262144 --max-num-seqs 1 --limit-mm-per-prompt '{"image":250}'
```

服务端模型名必须与 `config.local.json` 的 `model` 完全一致；权重架构须被服务版本支持。原部署使用 vLLM 0.27.1 / Transformers 5.15.0，本交接版未替新机器验证服务端加载。4 卡示例不意味着需要占用全部 32 张 H200。先保存实际 GPU 型号、服务启动命令、软件版本、上下文上限和图像处理配置，再做同配置比较。

`config.local.json` 的关键项：

| 字段 | 含义 |
|---|---|
| `model`, `base_url` | 实际服务模型名、以 `/v1` 结尾的地址 |
| `input_format` | 本地 vLLM 用 `openai_images`；接受原 video 帧列表的接口用 `dashscope_video` |
| `initial_frames` | 128 或 250；改变后使用新的预处理工作目录 |
| `stream` | 支持 SSE 时启用，记录首 token 时延和近似解码速度 |
| `qa_concurrency` | 正整数，由部署者按机器和服务容量自行配置；1 用于串行测速，单题内部 HTTP 串行 |
| `max_rounds`, `max_repairs` | 原执行配置 8 / 7 |
| `max_requests_per_question`, `max_images_per_question` | 每题保护上限 80 / 8000；不是实际开销 |
| `default_max_output_tokens` | 未显式指定输出上限的阶段补设 8192；原显式上限保留 |

`OPENAI_BASE_URL`、`PAVA_MODEL` 环境变量会覆盖配置文件。私有服务需要认证时，通过 `OPENAI_API_KEY` 环境变量加载自己的密钥；不要写入 JSON、命令示例或 Git。无认证的本地服务不需要密钥。

## 2. 视频与题单

**先备齐 80 题子集对应的 75 个视频，不要求先下载完整数据集。下载渠道、工具和加速方式由部署者自行选择；已有匹配视频可直接复用，不必运行本仓库的下载器。**

视频编号见 [data/video_ids80.txt](data/video_ids80.txt)：每行一个视频，三列依次为数据集、**官方视频 ID**、程序读取的文件名，按数据集去重。题目与视频的完整对应关系在 [data/roster80.json](data/roster80.json)。这些是视频编号，不是压缩分包编号或题目编号；不要把项目内部的 `video_id` 前缀当作官方下载 ID。部分 LongVideoBench 视频的官方 ID 与文件名并非简单加 `.mp4` 的关系，请以清单第三列为准。

| 数据集及当前评测范围 | 试跑题数 | 所需视频数 | 参考来源（可自行选择其他匹配来源） |
|---|---:|---:|---|
| LVBench | 20 | 16 | [公开数据](https://huggingface.co/datasets/lmms-eval/LVBench) |
| LongVideoBench validation | 20 | 20 | [官方数据](https://huggingface.co/datasets/longvideobench/LongVideoBench) |
| Video-MME Long | 20 | 19 | [公开数据](https://huggingface.co/datasets/lmms-eval/Video-MME) |
| MLVU Dev 七类选择题 | 20 | 20 | [选择题数据源](https://huggingface.co/datasets/sy1998/MLVU_dev)、[官方开发集](https://huggingface.co/datasets/MLVU/MVLU) |
| 合计 | 80 | 75 | |

MLVU 只需当前开发集选择题视频，不要求开放式生成任务或独立 MLVU Test。参考镜像里的 parquet split 虽名为 `test`，内容对应 MLVU Dev，不能据此换成独立测试集。使用需要登录的参考来源时，由下载者用自己的账号取得访问权；这不限定必须使用该渠道。

冻结父题单在 `data/roster597.json`，采样 seed 为 42，分集题数为 LVBench 180 / LongVideoBench 150 / Video-MME Long 140 / MLVU 127，共 465 个视频。80 题是其中每集前 20 题，没有重新随机采样。题单只含题干、官方选项和 ID，不含答案标签；按四数据集交错运行，各数据集内部保持原顺序。**下载更多视频不代表扩大当前实验题数。**

### 用自己的数据接入

将视频放到或通过软链接映射到以下结构。`/data/longvideo` 只是示例，可通过 `--video-root` 指定自己的根目录；无需复制、搬动或重新下载已有数据：

```text
<video-root>/LVBench/videos/<video_filename>
<video-root>/LongVideoBench/videos/<video_filename>
<video-root>/Video-MME/videos/<video_filename>
<video-root>/MLVU/videos/<video_filename>
```

文件名使用 `video_ids80.txt` 第三列。若原始目录按任务或时长分层，可对所需文件逐个建立软链接；文件已平铺时也可链接整个 `videos` 目录。程序只读取这些本地路径，**不依赖下载器、下载账号或 `video_manifest.json`**。保持数据集视频的完整内容和时间轴，不能用同名截断片段替代。记录实际来源；不同版本或转码视频应重新准备帧和索引，不直接套用旧缓存。评分标签放在独立的 evaluation 目录，不能混入推理材料。

### 可选：使用仓库自带的子集下载器

只有需要这一下载方式时才运行，已有视频或自有渠道可跳过整个步骤：

```bash
python download_videos.py --roster data/roster80.json --plan
python download_videos.py /data/longvideo --roster data/roster80.json
```

脚本只保存这 75 个视频，按现有参考源计算约 18.33 GB；**这不是下载流量估计**。它会下载涉及的完整压缩分卷，再选择提取，分卷下载量可能接近数百 GB，且中途需要额外磁盘空间。此脚本的缓存策略建议至少 250 GB 可用空间；这个要求不适用于对方的逐视频下载方式。分卷支持续传，成功提取后删除本程序的缓存归档，后续扩展题单可能重新下载部分分卷。不指定 `--roster` 时仍仅处理冻结 597 题的 465 个视频，不是下载完整基准。

## 3. 预处理

当前先准备每集 20 题，共 80 题。下列命令生成新的本地 ASR、1 fps OpenCLIP 索引和初始帧；不调用远程语言模型。默认父题单仍是 `roster597.json`，`--per-dataset 20` 与冻结的 `roster80.json` 完全对应，便于后续原目录扩展：

```bash
pip install faster-whisper
python -B pava.py prepare --config config.local.json --video-root /data/longvideo --work /data/pava128 --openclip-checkpoint /models/open_clip_pytorch_model.bin --device cuda:0 --asr-model /models/faster-whisper-large-v3 --asr-device cuda --per-dataset 20
```

已有字幕可将 `--asr-model ... --asr-device ...` 替换为 `--transcripts-dir /data/transcripts`。每个文件命名为 `<video_id>.json`，格式为：

```json
{"video_id":"example_video","producer":"actual_transcription_source","segments":[{"id":0,"start":0.0,"end":1.5,"text":"Transcript text."}]}
```

只接受字幕内容和来源信息，不接受答案标签。时间以视频起点为零，单位秒。来源不明保留 `unknown`，不要编造生成模型。给出两个参数时优先读已有字幕，缺失再本地转写。只有明确需要纯视觉实验才用 `--no-asr`。空转写或无音轨会记录状态并使用视觉路径。

本入口生成的 ASR 是 `faster-whisper`，不自动等同于旧实验的 WhisperX/官方字幕；正式比较应保持转写材料一致，并报告实际来源。生成字幕的耗时计入预处理；导入字幕的历史生成成本未知。

同一工作目录重跑准备命令会复用绑定一致的材料；后续获准后去掉 `--per-dataset 20` 可扩展到完整 597 题。更换帧数、模型权重、字幕来源或工作路径时使用新目录。

## 4. 任务顺序与运行

1. 完成 `check.py` 和数据准备，先跑每集 1 题，共 4 题，确认服务接受图片、输出格式正确、usage 可用。
2. 第一版跑 **80 题，每集 20 题**，记录准确率与完整开销。若要单独测不含并发排队的速度，可先用每集 5 题、共 20 题串行测速。
3. 80 题结果核对后，由用户决定是否运行第二版 **597 题**；不要自动扩跑。

这些前缀样本来自本仓库的 597 题题单，**不是历史 80 题试跑名单**。本仓库不包含旧成绩。

默认只检查材料并打印计划；只有 `--execute` 才发出模型请求。批量运行使用 `config.local.json` 中自行设置的 `qa_concurrency`，也可用 `--concurrency` 覆盖：

```bash
python -B pava.py run --config config.local.json --work /data/pava128 --per-dataset 1 --concurrency 1
python -B pava.py run --config config.local.json --work /data/pava128 --per-dataset 1 --concurrency 1 --run-id smoke4 --execute
python -B pava.py run --config config.local.json --work /data/pava128 --per-dataset 5 --concurrency 1 --run-id speed20 --execute
tmux new-session -d -s pava80 'source .venv/bin/activate && python -u -B pava.py run --config config.local.json --work /data/pava128 --per-dataset 20 --run-id pilot80 --execute > /data/pava128/pilot80.log 2>&1'
```

第二版 597 题的命令已备好，**只在用户决定扩跑后执行**；先按同样的本地路径规则自行补齐新增视频，不限定下载方式：

```bash
python -B pava.py prepare --config config.local.json --video-root /data/longvideo --work /data/pava128 --openclip-checkpoint /models/open_clip_pytorch_model.bin --device cuda:0 --asr-model /models/faster-whisper-large-v3 --asr-device cuda
tmux new-session -d -s pava597 'source .venv/bin/activate && python -u -B pava.py run --config config.local.json --work /data/pava128 --run-id main597 --execute > /data/pava128/main597.log 2>&1'
```

两套配置和统计方式保持一致，run 目录分别为 `pilot80` / `main597`。80 题多次被运行时，每批开销分别保留；不要把先前试跑成本漏记。客户端并发与服务端 `max-num-seqs` 都要记录；存在并发排队时，不能把这类时延写成独占测速。

在仓库根目录启动 tmux 命令；新会话必须继承或重新加载自己的认证环境。输入是题干、选项、字幕和抽取帧，GT 不进入推理。只复用媒体预处理材料，每次 run 使用新的理解/观察/答案缓存。已有 run-id 不允许覆盖。401/402/403/429/5xx 会停止新增请求和派发；无自动服务重试。当前入口没有断点续跑命令，不要把新 run 冒充旧 run 的无成本续接。

## 5. 统计、评分与交付

运行结束后：

```bash
python -B report.py /data/pava128/runs/speed20
python -B report.py /data/pava128/runs/pilot80 --labels /data/evaluation/answers.json
python -B report.py /data/pava128/runs/main597 --labels /data/evaluation/answers.json
```

评分标签单独准备为 `{"question_id":"A", ...}`，值为官方选项字母；全部计划题结束后才能读取。失败/未答与准确率一并报告，评分分母为全部计划题。不要将标签放入预处理材料或推理工作目录。

| 统计项 | 输出与口径 |
|---|---|
| 输入、输出、总 token | 每题、每阶段、每数据集及全体；使用服务端 usage，缺失次数单列 |
| LLM calls | 每次实际 HTTP 尝试，包含失败；不是 acquisition 次数 |
| 上下文长度 | 每请求 `prompt_tokens`，报告均值/P95/最大值；区别于每题累计 token |
| 图片量 | 提交次数，重复携带重复计数；不是去重帧数 |
| 在线时延 | `online_qa_seconds` 包含动态检索、抽帧和模型调用；另记 worker / batch 墙钟 |
| 流式速度 | 每调用 TTFT、请求时长、近似解码 tokens/s；不支持时保留未知 |
| 预处理 | 独立视频的索引、初始帧、ASR 准备耗时及本地模型加载；同视频多题只核算一次，再按题摊销 |
| 端到端核算 | 预处理摊销 + 在线 QA；这是核算值，不冒充新测的冷启动墙钟；服务端加载及未知导入 ASR 成本单列 |

输入 token 已包含服务端计入的视觉部分时，不再次添加视觉 token。保留 raw usage，不强行估算缓存、thinking 或视觉 token 细分。组件计时有包含关系，不能把模型加载和总预处理再加一遍。

交付 `config.json`、`inputs.json`、`finished.json`、`results.json`、`cost_summary.json`、`question_costs.csv`，以及 `materials/preparation/*.json` 和实际服务启动/硬件信息。逐请求证据在 `questions/<id>/http/`，包含请求文字、图片和回复，供私下核对，**不要提交到 Git**。

## 文件入口与来源

`pava.py`：准备与运行；`transport.py`：模型协议适配；`common.py`：计数；`report.py`：汇总与事后评分；`check.py`：无网络检查。新机器先运行小批验收，再扩大范围。

`core/` 包含从 [SalesforceAIResearch/ActiveVideoPerception](https://github.com/SalesforceAIResearch/ActiveVideoPerception) 派生的代码及 PAVA 改动。上游许可保留在 [LICENSE.txt](LICENSE.txt)，修改范围见 [NOTICE.md](NOTICE.md)。
