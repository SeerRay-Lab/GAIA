<p align="center">
  <img src="assets/logo.jpg" width="120">
</p>

<h1 align="center">
  GAIA: A Data Flywheel System for Training GUI Test-Time Scaling Critic Models
</h1>

<p align="center">
  📄 <a href="https://arxiv.org/abs/2601.18197">arXiv</a> &nbsp;|&nbsp;
  🌟 <a href="https://github.com/SeerRay-Lab/GAIA">GitHub</a> &nbsp;|&nbsp;
  🤗 <a href="https://huggingface.co/datasets/SeerRay-Lab/GAIA-Dataset-v1.0">Dataset</a>
</p>

## Overview

This repository presents **GAIA**, a data flywheel system designed to train **GUI action critic models** that improve the reliability and performance of GUI agents at test time.

Modern GUI agents powered by large vision-language models (LVLMs) have demonstrated strong capabilities in understanding instructions and interacting with interfaces. However, a critical limitation remains: **irreversible errors**—a single incorrect action (e.g., mis-click or wrong input) can derail the entire task.

![GAIA Pipeline](assets/GAIA.png)

To address this, GAIA introduces a **critic-driven test-time scaling framework**, where an **Intuitive Critic Model (ICM)** evaluates candidate actions before execution. The system operates in a **data flywheel loop**:

- Collect real action trajectories from GUI agents
- Construct high-quality positive and negative samples
- Train a critic model for action correctness judgment
- Use the critic to guide agent decisions (Best-of-N selection)
- Re-collect harder samples and iteratively refine the critic (ICM → ICM-r2)

This iterative process continuously improves both:
- the **critic's discriminative capability**
- and the **agent's execution accuracy**

GAIA enables **plug-and-play performance gains** for both open-source and closed-source GUI agents without requiring expensive retraining.

---

## Dataset

We release the **GAIA Dataset**, which contains large-scale, real-action-based positive and negative samples for training GUI action critics:

- 👉 [GAIA Dataset on HuggingFace](https://huggingface.co/datasets/SeerRay-Lab/GAIA-Dataset-v1.0)

Key characteristics:
- Constructed from **real GUI agent interactions** (not synthetic heuristics)
- Balanced **positive / negative action samples**
- Covers diverse GUI environments and action types
- Supports training of binary **action correctness classifiers**

---

## Datasheet

The **datasheet for the dataset** (including data collection, annotation protocol, and distribution analysis) is also provided in this repository.

Please refer to the corresponding file for:
- Data sources and benchmarks
- Labeling strategy (GT alignment vs. deviation)
- Action space definition
- Statistical analysis of samples

---

## Code

A minimal implementation for **training and evaluating the Intuitive Critic Model (ICM)** for GUI action correctness.

### Repository layout

```
.
├── prompts/
│   ├── gui_agent.txt                              # policy/agent system prompt
│   ├── critic_sft.txt                             # SFT critic prompt (answers bare `correct`/`wrong`)
│   └── gui_critic.txt                             # GRPO critic prompt (answers within <link></link>)
├── scripts/
│   ├── train_sft_full.sh                          # full-parameter SFT (ms-swift)
│   ├── train_sft_lora.sh                          # LoRA SFT (ms-swift)
│   └── train_grpo.sh                              # GRPO / RL stage (ms-swift + vLLM)
├── src/
│   ├── generate_sft_dataset.py                    # build critic SFT data from good/bad agent cases
│   ├── infer_critic.py                            # multi-GPU critic inference / accuracy eval
│   ├── benchmark_screenspot.py                    # ScreenSpot-v2 grounding baseline
│   ├── benchmark_screenspot_best_of_n.py          # Best-of-N (no critic)
│   ├── benchmark_screenspot_best_of_n_critic.py   # Best-of-N selection with the critic
│   ├── merge_lora.py                              # merge a LoRA adapter into the base model
│   ├── split_dataset.py                           # shuffle + train/val split
│   ├── metrics.py                                 # precision / recall / F1 from a results jsonl
│   └── smart_resize.py                            # Qwen2.5-VL image resize helper
├── data/sample/                                   # real example rows + screenshots (mirrors the HF dataset)
└── requirements.txt
```

### Setup

```bash
pip install -r requirements.txt        # torch 2.5 / transformers 4.51 / ms-swift / vllm / trl
```

Base models are referenced by HuggingFace id (e.g. `Qwen/Qwen2.5-VL-7B-Instruct`);
override any path via the script/CLI arguments. Trained checkpoints, datasets, and
benchmark images are **not** shipped — point the arguments at your own copies.
Run all commands from the repository root.

### Data format

Training and eval data are JSON lists of chat records, matching the released
[GAIA dataset](https://huggingface.co/datasets/SeerRay-Lab/GAIA-Dataset-v1.0)
(`data/sample/` holds real example rows + screenshots):

```json
{
  "messages": [
    {"role": "system",    "content": "You are an expert in evaluating the performance of a phone operating agent. ... You should whether answer [correct] or [wrong]."},
    {"role": "user",      "content": "The goal of the task (instruction): <instruction>\nAction (plan) history: <history>\nCurrent action of the agent: <action>\nScreenshot: <image>"},
    {"role": "assistant", "content": "correct"}
  ],
  "images": ["data/sample/images/<screenshot>.jpg"]
}
```

The assistant label is the bare string `correct` or `wrong`. The SFT system
prompt is embedded in each record (the SFT scripts pass `--system ""`), so
`prompts/critic_sft.txt` is just a copy for reference. The `<link>correct</link>`
format in `prompts/gui_critic.txt` is the **separate GRPO-stage** prompt
(`train_grpo.sh` passes it via `--system`).

### Pipeline

1. **Build SFT data** from collected good/bad agent action cases. A local
   Qwen3 model summarizes the critic's rationale into a short label (runs fully
   offline, no external API):
   ```bash
   python src/generate_sft_dataset.py good_cases.jsonl bad_cases.jsonl out.json 0 1
   python src/split_dataset.py out.json data/sft 0.8
   ```

2. **Train** (SFT then optional GRPO). Override `MODEL` / `DATASET` / `OUTPUT_DIR`
   via env vars:
   ```bash
   DATASET=data/sft_train.json bash scripts/train_sft_lora.sh
   python src/merge_lora.py --model-path <lora_ckpt> --model-base Qwen/Qwen2.5-VL-7B-Instruct --model-save checkpoints/critic-merged
   DATASET=data/grpo_train.json PLUGIN=<plugin.py> bash scripts/train_grpo.sh
   ```
   > **GRPO note:** `train_grpo.sh` expects an ms-swift external plugin that
   > registers two reward functions, `external_gui_agent_critic_format` and
   > `external_gui_agent_critic_link`. Provide your own `plugin.py` via `PLUGIN`.

3. **Evaluate** the critic's correct/wrong accuracy:
   ```bash
   python src/infer_critic.py --model_path checkpoints/critic-merged \
       --input_file data/sample/critic_eval_sample.json --exp_name my_eval
   python src/metrics.py rst/my_eval_merged.jsonl
   ```

4. **Benchmark** Best-of-N action selection on ScreenSpot-v2 (download the
   dataset to `data/ScreenSpot-v2/`):
   ```bash
   python src/benchmark_screenspot_best_of_n_critic.py --model_path <actor> --critic_path checkpoints/critic-merged --task all
   ```

---

## Citation

If you find this work useful, please consider citing:

```
@inproceedings{wang2026gaia,
  title={GAIA: A Data Flywheel System for Training GUI Test-Time Scaling Critic Models},
  author={Wang, Shaokang and Fu, Pei and Zhang, Ruoceng and Zhang, Shaojie and Xi, Xiuwen and Yang, Jiahui and Qin, Bin and Huang, Ying and Luo, Zhenbo and Luan, Jian},
  booktitle={European Conference on Computer Vision},
  year={2026},
  organization={Springer}
}
```

---

## License

This project is released under the [Apache 2.0 License](./LICENSE).
