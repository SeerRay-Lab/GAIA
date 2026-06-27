# GAIA: A Data Flywheel System for Training GUI Test-Time Scaling Critic Models

## Overview

This repository presents **GAIA**, a data flywheel system designed to train **GUI action critic models** that improve the reliability and performance of GUI agents at test time.

Modern GUI agents powered by large vision-language models (LVLMs) have demonstrated strong capabilities in understanding instructions and interacting with interfaces. However, a critical limitation remains: **irreversible errors**—a single incorrect action (e.g., mis-click or wrong input) can derail the entire task.

To address this, GAIA introduces a **critic-driven test-time scaling framework**, where an **Intuitive Critic Model (ICM)** evaluates candidate actions before execution. The system operates in a **data flywheel loop**:

- Collect real action trajectories from GUI agents
- Construct high-quality positive and negative samples
- Train a critic model for action correctness judgment
- Use the critic to guide agent decisions (Best-of-N selection)
- Re-collect harder samples and iteratively refine the critic (ICM → ICM-r2)

This iterative process continuously improves both:
- the **critic’s discriminative capability**
- and the **agent’s execution accuracy**

GAIA enables **plug-and-play performance gains** for both open-source and closed-source GUI agents without requiring expensive retraining.

---

## Dataset

We release the **GAIA Dataset**, which contains large-scale, real-action-based positive and negative samples for training GUI action critics:

- 👉 [GAIA Dataset on HuggingFace](https://huggingface.co/datasets/SeerRay-Lab/GAIA-Dataset-v1.0?utm_source=chatgpt.com)

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

The current repository provides a minimal implementation for:

> Training and evaluating the GAIA-based Intuitive Critic Model (ICM) for GUI action correctness.

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


---
