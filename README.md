# Code for DSMV

This repository contains the code for the paper:

> **Dual-Stage Multi-View Transformer for Medical Difference Visual Question Answering**

## Overview

Medical Difference Visual Question Answering (Diff-VQA) aims to answer questions about the *differences* between two medical images (e.g., a "before" and an "after" image). This repository implements the **Dual-Stage Multi-View Transformer (DSMV)** model, which:

- Fuses multi-view visual features with question representations in a dual-stage transformer architecture;
- Detects fine-grained visual changes between paired images;
- Generates natural-language answers to difference and non-difference questions.

## Repository Structure

```
model/
├── configs/            # Configuration files (Python + YAML)
├── datasets/           # Dataset loading and preprocessing
├── models/             # DSMV model components
├── utils/              # Utilities (logging, runtime, etc.)
├── train_mimic.py      # Training script
├── test_mimic.py       # Evaluation / testing script
└── evaluation.py       # Evaluation metrics
```

## Requirements

Please refer to the configuration files for model hyperparameters. The code depends on `torch`, `pycocotools`, and the standard scientific Python stack.

## Usage

```bash
# Train
python model/train_mimic.py

# Evaluate
python model/test_mimic.py
```

Use `--help` on either script for the full list of options.
