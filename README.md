# CL-DQE

PyTorch implementation of **CL-DQE: Cross-Language Contrastive Representation Learning for English-Chinese Semantic Alignment and Translation Quality Estimation**.

CL-DQE is a unified framework for English-Chinese semantic alignment and reference-free machine translation quality estimation. It combines multi-scale cross-language representation learning, contrastive semantic alignment, local error modeling, multi-factor quality estimation, and graph-based quality inference.

---

## Overview

CL-DQE jointly addresses two related tasks:

- English-Chinese semantic alignment identification
- Reference-free machine translation quality estimation

The framework contains the following main components:

- **MSCLE**: Multi-Scale Cross-Language Encoding
- **CCRL**: Cross-Language Contrastive Representation Learning
- **ESA**: Explicit Semantic Alignment
- **LEA**: Local Error-Aware Modeling
- **MSQE**: Multi-Factor Sentence-Level Quality Estimation
- **GQI**: Graph-based Quality Inference

The complete framework is trained in an end-to-end multi-task learning manner.

---

## Model Architecture

The model first encodes English and Chinese sentences using a multi-scale dual-channel encoder.

Fine-grained BiGRU representations are used to capture token-level contextual information, while coarse-grained BiGRU, CNN, and dilated convolution are introduced to model long-range and multi-scale semantic dependencies.

Cross-language contrastive learning is then used to construct a shared semantic space between English and Chinese sentence representations.

The learned representations are further processed by:

1. an explicit semantic alignment prediction head;
2. a local error-aware cross-language attention mechanism;
3. a multi-factor translation quality estimation module;
4. a graph neural network for final quality inference.

---

## Datasets

The experiments use two public datasets.

### PAWS-X

PAWS-X is used for English-Chinese semantic alignment evaluation.

Dataset:

https://huggingface.co/datasets/hgissbkh/paws-x

### WMT20 Quality Estimation

The WMT20 English-Chinese Quality Estimation dataset is used for sentence-level translation quality estimation.

The dataset contains:

- English source sentences
- Chinese machine translation outputs
- Human quality annotations / HTER scores

Dataset information:

https://github.com/deep-spin/deep-spin.github.io

---

## Requirements

Recommended environment:

```text
Python >= 3.8
PyTorch >= 1.12
CUDA >= 11.3
