# Fine Tuning your LLMs with NVIDIA-powered UCS-X and Hugging Face APIs

**Cisco Live Session: DEVNET-2081**

## Overview

Fine-tuning a large language model (LLM) is crucial for adapting it to specific tasks or domains, thereby improving performance and accuracy. This process enhances the model's ability to generate more relevant and context-specific responses, making it effective for specialized applications.

In contrast, Retrieval-Augmented Generation (RAG) boosts performance by using external knowledge sources, suitable for scenarios requiring the most current or specialized information without changing the model's core parameters. Both methods are important and should be employed based on the focus.

In this workshop, you will learn how to fine-tune an LLM using PyTorch and the Hugging Face Trainer API. To evaluate if fine-tuning was effective, attendees will measure the accuracy of the LLM before and after fine-tuning. This workshop will explore writing Python code to leverage NVIDIA's GPUs to fine-tune the LLM running on Cisco’s UCS-X series hardware.

## Acknowledgement

This code is **heavily based on the [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/en/chapter0/1?fw=pt)**, which provides excellent tutorials and educational resources for working with Transformers and fine-tuning language models using PyTorch.

## Prerequisites

* Python 3.8+
* PyTorch
* Transformers (`pip install transformers`)
* Datasets (`pip install datasets`)
* Evaluate (`pip install evaluate`)
* NVIDIA GPU and UCS-X infrastructure (for hardware acceleration)

## Dataset

This example uses the [GLUE MRPC dataset](https://huggingface.co/datasets/glue/viewer/mrpc), which is a sentence pair classification task to determine whether two sentences are semantically equivalent.

## Model

The model used is `bert-base-uncased` from Hugging Face's Transformers library, fine-tuned for binary classification.

## How It Works

1. **Load Dataset**
   Loads the MRPC subset of the GLUE benchmark using the `datasets` library.

2. **Preprocessing**
   Tokenizes sentence pairs and applies truncation and padding.

3. **Model Setup**
   Loads a pre-trained BERT model and prepares it for sequence classification.

4. **Trainer Configuration**
   Defines training arguments, evaluation strategy, and metric computation.

5. **Training**
   Fine-tunes the model on the training set using the Hugging Face `Trainer`.

6. **Evaluation**
   Evaluates the fine-tuned model on the validation set to measure accuracy improvements.

## Usage

```bash
python training.py
```

> Replace `training.py` with your script filename if different.

