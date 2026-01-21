Prompt-Guard AI
Systematic Detection of Prompt Injection and Jailbreak Attacks using Classical ML, Deep Neural Networks, and LLM Reasoning

Prompt-Guard AI is an end-to-end AI security research and engineering project focused on detecting prompt-injection and jailbreak attacks targeting Large Language Models (LLMs).
The project performs a methodologically rigorous comparison between feature-based ML classifiers, sequence-aware deep neural models, transformer-based fine-tuned models, and LLM-native reasoning approaches, all evaluated on real-world adversarial datasets.

Problem Context & Motivation

Prompt injection has emerged as a first-class security vulnerability in modern GenAI systems, enabling attackers to:

Override system and developer instructions

Exfiltrate hidden prompts or sensitive data

Bypass alignment and content safety layers

Induce unintended tool invocation or policy violations

Unlike traditional NLP classification tasks, prompt-injection detection must operate under distribution shift, adversarial intent, and semantic obfuscation.
This project investigates which modeling paradigms are most robust under these constraints and how reasoning-based LLM approaches compare against learned discriminative models.

Research Objectives

Design a binary prompt-level classification pipeline for injection and jailbreak detection

Implement and benchmark:

Classical feature-based ML

Recurrent neural architectures

Transformer fine-tuning

LLM prompting and reasoning strategies

Evaluate generalization on adversarially curated jailbreak datasets

Analyze precision-recall trade-offs, robustness, and failure modes

Demonstrate practical deployment via an interactive inference UI

Modeling Approaches
Classical & Neural Discriminative Models

Keyword Rule Baseline
A deterministic heuristic detector for establishing a lower-bound baseline.

TF-IDF + Logistic Regression

Sparse lexical representation using TF-IDF

Linear decision boundary optimized via log-loss

Serves as a strong classical ML baseline with low inference cost

BiLSTM (Bidirectional LSTM)

Learns sequential and contextual dependencies in prompt text

Captures order-sensitive attack patterns missed by bag-of-words models

Trained with cross-entropy loss on balanced prompt distributions

BERT Fine-Tuned (bert-base-uncased)

12-layer transformer (~110M parameters)

Fine-tuned end-to-end for binary sequence classification

Achieves the highest robustness on adversarial jailbreak prompts

Serves as the primary neural reference model

LLM-Native Reasoning Approaches (Gemini 2.5 Flash)

The same task is solved without gradient-based learning, using prompt engineering:

Zero-Shot Classification
Direct instruction-based inference without examples

Few-Shot + Chain-of-Thought (CoT)
Demonstrates attack reasoning using labeled exemplars

Self-Consistency
Multiple independent reasoning paths aggregated via majority vote

These methods evaluate the reasoning capability of LLMs as safety classifiers rather than their generative performance.

Datasets
Prompt Injection Safety Dataset

Source: jayavibhav/prompt-injection-safety (Hugging Face)

Mixed benign and malicious prompts

Used for training and validation

Manually curated attack categories and benign instructions

Evaded Prompt Injection & Jailbreak Dataset

Source: Mindgard/evaded-prompt-injection-and-jailbreak-samples (Hugging Face)

Real-world adversarial jailbreak prompts

Designed to bypass naive safety filters

Used exclusively as an out-of-distribution test set

All datasets are normalized into a unified JSONL schema and split into training, validation, and adversarial test partitions.

Evaluation Methodology

Binary classification metrics:

Accuracy

Precision

Recall

F1-score

Validation on in-distribution data

Stress-testing on adversarial jailbreak samples

Balanced evaluation subset for LLM prompting to ensure fair comparison

Key Findings

BERT fine-tuning delivers the strongest overall performance and adversarial robustness

BiLSTM provides competitive accuracy with significantly lower computational overhead

TF-IDF + Logistic Regression remains a surprisingly effective baseline

LLM prompting methods exhibit:

High recall (aggressive attack detection)

Lower precision (over-flagging benign prompts)

Results suggest that LLMs are better suited as reasoning and explanation layers, not primary detectors

A hybrid architecture—neural classifiers for detection + LLMs for interpretability—is the most practical design pattern.

System Demonstration

A production-style Streamlit application enables real-time inference:

Prompt input interface

Parallel evaluation across all models

Confidence scores and verdict aggregation

Optional LLM reasoning visualization

Run locally:

streamlit run src/app/streamlit_app.py

Engineering Stack

Python

PyTorch

Hugging Face Transformers

scikit-learn

Streamlit

Google Gemini API

Google Cloud Storage (model artifacts)

Repository Notes

Due to GitHub size constraints, trained model artifacts (e.g., fine-tuned BERT weights) are stored externally.
The repository contains complete, reproducible training and evaluation pipelines.
