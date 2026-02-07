# Prompt-Guard AI
## Prompt Injection & Jailbreak Detection using ML, Deep Learning, and LLM Reasoning

Prompt-Guard AI is an end-to-end AI security project focused on detecting **prompt injection and jailbreak attacks** targeting Large Language Models (LLMs). The project systematically compares **classical machine learning**, **deep neural architectures**, **transformer-based fine-tuning**, and **LLM-native reasoning approaches** on the same security-critical task.

---

## Background and Motivation

Prompt injection has emerged as a critical vulnerability in modern GenAI systems, allowing malicious users to override system instructions, bypass safety controls, and extract sensitive information. Unlike traditional NLP tasks, prompt injection detection must operate under adversarial intent, semantic obfuscation, and distribution shift.

This project explores which modeling paradigms are most effective and robust for prompt-level security enforcement.                                                 da

-
## Project Objectives

- Design a binary prompt-level classifier for injection and jailbreak detection  
- Implement and benchmark multiple AI paradigms on the same datasets  
- Evaluate robustness against adversarial jailbreak prompts  
- Compare precision, recall, and failure modes across approaches  
- Demonstrate practical deployment via an interactive inference interface  

---
## Modeling Approaches

### Classical and Neural Models

**Keyword Rule Baseline**  
A deterministic heuristic detector used as a lower-bound reference.

**TF-IDF + Logistic Regression**  
A sparse lexical feature representation with a linear decision boundary, serving as a strong classical ML baseline with low inference cost.

**BiLSTM (Bidirectional LSTM)**  
A sequence-aware deep learning model capturing contextual and order-sensitive attack patterns that bag-of-words models fail to detect.

**BERT Fine-Tuned (bert-base-uncased)**  
A 12-layer transformer (~110M parameters) fine-tuned end-to-end for binary sequence classification. This model provides the highest robustness against adversarial jailbreak prompts.

---

### LLM Reasoning-Based Detection

The same task is evaluated using **Gemini 2.5 Flash** without gradient-based learning:

- Zero-Shot Prompting  
- Few-Shot Prompting with Chain-of-Thought  
- Self-Consistency via multiple reasoning paths and majority voting  

These methods assess the reasoning capability of LLMs as safety classifiers.

---

## Datasets

**Prompt Injection Safety Dataset**  
Source: `jayavibhav/prompt-injection-safety` (Hugging Face)  
Used for training and validation. Contains labeled benign and malicious prompts.

**Evaded Prompt Injection & Jailbreak Dataset**  
Source: `Mindgard/evaded-prompt-injection-and-jailbreak-samples` (Hugging Face)  
Used as an adversarial test set containing real-world jailbreak attempts.

All datasets are normalized into a unified JSONL schema and split into training, validation, and test partitions.

---

## Evaluation Methodology

- Binary classification metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
- In-distribution validation evaluation  
- Out-of-distribution adversarial testing  
- Balanced evaluation subset for LLM prompting comparison  

---

## Key Findings

- BERT fine-tuning achieves the highest accuracy and adversarial robustness  
- BiLSTM offers strong performance with lower computational overhead  
- TF-IDF + Logistic Regression remains a surprisingly competitive baseline  
- LLM prompting methods exhibit high recall but lower precision, frequently over-flagging benign prompts  

Overall, neural classifiers are more reliable for detection, while LLMs are better suited for reasoning and explanation layers.

---

## Interactive Demo

A Streamlit-based application enables real-time inference:

- User prompt input  
- Parallel predictions across all models  
- Confidence scores and final verdict aggregation  
- Optional LLM reasoning output  

## Repository Notes
Due to GitHub size constraints, trained model artifacts (e.g., fine-tuned BERT weights) are stored externally.
The repository contains complete, reproducible training and evaluation pipelines.

