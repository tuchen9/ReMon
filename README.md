# Website Owner Identification through Multi-level Contrastive Representation Learning

## Introduction
ReMon is a novel framework for website owner identification that formulates the task as webpage representation learning. It leverages LLM-based text rewriting to reduce noise and multi-level contrastive learning to capture website–owner relations. ReMon achieves state-of-the-art performance on real-world datasets, especially in challenging scenarios where WHOIS records are incomplete or webpages do not explicitly reveal owner names.

## Requirements
```
python3.10.13
cuda12.1
pytorch2.1.0
numpy 1.26.0
transformers 4.35.0
scipy 1.11.3
scikit-learn 1.3.2
```

## How to run the code
```
python -m torch.distributed.run --nproc_per_node=2 train.py -m ConOA -d WOI_a
python clustering.py -d WOI_a
```
