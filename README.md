# Master thesis project: Personality-Aware LLM-based Dialogue Simulation for Pedestrians

The goal is to fine-tune an LLM for personality-based dialogue generation for social simulation purposes. 

### Description
Social simulation is used to study interactions, scenarios or societal patterns in a experimental environments, that are difficult to study in real life for various reasons. LLMs bring the potential to change how simulations are conducted, possibly replacing or enhancing rule-based simulation approaches.
One way simulation can be enhanced is by generating synthetic but realistic conversations that can make simulation more credible and introduce new facets of interaction. 

This project aims to fine-tune an LLM to do exactly that: to generate realistic, scenario-based dialogue between simulated agents in multi-agent simulations. To make it more realistic, this dialogue should convey the personality of the speaker. 
Therefore the fine-tuning is conducted using a dataset of foum posts, labelled with the Myers-Briggs Type Indicator (MBTI). 

### Methodology
#### Data
Initial dataset: [Kaggle MBTI](https://www.kaggle.com/datasets/datasnaek/mbti-type)
Cleaned and augmented dataset: [MBTI_balanced](https://huggingface.co/datasets/DrinkIcedT/mbti_balanced)

#### Data Preparation
The data is cleaned and preprocessed with a common NLP cleaning pipeline, see [Preparation Notebook](notebooks/exploration_and_preprocessing.ipynb)

Since the dataset is heavily unbalanced, I needed to oversample as well as undersample, see [Augmentation Notebook](notebooks/data_augmentation.ipynb)
The evaluation of the augmented data was done [here](notebooks/augmented_data_eval.ipynb) and [here](notebooks/PerplexityScore.ipynb)

The following metrics for the evaluation were calculated:
- BLEU
- Self-BLEU
- TTR
- Perplexity
- Cosine Similarity

#### Model Selection and Finetuning
Model: [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
Fine-tuning was done on the Alvis HPC cluster from Chalmers as part of the National Academic Infrastructure for Super­computing in Sweden ([NAISS](https://www.naiss.se/))

### Results
Work in progress!
