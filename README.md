# Awesome-AI-Papers

This repository is used to collect papers and code in the field of AI. The contents contain the following parts:

## Table of Content  

- [NLP](#nlp)
  - [Word2Vec](#1-word2vec)
  - [Seq2Seq](#2-seq2seq)
  - [Pretraining](#3-pretraining)
    - [Large Language Model](#31-large-language-model)
    - [LLM Application](#32-llm-application)
    - [LLM Technique](#33-llm-technique)
    - [LLM Theory](#34-llm-theory)
- [CV](#cv)
- [Multimodal](#multimodal)
  - [Audio](#1-audio)
  - [BLIP](#2-blip)
  - [CLIP](#3-clip)
  - [Diffusion Model](#4-diffusion-model)
  - [Multimodal LLM](#5-multimodal-llm)
  - [Text2Image](#6-text2image)
  - [Text2Video](#7-text2video)
  - [Survey for Multimodal](#8-survey-for-multimodal)
- [Reinforcement Learning](#reinforcement-learning)
- [GNN](#gnn)
- [Transformer Architecture](#transformer-architecture)

```bash
  ├─ NLP/  
  │  ├─ Word2Vec/  
  │  ├─ Seq2Seq/           
  │  └─ Pretraining/  
  │    ├─ Large Language Model/          
  │    ├─ LLM Application/ 
  │      ├─ AI Agent/          
  │      ├─ Academic/          
  │      ├─ Code/       
  │      ├─ Financial Application/
  │      ├─ Information Retrieval/  
  │      ├─ Math/     
  │      ├─ Medicine and Law/   
  │      ├─ Recommend System/      
  │      └─ Tool Learning/             
  │    ├─ LLM Technique/ 
  │      ├─ Alignment/          
  │      ├─ Context Length/          
  │      ├─ Corpus/       
  │      ├─ Evaluation/
  │      ├─ Hallucination/  
  │      ├─ Inference/     
  │      ├─ MoE/   
  │      ├─ PEFT/     
  │      ├─ Prompt Learning/   
  │      ├─ RAG/       
  │      └─ Reasoning/       
  │    ├─ LLM Theory/       
  │    └─ Chinese Model/             
  ├─ CV/  
  │  ├─ CV Application/          
  │  ├─ Contrastive Learning/         
  │  ├─ Foundation Model/ 
  │  ├─ Generative Model (GAN and VAE)/          
  │  ├─ Image Editing/          
  │  ├─ Object Detection/          
  │  ├─ Semantic Segmentation/            
  │  └─ Video/          
  ├─ Multimodal/       
  │  ├─ Audio/          
  │  ├─ BLIP/         
  │  ├─ CLIP/        
  │  ├─ Diffusion Model/   
  │  ├─ Multimodal LLM/          
  │  ├─ Text2Image/          
  │  ├─ Text2Video/            
  │  └─ Survey/           
  │─ Reinforcement Learning/ 
  │─ GNN/ 
  └─ Transformer Architecture/          
```

---

## NLP

### 1. Word2Vec

- **Efficient Estimation of Word Representations in Vector Space**, _Mikolov et al._, arxiv 2013. \[[paper](https://arxiv.org/abs/1301.3781)\]
- **Distributed Representations of Words and Phrases and their Compositionality**, _Mikolov et al._, NIPS 2013. \[[paper](https://arxiv.org/abs/1310.4546)\]
- **Distributed representations of sentences and documents**, _Le and Mikolov_, ICML 2014. \[[paper](https://arxiv.org/abs/1405.4053)\]
- **Word2vec Explained: deriving Mikolov et al.'s negative-sampling word-embedding method**, _Goldberg and Levy_, arxiv 2014. \[[paper](https://arxiv.org/abs/1402.3722)\]
- **word2vec Parameter Learning Explained**, _Rong_, arxiv 2014. \[[paper](https://arxiv.org/abs/1411.2738)\]
- **Glove: Global vectors for word representation.**，_Pennington et al._, EMNLP 2014. \[[paper](https://aclanthology.org/D14-1162/)\]\[[code](https://github.com/stanfordnlp/GloVe)\]
- fastText: **Bag of Tricks for Efficient Text Classification**, _Joulin et al._, arxiv 2016. \[[paper](https://arxiv.org/abs/1607.01759)\]\[[code](https://github.com/facebookresearch/fastText)\]
- ELMo: **Deep Contextualized Word Representations**, _Peters et al._, arxiv. 2018. \[[paper](https://arxiv.org/abs/1802.05365)\]

### 2. Seq2Seq

- **Generating Sequences With Recurrent Neural Networks**, _Graves_, arxiv 2013. \[[paper](https://arxiv.org/abs/1308.0850)\]
- **Sequence to Sequence Learning with Neural Networks**, _Sutskever et al._, NeruIPS 2014. \[[paper](https://arxiv.org/abs/1409.3215)\]
- **Neural Machine Translation by Jointly Learning to Align and Translate**, _Bahdanau et al._, ICLR 2015. \[[paper](https://arxiv.org/abs/1409.0473)\]\[[code](https://github.com/lisa-groundhog/GroundHog)\]
- **On the Properties of Neural Machine Translation: Encoder-Decoder Approaches**, _Cho et al._, arxiv 2014. \[[paper](https://arxiv.org/abs/1409.1259)\]
- **Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation**, _Cho et al._, arxiv 2014. \[[paper](https://arxiv.org/abs/1406.1078)\]
- \[[fairseq](https://github.com/facebookresearch/fairseq)\]

### 3. Pretraining

- **Attention Is All You Need**, _Vaswani et al._, NIPS 2017. \[[paper](https://arxiv.org/abs/1706.03762)\]\[[code](https://github.com/tensorflow/tensor2tensor)\]
- GPT: **Improving language understanding by generative pre-training**, _Radford et al._, preprint 2018.  \[[paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)\]\[[code](https://github.com/openai/finetune-transformer-lm)\]
- GPT-2: **Language Models are Unsupervised Multitask Learners**, _Radford et al._, OpenAI blog 2019. \[[paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)\]\[[code](https://github.com/openai/gpt-2)\]
- GPT-3: **Language Models are Few-Shot Learners**, _Brown et al._, NeurIPS 2020. \[[paper](https://arxiv.org/abs/2005.14165)\]\[[code](https://github.com/karpathy/nanoGPT)\]
- InstructGPT: **Training language models to follow instructions with human feedback**, _Ouyang et al._, NeurIPS 2022. \[[paper](https://arxiv.org/abs/2203.02155)\]\[[code]\]
- **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**, _Devlin et al._, arxiv 2018. \[[paper](https://arxiv.org/abs/1810.04805)\]\[[code](https://github.com/google-research/bert)\]
- **RoBERTa: A Robustly Optimized BERT Pretraining Approach**, _Liu et al._, arxiv 2019. \[[paper](https://arxiv.org/abs/1907.11692)\]\[[code](https://github.com/facebookresearch/fairseq)\]
- **What Does BERT Look At_An Analysis of BERT's Attention**, _Clark et al._, arxiv 2019. \[[paper](https://arxiv.org/abs/1906.04341)\]\[[code](https://github.com/clarkkev/attention-analysis)\]
- **DeBERTa: Decoding-enhanced BERT with Disentangled Attention**, _He et al._, ICLR 2021. \[[paper](https://arxiv.org/abs/2006.03654)\]\[[code](https://github.com/microsoft/DeBERTa)\]
- **DistilBERT: a distilled version of BERT_smaller, faster, cheaper and lighter** _Sanh et al._, arxiv 2019. \[[paper](https://arxiv.org/abs/1910.01108)\]\[[code](https://github.com/huggingface/transformers)\]
- **BERT Rediscovers the Classical NLP Pipeline**, _Tenney et al._, arxiv 2019. \[[paper](https://arxiv.org/abs/1905.05950)\]\[[code](https://github.com/nyu-mll/jiant)\]
- **TinyStories: How Small Can Language Models Be and Still Speak Coherent English**, _Eldan and Li_, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.07759)\]\[[code]\]

#### 3.1 Large Language Model

- **A Survey of Large Language Models**, _Zhao etal._, arxiv 2023. \[[paper](https://arxiv.org/abs/2303.18223)\]\[[code](https://github.com/RUCAIBox/LLMSurvey)\]
- **Efficient Large Language Models: A Survey**, _Wan et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.03863)\]\[[code](https://github.com/AIoT-MLSys-Lab/Efficient-LLMs-Survey)\]
- **Challenges and Applications of Large Language Models**, _Kaddour et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2307.10169)\]
- **A Comprehensive Survey on Pretrained Foundation Models: A History from BERT to ChatGPT**, _Zhou et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2302.09419)\]
- Anthropic: **Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback**, _Bai et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2204.05862)\]\[[code](https://github.com/anthropics/hh-rlhf)\]
- Anthropic: **Constitutional AI: Harmlessness from AI Feedback**, _Bai et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2212.08073)\]\[[code](https://github.com/anthropics/ConstitutionalHarmlessnessPaper)\]
- Anthropic: **Model Card and Evaluations for Claude Models**, Anthropic, 2023. \[[paper](https://www-files.anthropic.com/production/images/Model-Card-Claude-2.pdf)\]
- **BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension**, _Lewis et al._, arxiv 2019. \[[paper](https://arxiv.org/abs/1910.13461)\]\[[code]\]
- **BLOOM_A 176B-Parameter Open-Access Multilingual Language Model**, _BigScience Workshop_, arxiv 2022. \[[paper](https://arxiv.org/abs/2211.05100)\]\[[code]\]
- **Code Llama: Open Foundation Models for Code**, _Rozière et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2308.12950)\]\[[code](https://github.com/facebookresearch/codellama)\]
- Codex: **Evaluating Large Language Models Trained on Code**, _Chen et al._, arxiv 2021. \[[paper](https://arxiv.org/abs/2107.03374)\]\[[code](https://github.com/openai/human-eval)\]
- **Colossal-AI: A Unified Deep Learning System For Large-Scale Parallel Training**, _Li et al._, ICPP 2023. \[[paper](https://arxiv.org/abs/2110.14883)\]\[[code](https://github.com/hpcaitech/ColossalAI)\]
- **Gemini: A Family of Highly Capable Multimodal Models**, _Gemini Team, Google_, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.11805)\]\[[Unofficial Implementation](https://github.com/kyegomez/Gemini)\]
- **GPT-4 Technical Report**, _OpenAI_, arxiv 2023. \[[paper](https://arxiv.org/abs/2303.08774)\]
- **GPT-4V(ision) System Card**, _OpenAI_, OpenAI blog 2023. \[[paper](https://openai.com/research/gpt-4v-system-card)\]
- **Sparks of Artificial General Intelligence_Early experiments with GPT-4**, _Bubeck et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2303.12712)\]
- **The Dawn of LMMs_Preliminary Explorations with GPT-4V(ision)**, _Yang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.17421)\]\[[code](https://github.com/guidance-ai/guidance)\]
- **LaMDA: Language Models for Dialog Applications**, _Thoppilan et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2201.08239)\]\[[code](https://github.com/conceptofmind/LaMDA-rlhf-pytorch)\]
- **LLaMA: Open and Efficient Foundation Language Models**, _Touvron et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2302.13971)\]\[[code](https://github.com/facebookresearch/llama/tree/llama_v1)\]\[[llama.cpp](https://github.com/ggerganov/llama.cpp)\]
- **Llama 2: Open Foundation and Fine-Tuned Chat Models**, _Touvron et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2307.09288)\]\[[code](https://github.com/facebookresearch/llama)\]\[[llama2.c](https://github.com/karpathy/llama2.c)\]
- **Stanford Alpaca: An Instruction-following LLaMA Model**, _Taori et al._, Stanford blog 2023. \[[paper](https://crfm.stanford.edu/2023/03/13/alpaca.html)\]\[[code](https://github.com/tatsu-lab/stanford_alpaca)\]
- **Mistral 7B**, _Jiang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.06825)\]\[[code](https://github.com/mistralai/mistral-src)\]
- Minerva: **Solving Quantitative Reasoning Problems with Language Models**, _Lewkowycz et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2206.14858)\]
- **OPT: Open Pre-trained Transformer Language Models**, _Zhang et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2205.01068)\]\[[code](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT)\]
- **PaLM: Scaling Language Modeling with Pathways**, _Chowdhery et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2204.02311)\]\[[code](https://github.com/lucidrains/PaLM-pytorch)\]\[[PaLM-rlhf-pytorch](https://github.com/lucidrains/PaLM-rlhf-pytorch)\]
- **PaLM 2 Technical Report**, _Anil et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.10403)\]
- **PaLM-E: An Embodied Multimodal Language Model**, _Driess et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2303.03378)\]\[[code](https://github.com/kyegomez/PALM-E)\]
- T5: **Exploring the limits of transfer learning with a unified text-to-text transformer**, _Raffel et al._, Journal of Machine Learning Research 2023. \[[paper](https://arxiv.org/abs/1910.10683)\]\[[code](https://github.com/google-research/text-to-text-transfer-transformer)\]
- **Visual ChatGPT: Talking, Drawing and Editing with Visual Foundation Models**, _Wu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2303.04671)\]\[[code](https://github.com/moymix/TaskMatrix)\]
- **WebGPT: Browser-assisted question-answering with human feedback**, _Nakano et al._, arxiv 2021. \[[paper](https://arxiv.org/abs/2112.09332)\]

#### 3.2 LLM Application

- **A Watermark for Large Language Models**， _Kirchenbauer et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2301.10226)\]\[[code](https://github.com/jwkirchenbauer/lm-watermarking)\]
- **SeqXGPT: Sentence-Level AI-Generated Text Detection**, _Wang et al._, EMNLP 2023. \[[paper](https://arxiv.org/abs/2310.08903)\]\[[code](https://github.com/Jihuai-wpy/SeqXGPT)\]
- **AlpaGasus: Training A Better Alpaca with Fewer Data**, _Chen et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2307.08701)\]\[[code](https://github.com/gpt4life/alpagasus)\]
- **AutoMix: Automatically Mixing Language Models**, _Madaan et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.12963)\]\[[code](https://github.com/automix-llm/automix)\]
- **ChipNeMo: Domain-Adapted LLMs for Chip Design**, _Liu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.00176)\]
- **GAIA: A Benchmark for General AI Assistants**, _Mialon et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.12983)\]\[[code](https://huggingface.co/gaia-benchmark)\]
- **HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face**, _Shen et al._, NeurIPS 2023. \[[paper](https://arxiv.org/abs/2303.17580)\]\[[code](https://github.com/microsoft/JARVIS)\]
- **MemGPT: Towards LLMs as Operating Systems**, _Packer et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.08560)\]\[[code](https://github.com/cpacker/MemGPT)\]
- **OpenChat: Advancing Open-source Language Models with Mixed-Quality Data**, _Wang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.11235)\]\[[code](https://github.com/imoneoi/openchat)\]
- **Orca: Progressive Learning from Complex Explanation Traces of GPT-4**, _Mukherjee et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2306.02707)\]
- **PDFTriage: Question Answering over Long, Structured Documents**, _Saad-Falcon et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.08872)\]\[[code]\]
- **Prompt2Model: Generating Deployable Models from Natural Language Instructions**, _Viswanathan et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2308.12261)\]\[[code](https://github.com/neulab/prompt2model)\]
- **Shepherd: A Critic for Language Model Generation**, _Wang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2308.04592)\]\[[code](https://github.com/facebookresearch/Shepherd)\]
- **Alpaca: A Strong, Replicable Instruction-Following Model**, _Taori et al._, Stanford Blog 2023. \[[paper](https://crfm.stanford.edu/2023/03/13/alpaca.html)\]\[[code](https://github.com/tatsu-lab/stanford_alpaca)\]
- Vicuna: **Judging LLM-as-a-judge with MT-Bench and Chatbot Arena**, _Zheng et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2306.05685)\]\[[code](https://github.com/lm-sys/FastChat)\]
- **WizardLM: Empowering Large Language Models to Follow Complex Instructions**, _Xu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2304.12244)\]\[[code](https://github.com/nlpxucan/WizardLM)\]
- **WebGLM: Towards An Efficient Web-Enhanced Question Answering System with Human Preferences**, _Liu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2306.07906)\]\[[code](https://github.com/THUDM/WebGLM)\]

##### 3.2.1 AI Agent

- **RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control**, _Brohan et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2307.15818)\]\[[Unofficial Implementation](https://github.com/kyegomez/RT-2)\]

##### 3.2.2 Academic

##### 3.2.3 Code

##### 3.2.4 Financial Application

##### 3.2.5 Information Retrieval

- **Large Language Models for Information Retrieval: A Survey**, _Zhu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2308.07107)\]\[[code](https://github.com/RUC-NLPIR/LLM4IR-Survey)\]

##### 3.2.6 Math

##### 3.2.7 Medicine and Law

- Med-PaLM: **Large language models encode clinical knowledge**, _Singhal et al._, Nature 2023. \[[paper](https://www.nature.com/articles/s41586-023-06291-2)\]\[[Unofficial Implementation](https://github.com/kyegomez/Med-PaLM)\]

##### 3.2.8 Recommend System

##### 3.2.9 Tool Learning

- **Tool Learning with Foundation Models**, _Qin et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2304.08354)\]\[[code](https://github.com/OpenBMB/BMTools)\]
- **Toolformer: Language Models Can Teach Themselves to Use Tools**, _Schick et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2302.04761)\]\[[code](https://github.com/lucidrains/toolformer-pytorch)\]

#### 3.3 LLM Technique

##### 3.3.1 Alignment

##### 3.3.2 Context Length

##### 3.3.3 Corpus

##### 3.3.4 Evaluation

##### 3.3.5 Hallucination

- **Holistic Evaluation of Language Models**, _Liang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2211.09110)\]\[[code](https://github.com/stanford-crfm/helm)\]

##### 3.3.6 Inference

##### 3.3.7 MoE

##### 3.3.8 PEFT (Parameter-efficient Fine-tuning)

##### 3.3.9 Prompt Learning

- **OpenPrompt: An Open-source Framework for Prompt-learning**, _Ding et al._, arxiv 2021. \[[paper](https://arxiv.org/abs/2111.01998)\]\[[code](https://github.com/thunlp/OpenPrompt)\]
- \[[PromptPapers](https://github.com/thunlp/PromptPapers)\]

##### 3.3.10 RAG (Retrieval Augmented Generation)

- **Retrieval-Augmented Generation for Large Language Models: A Survey**, _Gao et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.10997)\]\[[code](https://github.com/Tongji-KGLLM/RAG-Survey)\]

##### Text Embedding

- **Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks**, _Reimers et al._, EMNLP 2019. \[[paper](https://arxiv.org/abs/1908.10084)\]\[[code](https://github.com/UKPLab/sentence-transformers)\]
- **SimCSE: Simple Contrastive Learning of Sentence Embeddings**, _Gao et al._, EMNLP 2021. \[[paper](https://arxiv.org/abs/2104.08821)\]\[[code](https://github.com/princeton-nlp/SimCSE)\]
- OpenAI: **Text and Code Embeddings by Contrastive Pre-Training**, _Neelakantan et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2201.10005)\]\[[blog](https://openai.com/blog/introducing-text-and-code-embeddings)\]
- BGE: **C-Pack: Packaged Resources To Advance General Chinese Embedding**, _Xiao et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.07597)\]\[[code](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/baai_general_embedding)\]\[[FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)\]
- LLM-Embedder: **Retrieve Anything To Augment Large Language Models**, _Zhang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.07554)\]\[[code](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_embedder)\]\[[FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)\]
- **Improving Text Embeddings with Large Language Models**, _Wang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.00368)\]\[[code](https://github.com/microsoft/unilm)\]

##### 3.3.11 Reasoning

- **Tree of Thoughts: Deliberate Problem Solving with Large Language Models**, _Yao et al._, NeurIPS 2023. \[[paper](https://arxiv.org/abs/2305.10601)\]\[[code](https://github.com/princeton-nlp/tree-of-thought-llm)\]\[[Plug in and Play Implementation](https://github.com/kyegomez/tree-of-thoughts)\]

#### 3.4 LLM Theory

#### 3.5 Chinese Model

---

## CV

### 1. Basic for CV

### 2. Contrastive Learning

### 3. CV Application

### 4. Foundation Model

- ViT: **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**, _Dosovitskiy et al._, ICLR 2021. \[[paper](https://arxiv.org/abs/2010.11929)\]\[[code](https://github.com/google-research/vision_transformer)\]\[[Pytorch Implementation](https://github.com/lucidrains/vit-pytorch)\]

### 5. Generative Model (GAN and VAE)

- GAN: **Generative Adversarial Networks**, _Goodfellow et al._, arxiv 2014. \[[paper](https://arxiv.org/abs/1406.2661)\]\[[code](https://github.com/goodfeli/adversarial)\]\[[Pytorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN)\]
- GigaGAN: **Scaling up GANs for Text-to-Image Synthesis**, _Kang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2303.05511)\]\[[code](https://github.com/lucidrains/gigagan-pytorch)\]
- \[[pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)\]
- VAE: **Auto-Encoding Variational Bayes**, _Kingma et al._, arxiv 2013. \[[paper](https://arxiv.org/abs/1312.6114)\]\[[code](https://github.com/jaanli/variational-autoencoder)\]\[[Pytorch-VAE](https://github.com/AntixK/PyTorch-VAE)\]
- VQ-VAE: **Neural Discrete Representation Learning**, _Oord et al._, NIPS 2017. \[[paper](https://arxiv.org/abs/1711.00937)\]\[[code](https://github.com/AntixK/PyTorch-VAE/blob/master/models/vq_vae.py)\]\[[vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch)\]
- VQ-VAE-2: **Generating Diverse High-Fidelity Images with VQ-VAE-2**, Razavi et al._, arxiv 2019. \[[paper](https://arxiv.org/abs/1906.00446)\]\[[code](https://github.com/rosinality/vq-vae-2-pytorch)\]

### 6. Image Editing

### 7. Object Detection

### 8. Semantic Segmentation

- **U-Net: Convolutional Networks for Biomedical Image Segmentation**, _Ronneberger et al._, arxiv 2015. \[[paper](https://arxiv.org/abs/1505.04597)\]\[[code](https://github.com/milesial/Pytorch-UNet)\]

### 9. Video

### 10. Survey for CV

---

## Multimodal

### 1. Audio

- Whisper: **Robust Speech Recognition via Large-Scale Weak Supervision**, _Radford et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2212.04356)\]\[[code](https://github.com/openai/whisper)\]
- **WhisperX: Time-Accurate Speech Transcription of Long-Form Audio**, _Bain et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2303.00747)\]\[[code](https://github.com/m-bain/whisperX)\]
- VALL-E: **Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers**, _Wang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2301.02111)\]\[[code](https://github.com/microsoft/unilm)\]
- VALL-E-X: **Speak Foreign Languages with Your Own Voice: Cross-Lingual Neural Codec Language Modeling**, _Zhang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2303.03926)\]\[[code](https://github.com/microsoft/unilm)\]
- **Seamless: Multilingual Expressive and Streaming Speech Translation**, _Seamless Communication et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.05187)\]\[[code](https://github.com/facebookresearch/seamless_communication)\]
- **SeamlessM4T: Massively Multilingual & Multimodal Machine Translation**, _Seamless Communication et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2308.11596)\]\[[code](https://github.com/facebookresearch/seamless_communication)\]
- **StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models**, _Li et al._, NeurIPS 2023. \[[paper](https://arxiv.org/abs/2306.07691)\]\[[code](https://github.com/yl4579/StyleTTS2)\]
- **Amphion: An Open-Source Audio, Music and Speech Generation Toolkit**, _Zhang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.09911)\]\[[code](https://github.com/open-mmlab/Amphion)\]
- **Distil-Whisper: Robust Knowledge Distillation via Large-Scale Pseudo Labelling**，_Gandhi et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.00430)\]\[[code](https://github.com/huggingface/distil-whisper)\]
- **LLaSM: Large Language and Speech Model**, _Shu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2308.15930)\]\[[code](https://github.com/LinkSoul-AI/LLaSM)\]

- **Github Repositories**
- [https://github.com/coqui-ai/TTS](https://github.com/coqui-ai/TTS)
- [https://github.com/suno-ai/bark](https://github.com/suno-ai/bark)
- [https://github.com/babysor/MockingBird](https://github.com/babysor/MockingBird)
- [https://github.com/netease-youdao/EmotiVoice](https://github.com/netease-youdao/EmotiVoice)
- [https://github.com/fishaudio/Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [https://github.com/jaywalnut310/vits](https://github.com/jaywalnut310/vits)
- [https://github.com/pyannote/pyannote-audio](https://github.com/pyannote/pyannote-audio)
- [https://github.com/alibaba-damo-academy/FunASR](https://github.com/alibaba-damo-academy/FunASR)
- [https://github.com/jianchang512/clone-voice](https://github.com/jianchang512/clone-voice)
- [https://github.com/OpenTalker/video-retalking](https://github.com/OpenTalker/video-retalking)
- [https://github.com/Rudrabha/Wav2Lip](https://github.com/Rudrabha/Wav2Lip)
- [https://github.com/Zz-ww/SadTalker-Video-Lip-Sync](https://github.com/Zz-ww/SadTalker-Video-Lip-Sync)
- [https://github.com/OpenTalker/SadTalker](https://github.com/OpenTalker/SadTalker)

### 2. Blip

- ALBEF: **Align before Fuse: Vision and Language Representation Learning with Momentum Distillation**, _Li et al._, NeurIPS 2021. \[[paper](https://arxiv.org/abs/2107.07651)\]\[[code](https://github.com/salesforce/ALBEF)\]
- **BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation**, _Li et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2201.12086)\]\[[code](https://github.com/salesforce/BLIP)\]
- **BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models**, _Li et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2301.12597)\]\[[code](https://github.com/salesforce/LAVIS/tree/main/projects/blip2)\]
- **InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning**, _Dai et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.06500)\]\[[code](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip)\]
- **X-InstructBLIP: A Framework for aligning X-Modal instruction-aware representations to LLMs and Emergent Cross-modal Reasoning**, _Panagopoulou et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.18799)\]\[[code](https://github.com/artemisp/LAVIS-XInstructBLIP)\]
- **LAVIS: A Library for Language-Vision Intelligence**, _Li et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2209.09019)\]\[[code](https://github.com/salesforce/LAVIS)\]
- BeiT-V3: **Image as a Foreign Language: BEiT Pretraining for All Vision and Vision-Language Tasks**, _Wang et al._, CVPR 2023. \[[paper](https://arxiv.org/abs/2208.10442)\]\[[code](https://github.com/microsoft/unilm/tree/master/beit3)\]

### 3. Clip

- CLIP: **Learning Transferable Visual Models From Natural Language Supervision**, _Radford et al._, ICML 2021. \[[paper](https://arxiv.org/abs/2103.00020)\]\[[code](https://github.com/OpenAI/CLIP)\]
- DALL-E2: **Hierarchical Text-Conditional Image Generation with CLIP Latents**, _Ramesh et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2204.06125)\]\[[code](https://github.com/lucidrains/DALLE2-pytorch)\]
- **HiCLIP: Contrastive Language-Image Pretraining with Hierarchy-aware Attention**, _Geng et al._, ICLR 2023. \[[paper](https://arxiv.org/abs/2303.02995)\]\[[code](https://github.com/jeykigung/HiCLIP)\]
- **Chinese CLIP: Contrastive Vision-Language Pretraining in Chinese**, _Yang et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2211.01335)\]\[[code](https://github.com/OFA-Sys/Chinese-CLIP)\]
- **Alpha-CLIP: A CLIP Model Focusing on Wherever You Want**, _Sun et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.03818)\]\[[code](https://github.com/SunzeY/AlphaCLIP)\]

### 4. Diffusion Model

- **Denoising Diffusion Probabilistic Models**，_Ho et al._, NeurIPS 2020. \[[paper](https://arxiv.org/abs/2006.11239)\]\[[code](https://github.com/hojonathanho/diffusion)\]\[[Pytorch Implementation](https://github.com/lucidrains/denoising-diffusion-pytorch)\]
- **Improved Denoising Diffusion Probabilistic Models**, _Nichol and Dhariwal_, ICML 2021. \[[paper](https://arxiv.org/abs/2102.09672)\]\[[code](https://github.com/openai/improved-diffusion)\]
- **Diffusion Models Beat GANs on Image Synthesis**, _Dhariwal and Nichol_, NeurIPS 2021. \[[paper](https://arxiv.org/abs/2105.05233)\]\[[code](https://github.com/openai/guided-diffusion)\]
- **Classifier-Free Diffusion Guidance**, _Ho and Salimans_, NeurIPS 2021. \[[paper](https://arxiv.org/abs/2207.12598)\]\[[code](https://github.com/lucidrains/classifier-free-guidance-pytorch)\]
- **GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models**, _Nichol et al._, arxiv 2021. \[[paper](https://arxiv.org/abs/2112.10741)\]\[[code](https://github.com/openai/glide-text2im)\]
- DALL-E2: **Hierarchical Text-Conditional Image Generation with CLIP Latents**, _Ramesh et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2204.06125)\]\[[code](https://github.com/lucidrains/DALLE2-pytorch)\]\[[dalle-mini](https://github.com/borisdayma/dalle-mini)\]
- Stable-Diffusion: **High-Resolution Image Synthesis with Latent Diffusion Models**, _Rombach et al._, CVPR 2022. \[[paper](https://arxiv.org/abs/2112.10752)\]\[[code](https://github.com/CompVis/latent-diffusion)\]
- **SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis**, _Podell et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2307.01952)\]\[[code](https://github.com/Stability-AI/generative-models)\]
- **SDXL-Turbo: Adversarial Diffusion Distillation**, _Sauer et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.17042)\]\[[code](https://github.com/Stability-AI/generative-models)\]
- LCM: **Latent Consistency Models_Synthesizing High-Resolution Images with Few-Step Inference**, _Luo et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.04378)\]\[[code](https://github.com/luosiallen/latent-consistency-model)\]
- **LCM-LoRA: A Universal Stable-Diffusion Acceleration Module**, _Luo et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.05556)\]\[[code](https://github.com/luosiallen/latent-consistency-model)\]
- **StreamDiffusion: A Pipeline-level Solution for Real-time Interactive Generation**, _Kodaira et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.12491)\]\[[code](https://github.com/cumulo-autumn/StreamDiffusion)\]
- **Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets**, _Blattmann et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.15127)\]\[[code](https://github.com/Stability-AI/generative-models)\]
- **Consistency Models**, _Song et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2303.01469)\]\[[code](https://github.com/openai/consistency_models)\]\[[Consistency Decoder](https://github.com/openai/consistencydecoder)\]
- **A Survey on Video Diffusion Models**, _Xing et al._, srxiv 2023. \[[paper](https://arxiv.org/abs/2310.10647)\]\[[code](https://github.com/ChenHsing/Awesome-Video-Diffusion-Models)\]
- **Diffusion Models: A Comprehensive Survey of Methods and Applications**, _Yang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2209.00796)\]\[[code](https://github.com/YangLing0818/Diffusion-Models-Papers-Survey-Taxonomy)\]
- **Language Model Beats Diffusion -- Tokenizer is Key to Visual Generation**, _Yu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.05737)\]
- **The Chosen One: Consistent Characters in Text-to-Image Diffusion Models**, _Avrahami et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.10093)\]\[[code](https://github.com/ZichengDuan/TheChosenOne)\]
- **UniDiffuser: One Transformer Fits All Distributions in Multi-Modal Diffusion**, _Bao et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2303.06555)\]\[[code](https://github.com/thu-ml/unidiffuser)\]

- **Github Repositories**
- \[[stable-diffusion](https://github.com/CompVis/stable-diffusion)\]
- \[[stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)\]
- \[[stablediffusion](https://github.com/Stability-AI/stablediffusion?tab=readme-ov-file)\]
- \[[Awesome-Diffusion-Models](https://github.com/diff-usion/Awesome-Diffusion-Models)\]
- \[[Fooocus](https://github.com/lllyasviel/Fooocus)\]
- \[[ComfyUI](https://github.com/comfyanonymous/ComfyUI)\]
- \[[diffusers](https://github.com/huggingface/diffusers)\]

### 5. Multimodal LLM

- **A Survey on Multimodal Large Language Models**, Yin et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2306.13549)\]\[[code](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models)\]
- **BuboGPT: Enabling Visual Grounding in Multi-Modal LLMs**, _Zhao et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2307.08581)\]\[[code](https://github.com/magic-research/bubogpt)\]

### 6. Text2Image

- DALL-E: **Zero-Shot Text-to-Image Generation**, _Ramesh et al._, arxiv 2021. \[[paper](https://arxiv.org/abs/2102.12092)\]\[[code](https://github.com/openai/DALL-E)\]
- DALL-E3: **Improving Image Generation with Better Captions**, _Betker et al._, OpenAI 2023. \[[paper](https://cdn.openai.com/papers/dall-e-3.pdf)\]\[[code](https://github.com/openai/consistencydecoder)\]\[[blog](https://openai.com/dall-e-3)\]
- ControlNet: **Adding Conditional Control to Text-to-Image Diffusion Models**, _Zhang et al._, ICCV 2023. \[[paper](https://arxiv.org/abs/2302.05543)\]\[[code](https://github.com/lllyasviel/ControlNet)\]
- **AnyText: Multilingual Visual Text Generation And Editing**, _Tuo et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.03054)\]\[[code](https://github.com/tyxsspa/AnyText)\]
- **LAION-5B: An open large-scale dataset for training next generation image-text models**, _Schuhmann et al._, NeurIPS 2022. \[[paper](https://arxiv.org/abs/2210.08402)\]\[[code](https://github.com/LAION-AI/laion-datasets)\]\[[blog](https://laion.ai/blog/laion-5b/)\]
- Imagen: **Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding**, _Saharia et al._, NeurIPS 2022. \[[paper](https://arxiv.org/abs/2205.11487)\]\[[code](https://github.com/lucidrains/imagen-pytorch)\]

### 7. Text2Video

- **Animate Anyone: Consistent and Controllable Image-to-Video Synthesis for Character Animation**, _Hu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.17117)\]\[[code](https://github.com/HumanAIGC/AnimateAnyone)\]
- **DreaMoving: A Human Video Generation Framework based on Diffusion Models**, _Feng et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.05107)\]\[[code](https://github.com/dreamoving/dreamoving-project)\]
- **MagicAnimate:Temporally Consistent Human Image Animation using Diffusion Model**, _Xu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.16498)\]\[[code](https://github.com/magic-research/magic-animate)\]
- **FlowVid: Taming Imperfect Optical Flows for Consistent Video-to-Video Synthesis**, _Liang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.17681)\]\[[code](https://github.com/Jeff-LiangF/FlowVid)\]

### 8. Survey for Multimodal

### 9. Other

---

## Reinforcement Learning

### 1.Basic for RL

- PPO: **Proximal Policy Optimization Algorithms**, _Schulman et al._, arxiv 2017. \[[paper](https://arxiv.org/abs/1707.06347)\]\[[code](https://github.com/openai/baselines)\]

### 2. LLM for decision making 

- **Decision Transformer_Reinforcement Learning via Sequence Modeling**, _Chen et al._, NeurIPS 2021. \[[paper](https://arxiv.org/abs/2106.01345)\]\[[code](https://github.com/kzl/decision-transformer)\]
- Trajectory Transformer: **Offline Reinforcement Learning as One Big Sequence Modeling Problem**, _Janner et al._, NeurIPS 2021. \[[paper](https://arxiv.org/abs/2106.02039)\]\[[code](https://github.com/JannerM/trajectory-transformer)\]
- **Guiding Pretraining in Reinforcement Learning with Large Language Models**, _Du et al._, ICML 2023. \[[paper](https://arxiv.org/abs/2302.06692)\]\[[code](https://github.com/yuqingd/ellm)\]
- **Introspective Tips: Large Language Model for In-Context Decision Making**, _Chen et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.11598)\]

---

## GNN

### Survey for GNN

---

## Transformer Architecture

- **Attention is All you Need**, _Vaswani et al._, NIPS 2017. \[[paper](https://arxiv.org/abs/1706.03762)\]\[[code](https://github.com/jadore801120/attention-is-all-you-need-pytorch)\]\[[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)\]
- RoPE: **RoFormer: Enhanced Transformer with Rotary Position Embedding**, _Su et al._, arxiv 2021. \[[paper](https://arxiv.org/abs/2104.09864)\]\[[code](https://github.com/ZhuiyiTechnology/roformer)\]\[[rotary-embedding-torch](https://github.com/lucidrains/rotary-embedding-torch)\]
- **RWKV: Reinventing RNNs for the Transformer Era**, _Peng et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.13048)\]\[[code](https://github.com/BlinkDL/RWKV-LM)\]
- **Mamba: Linear-Time Sequence Modeling with Selective State Spaces**, _Gu and Dao_, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.00752)\]\[[code](https://github.com/state-spaces/mamba)\]


