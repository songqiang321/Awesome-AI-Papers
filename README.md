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
    - [Chinese Model](#35-chinese-model)
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
  │      └─ Reasoning and Planning/       
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
<details>
<summary>NLP</summary>

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
- BPE: **Neural Machine Translation of Rare Words with Subword Units**, _Sennrich et al._, ACL 2016. \[[paper](https://arxiv.org/abs/1508.07909)\]\[[code](https://github.com/rsennrich/subword-nmt)\]
- Byte-Level BPE: **Neural Machine Translation with Byte-Level Subwords**, _Wang et al._, arxiv 2019. \[[paper](https://arxiv.org/abs/1909.03341)\]\[[code](https://github.com/facebookresearch/fairseq/tree/main/examples/byte_level_bpe)\]

### 2. Seq2Seq

- **Generating Sequences With Recurrent Neural Networks**, _Graves_, arxiv 2013. \[[paper](https://arxiv.org/abs/1308.0850)\]
- **Sequence to Sequence Learning with Neural Networks**, _Sutskever et al._, NeruIPS 2014. \[[paper](https://arxiv.org/abs/1409.3215)\]
- **Neural Machine Translation by Jointly Learning to Align and Translate**, _Bahdanau et al._, ICLR 2015. \[[paper](https://arxiv.org/abs/1409.0473)\]\[[code](https://github.com/lisa-groundhog/GroundHog)\]
- **On the Properties of Neural Machine Translation: Encoder-Decoder Approaches**, _Cho et al._, arxiv 2014. \[[paper](https://arxiv.org/abs/1409.1259)\]
- **Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation**, _Cho et al._, arxiv 2014. \[[paper](https://arxiv.org/abs/1406.1078)\]
- \[[fairseq](https://github.com/facebookresearch/fairseq)\]\[[pytorch-seq2seq](https://github.com/IBM/pytorch-seq2seq)\]

### 3. Pretraining

- **Attention Is All You Need**, _Vaswani et al._, NIPS 2017. \[[paper](https://arxiv.org/abs/1706.03762)\]\[[code](https://github.com/tensorflow/tensor2tensor)\]
- GPT: **Improving language understanding by generative pre-training**, _Radford et al._, preprint 2018.  \[[paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)\]\[[code](https://github.com/openai/finetune-transformer-lm)\]
- GPT-2: **Language Models are Unsupervised Multitask Learners**, _Radford et al._, OpenAI blog 2019. \[[paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)\]\[[code](https://github.com/openai/gpt-2)\]\[[llm.c](https://github.com/karpathy/llm.c)\]
- GPT-3: **Language Models are Few-Shot Learners**, _Brown et al._, NeurIPS 2020. \[[paper](https://arxiv.org/abs/2005.14165)\]\[[code](https://github.com/openai/gpt-3)\]\[[nanoGPT](https://github.com/karpathy/nanoGPT)\]\[[gpt-fast](https://github.com/pytorch-labs/gpt-fast)\]
- InstructGPT: **Training language models to follow instructions with human feedback**, _Ouyang et al._, NeurIPS 2022. \[[paper](https://arxiv.org/abs/2203.02155)\]\[[MOSS-RLHF](https://github.com/OpenLMLab/MOSS-RLHF)\]
- **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**, _Devlin et al._, arxiv 2018. \[[paper](https://arxiv.org/abs/1810.04805)\]\[[code](https://github.com/google-research/bert)\]\[[BERT-pytorch](https://github.com/codertimo/BERT-pytorch)\]
- **RoBERTa: A Robustly Optimized BERT Pretraining Approach**, _Liu et al._, arxiv 2019. \[[paper](https://arxiv.org/abs/1907.11692)\]\[[code](https://github.com/facebookresearch/fairseq)\]
- **What Does BERT Look At_An Analysis of BERT's Attention**, _Clark et al._, arxiv 2019. \[[paper](https://arxiv.org/abs/1906.04341)\]\[[code](https://github.com/clarkkev/attention-analysis)\]
- **DeBERTa: Decoding-enhanced BERT with Disentangled Attention**, _He et al._, ICLR 2021. \[[paper](https://arxiv.org/abs/2006.03654)\]\[[code](https://github.com/microsoft/DeBERTa)\]
- **DistilBERT: a distilled version of BERT_smaller, faster, cheaper and lighter** _Sanh et al._, arxiv 2019. \[[paper](https://arxiv.org/abs/1910.01108)\]\[[code](https://github.com/huggingface/transformers)\]
- **BERT Rediscovers the Classical NLP Pipeline**, _Tenney et al._, arxiv 2019. \[[paper](https://arxiv.org/abs/1905.05950)\]\[[code](https://github.com/nyu-mll/jiant)\]
- **How to Fine-Tune BERT for Text Classification?**, _Sun et al._, arxiv 2019. \[[paper](https://arxiv.org/abs/1905.05583)\]\[[code](https://github.com/xuyige/BERT4doc-Classification)\]
- **TinyStories: How Small Can Language Models Be and Still Speak Coherent English**, _Eldan and Li_, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.07759)\]\[[code]\]\[[phi-2](https://huggingface.co/microsoft/phi-2)\]
- \[[llm-course](https://github.com/mlabonne/llm-course)\]\[[intro-llm](https://intro-llm.github.io/)\]\[[llm-cookbook](https://github.com/datawhalechina/llm-cookbook)\]\[[hugging-llm](https://github.com/datawhalechina/hugging-llm)\]\[[generative-ai-for-beginners](https://github.com/microsoft/generative-ai-for-beginners)\]\[[awesome-generative-ai-guide](https://github.com/aishwaryanr/awesome-generative-ai-guide)\]\[[LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)\]\[[llm-action](https://github.com/liguodongiot/llm-action)\]
- \[[tokenizer_summary](https://huggingface.co/docs/transformers/tokenizer_summary)\]\[[minbpe](https://github.com/karpathy/minbpe)\]\[[tokenizers](https://github.com/huggingface/tokenizers)\]\[[tiktoken](https://github.com/openai/tiktoken)\]\[[SentencePiece](https://github.com/google/sentencepiece)\]

#### 3.1 Large Language Model

- **A Survey of Large Language Models**, _Zhao etal._, arxiv 2023. \[[paper](https://arxiv.org/abs/2303.18223)\]\[[code](https://github.com/RUCAIBox/LLMSurvey)\]\[[LLMBox](https://github.com/RUCAIBox/LLMBox)\]\[[LLMBook-zh](https://llmbook-zh.github.io/)\]\[[LLMsPracticalGuide](https://github.com/Mooler0410/LLMsPracticalGuide)\]
- **Efficient Large Language Models: A Survey**, _Wan et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.03863)\]\[[code](https://github.com/AIoT-MLSys-Lab/Efficient-LLMs-Survey)\]
- **Challenges and Applications of Large Language Models**, _Kaddour et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2307.10169)\]
- **A Comprehensive Survey on Pretrained Foundation Models: A History from BERT to ChatGPT**, _Zhou et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2302.09419)\]
- **From Google Gemini to OpenAI Q* (Q-Star): A Survey of Reshaping the Generative Artificial Intelligence (AI) Research Landscape**, _Mclntosh et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.10868)\]
- **A Survey of Resource-efficient LLM and Multimodal Foundation Models**, _Xu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.08092)\]\[[code](https://github.com/UbiquitousLearning/Efficient_Foundation_Model_Survey)\]
- **Large Language Models: A Survey**, _Minaee et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.06196)\]
- Anthropic: **Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback**, _Bai et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2204.05862)\]\[[code](https://github.com/anthropics/hh-rlhf)\]
- Anthropic: **Constitutional AI: Harmlessness from AI Feedback**, _Bai et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2212.08073)\]\[[code](https://github.com/anthropics/ConstitutionalHarmlessnessPaper)\]
- Anthropic: **Model Card and Evaluations for Claude Models**, Anthropic, 2023. \[[paper](https://www-files.anthropic.com/production/images/Model-Card-Claude-2.pdf)\]
- Anthropic: **The Claude 3 Model Family: Opus, Sonnet, Haiku**, Anthropic, 2024. \[[paper](https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf)\]
- **BLOOM_A 176B-Parameter Open-Access Multilingual Language Model**, _BigScience Workshop_, arxiv 2022. \[[paper](https://arxiv.org/abs/2211.05100)\]\[[code](https://github.com/bigscience-workshop)\]\[[model](https://huggingface.co/bigscience)\]
- **OPT: Open Pre-trained Transformer Language Models**, _Zhang et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2205.01068)\]\[[code](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT)\]
- Chinchilla: **Training Compute-Optimal Large Language Models**, _Hoffmann et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2203.15556)\]
- Gopher: **Scaling Language Models: Methods, Analysis & Insights from Training Gopher**, _Rae et al._, arxiv 2021. \[[paper](https://arxiv.org/abs/2112.11446)\]
- **GPT-NeoX-20B: An Open-Source Autoregressive Language Model**, _Black et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2204.06745)\]\[[code](https://github.com/EleutherAI/gpt-neox)\]
- **Gemini: A Family of Highly Capable Multimodal Models**, _Gemini Team, Google_, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.11805)\]\[[Gemini 1.0](https://storage.googleapis.com/deepmind-media/gemini/gemini_1_report.pdf)\]\[[Gemini 1.5](https://arxiv.org/abs/2403.05530)\]\[[Unofficial Implementation](https://github.com/kyegomez/Gemini)\]\[[MiniGemini](https://github.com/dvlab-research/MiniGemini)\]
- **Gemma: Open Models Based on Gemini Research and Technology**, _Google DeepMind_, 2024. \[[paper](https://arxiv.org/abs/2403.08295)\]\[[code](https://github.com/google/gemma_pytorch)\]\[[google-deepmind/gemma](https://github.com/google-deepmind/gemma)\]\[[gemma.cpp](https://github.com/google/gemma.cpp)\]\[[model](https://ai.google.dev/gemma)\]
- **GPT-4 Technical Report**, _OpenAI_, arxiv 2023. \[[paper](https://arxiv.org/abs/2303.08774)\]
- **GPT-4V(ision) System Card**, _OpenAI_, OpenAI blog 2023. \[[paper](https://openai.com/research/gpt-4v-system-card)\]
- **Sparks of Artificial General Intelligence_Early experiments with GPT-4**, _Bubeck et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2303.12712)\]
- **The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)**, _Yang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.17421)\]\[[guidance](https://github.com/guidance-ai/guidance)\]
- **LaMDA: Language Models for Dialog Applications**, _Thoppilan et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2201.08239)\]\[[LaMDA-rlhf-pytorch](https://github.com/conceptofmind/LaMDA-rlhf-pytorch)\]
- **LLaMA: Open and Efficient Foundation Language Models**, _Touvron et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2302.13971)\]\[[code](https://github.com/facebookresearch/llama/tree/llama_v1)\]\[[llama.cpp](https://github.com/ggerganov/llama.cpp)\]\[[ollama](https://github.com/jmorganca/ollama)\]\[[llamafile](https://github.com/Mozilla-Ocho/llamafile)\]
- **Llama 2: Open Foundation and Fine-Tuned Chat Models**, _Touvron et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2307.09288)\]\[[code](https://github.com/meta-llama/llama)\]\[[llama-recipes](https://github.com/meta-llama/llama-recipes)\]\[[llama2.c](https://github.com/karpathy/llama2.c)\]\[[lit-llama](https://github.com/Lightning-AI/lit-llama)\]\[[litgpt](https://github.com/Lightning-AI/litgpt)\]
- \[[llama3](https://github.com/meta-llama/llama3)\]
- **TinyLlama: An Open-Source Small Language Model**, _Zhang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.02385)\]\[[code](https://github.com/jzhang38/TinyLlama)\]\[[LiteLlama](https://huggingface.co/ahxt/LiteLlama-460M-1T)\]\[[MobiLlama](https://github.com/mbzuai-oryx/MobiLlama)\]
- **Stanford Alpaca: An Instruction-following LLaMA Model**, _Taori et al._, Stanford blog 2023. \[[paper](https://crfm.stanford.edu/2023/03/13/alpaca.html)\]\[[code](https://github.com/tatsu-lab/stanford_alpaca)\]\[[Alpaca-Lora](https://github.com/tloen/alpaca-lora)\]
- **Mistral 7B**, _Jiang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.06825)\]\[[code](https://github.com/mistralai/mistral-src)\]\[[model](https://huggingface.co/mistralai)\]
- **OLMo: Accelerating the Science of Language Models**, _Groeneveld et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.00838)\]\[[code](https://github.com/allenai/OLMo)\]\[[Dolma Dataset](https://github.com/allenai/dolma)\]
- Minerva: **Solving Quantitative Reasoning Problems with Language Models**, _Lewkowycz et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2206.14858)\]
- **PaLM: Scaling Language Modeling with Pathways**, _Chowdhery et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2204.02311)\]\[[PaLM-pytorch](https://github.com/lucidrains/PaLM-pytorch)\]\[[PaLM-rlhf-pytorch](https://github.com/lucidrains/PaLM-rlhf-pytorch)\]\[[PaLM](https://github.com/conceptofmind/PaLM)\]
- **PaLM 2 Technical Report**, _Anil et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.10403)\]
- **PaLM-E: An Embodied Multimodal Language Model**, _Driess et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2303.03378)\]\[[code](https://github.com/kyegomez/PALM-E)\]
- T5: **Exploring the limits of transfer learning with a unified text-to-text transformer**, _Raffel et al._, Journal of Machine Learning Research 2023. \[[paper](https://arxiv.org/abs/1910.10683)\]\[[code](https://github.com/google-research/text-to-text-transfer-transformer)\]\[[t5-pytorch](https://github.com/conceptofmind/t5-pytorch)\]
- **BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension**, _Lewis et al._, ACL 2020. \[[paper](https://arxiv.org/abs/1910.13461)\]\[[code](https://github.com/facebookresearch/fairseq/tree/main/examples/bart)\]
- FLAN: **Finetuned Language Models Are Zero-Shot Learners**, _Wei et al._, ICLR 2022. \[[paper](https://arxiv.org/abs/2109.01652)\]\[[code](https://github.com/google-research/flan)\]
- Scaling Flan: **Scaling Instruction-Finetuned Language Models**, _Chung et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2210.11416)\]\[[mode](https://github.com/google-research/t5x/blob/main/docs/models.md#flan-t5-checkpoints)\]
- **Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context**, _Dai et al._, ACL 2019. \[[paper](https://arxiv.org/abs/1901.02860)\]\[[code](https://github.com/kimiyoung/transformer-xl)\]
- **XLNet: Generalized Autoregressive Pretraining for Language Understanding**, _Yang et al._, NeurIPS 2019. \[[paper](https://arxiv.org/abs/1906.08237)\]\[[code](https://github.com/zihangdai/xlnet)\]
- **Visual ChatGPT: Talking, Drawing and Editing with Visual Foundation Models**, _Wu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2303.04671)\]\[[code](https://github.com/moymix/TaskMatrix)\]
- **WebGPT: Browser-assisted question-answering with human feedback**, _Nakano et al._, arxiv 2021. \[[paper](https://arxiv.org/abs/2112.09332)\]
- **Open Release of Grok-1**, _xAI_, 2024. \[[blog](https://x.ai/blog/grok-os)\]\[[code](https://github.com/xai-org/grok-1)\]\[[model](https://huggingface.co/xai-org/grok-1)\]\[[modelscope](https://modelscope.cn/models/AI-ModelScope/grok-1/summary)\]\[[hpcai-tech/grok-1](https://huggingface.co/hpcai-tech/grok-1)\]\[[dbrx](https://github.com/databricks/dbrx)\]\[[Command R+](https://huggingface.co/CohereForAI/c4ai-command-r-plus)\]\[[snowflake-arctic](https://github.com/Snowflake-Labs/snowflake-arctic)\]

#### 3.2 LLM Application

- **A Watermark for Large Language Models**, _Kirchenbauer et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2301.10226)\]\[[code](https://github.com/jwkirchenbauer/lm-watermarking)\]
- **SeqXGPT: Sentence-Level AI-Generated Text Detection**, _Wang et al._, EMNLP 2023. \[[paper](https://arxiv.org/abs/2310.08903)\]\[[code](https://github.com/Jihuai-wpy/SeqXGPT)\]\[[llm-detect-ai](https://github.com/yanqiangmiffy/llm-detect-ai)\]\[[detect-gpt](https://github.com/eric-mitchell/detect-gpt)\]\[[fast-detect-gpt](https://github.com/baoguangsheng/fast-detect-gpt)\]
- **AlpaGasus: Training A Better Alpaca with Fewer Data**, _Chen et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2307.08701)\]\[[code](https://github.com/gpt4life/alpagasus)\]
- **AutoMix: Automatically Mixing Language Models**, _Madaan et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.12963)\]\[[code](https://github.com/automix-llm/automix)\]
- **ChipNeMo: Domain-Adapted LLMs for Chip Design**, _Liu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.00176)\]
- **GAIA: A Benchmark for General AI Assistants**, _Mialon et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.12983)\]\[[code](https://huggingface.co/gaia-benchmark)\]
- **HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face**, _Shen et al._, NeurIPS 2023. \[[paper](https://arxiv.org/abs/2303.17580)\]\[[code](https://github.com/microsoft/JARVIS)\]
- **MemGPT: Towards LLMs as Operating Systems**, _Packer et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.08560)\]\[[code](https://github.com/cpacker/MemGPT)\]
- **UFO: A UI-Focused Agent for Windows OS Interaction**, _Zhang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.07939)\]\[[code](https://github.com/microsoft/UFO)\]
- **OS-Copilot: Towards Generalist Computer Agents with Self-Improvement**, _Wu et al._, ICLR 2024. \[[paper](https://arxiv.org/abs/2402.07456)\]\[[code](https://github.com/OS-Copilot/FRIDAY)\]
- **AIOS: LLM Agent Operating System**, _Mei et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.16971)\]\[[code](https://github.com/agiresearch/AIOS)\]
- **DB-GPT: Empowering Database Interactions with Private Large Language Models**, _Xue et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.17449)\]\[[code](https://github.com/eosphoros-ai/DB-GPT)\]\[[DocsGPT](https://github.com/arc53/DocsGPT)\]\[[privateGPT](https://github.com/imartinez/privateGPT)\]\[[localGPT](https://github.com/PromtEngineer/localGPT)\]
- **OpenChat: Advancing Open-source Language Models with Mixed-Quality Data**, _Wang et al._, ICLR 2024. \[[paper](https://arxiv.org/abs/2309.11235)\]\[[code](https://github.com/imoneoi/openchat)\]
- **OpenCodeInterpreter: Integrating Code Generation with Execution and Refinement**, _Zheng et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.14658)\]\[[code](https://github.com/OpenCodeInterpreter/OpenCodeInterpreter)\]
- **Orca: Progressive Learning from Complex Explanation Traces of GPT-4**, _Mukherjee et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2306.02707)\]
- **PDFTriage: Question Answering over Long, Structured Documents**, _Saad-Falcon et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.08872)\]\[[code]\]
- **Prompt2Model: Generating Deployable Models from Natural Language Instructions**, _Viswanathan et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2308.12261)\]\[[code](https://github.com/neulab/prompt2model)\]
- **Shepherd: A Critic for Language Model Generation**, _Wang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2308.04592)\]\[[code](https://github.com/facebookresearch/Shepherd)\]
- **Alpaca: A Strong, Replicable Instruction-Following Model**, _Taori et al._, Stanford Blog 2023. \[[paper](https://crfm.stanford.edu/2023/03/13/alpaca.html)\]\[[code](https://github.com/tatsu-lab/stanford_alpaca)\]
- **Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90\%* ChatGPT Quality**, _Chiang et al._, 2023. \[[blog](https://lmsys.org/blog/2023-03-30-vicuna/)\]
- **WizardLM: Empowering Large Language Models to Follow Complex Instructions**, _Xu et al._, ICLR 2024. \[[paper](https://arxiv.org/abs/2304.12244)\]\[[code](https://github.com/nlpxucan/WizardLM)\]
- **WebGLM: Towards An Efficient Web-Enhanced Question Answering System with Human Preferences**, _Liu et al._, KDD 2023. \[[paper](https://arxiv.org/abs/2306.07906)\]\[[code](https://github.com/THUDM/WebGLM)\]\[[AutoWebGLM](https://github.com/THUDM/AutoWebGLM)\]\[[AutoCrawler](https://github.com/EZ-hwh/AutoCrawler)\]\[[gpt-crawler](https://github.com/BuilderIO/gpt-crawler)\]\[[webllama](https://github.com/McGill-NLP/webllama)\]\[[gpt-researcher](https://github.com/assafelovic/gpt-researcher)\]\[[skyvern](https://github.com/Skyvern-AI/skyvern)\]
- **LLM4Decompile: Decompiling Binary Code with Large Language Models**, _Tan et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.05286)\] \[[code](https://github.com/albertan017/LLM4Decompile)\]

- \[[ray](https://github.com/ray-project/ray)\]\[[dask](https://github.com/dask/dask)\]\[[TaskingAI](https://github.com/TaskingAI/TaskingAI)\]\[[gpt4all](https://github.com/nomic-ai/gpt4all)\]\[[ollama](https://github.com/jmorganca/ollama)\]\[[llama.cpp](https://github.com/ggerganov/llama.cpp)\]\[[dify](https://github.com/langgenius/dify)\]\[[bisheng](https://github.com/dataelement/bisheng)\]

##### 3.2.1 AI Agent

- **LLM Powered Autonomous Agents**, _Lilian Weng_, 2023. \[[blog](https://lilianweng.github.io/posts/2023-06-23-agent/)\]\[[LLMAgentPapers](https://github.com/zjunlp/LLMAgentPapers)\]\[[LLM-Agents-Papers](https://github.com/AGI-Edgerunners/LLM-Agents-Papers)\]\[[awesome-language-agents](https://github.com/ysymyth/awesome-language-agents)\]\[[Awesome-Papers-Autonomous-Agent](https://github.com/lafmdp/Awesome-Papers-Autonomous-Agent)\]
- **A Survey on Large Language Model based Autonomous Agents**, _Wang et al._, \[[paper](https://arxiv.org/abs/2308.11432)\]\[[code](https://github.com/Paitesanshi/LLM-Agent-Survey)\]
- **The Rise and Potential of Large Language Model Based Agents: A Survey**, _Xi et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.07864)\]\[[code](https://github.com/WooooDyy/LLM-Agent-Paper-List)\]
- **Agent AI: Surveying the Horizons of Multimodal Interaction**, _Durante et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.03568)\]
- **Position Paper: Agent AI Towards a Holistic Intelligence**, _Huang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.00833)\]

- **AgentBench: Evaluating LLMs as Agents**, _Liu et al._, ICLR 2024. \[[paper](https://arxiv.org/abs/2308.03688)\]\[[code](https://github.com/THUDM/AgentBench)\]\[[OSWorld](https://github.com/xlang-ai/OSWorld)\]
- **Agents: An Open-source Framework for Autonomous Language Agents**, _Zhou et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.07870)\]\[[code](https://github.com/aiwaves-cn/agents)\]
- **AutoAgents: A Framework for Automatic Agent Generation**, _Chen et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.17288)\]\[[code](https://github.com/Link-AGI/AutoAgents)\]
- **AgentTuning: Enabling Generalized Agent Abilities for LLMs**, _Zeng et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.12823)\]\[[code](https://github.com/THUDM/AgentTuning)\]
- **AgentVerse: Facilitating Multi-Agent Collaboration and Exploring Emergent Behaviors**, _Chen et al._, ICLR 2024. \[[paper](https://arxiv.org/abs/2308.10848)\]\[[code](https://github.com/OpenBMB/AgentVerse/)\]
- **AppAgent: Multimodal Agents as Smartphone Users**, _Zhang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.13771)\]\[[code](https://github.com/mnotgod96/AppAgent)\]
- **Mobile-Agent: Autonomous Multi-Modal Mobile Device Agent with Visual Perception**, _Wang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.16158)\]\[[code](https://github.com/X-PLUG/MobileAgent)\]
- **Personal LLM Agents: Insights and Survey about the Capability, Efficiency and Security**, _Li et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.05459)\]\[[code](https://github.com/MobileLLM/Personal_LLM_Agents_Survey)\]
- **AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation**, _Wu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2308.08155)\]\[[code](https://github.com/microsoft/autogen)\]
- **CAMEL: Communicative Agents for "Mind" Exploration of Large Language Model Society**, _Li et al._, NeurIPS 2023. \[[paper](https://arxiv.org/abs/2303.17760)\]\[[code](https://github.com/camel-ai/camel)\]
- ChatDev: **Communicative Agents for Software Development**, _Qian et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2307.07924)\]\[[code](https://github.com/OpenBMB/ChatDev)\]\[[gpt-pilot](https://github.com/Pythagora-io/gpt-pilot)\]
- **MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework**, _Hong et al._, ICLR 2024 Oral. \[[paper](https://arxiv.org/abs/2308.00352)\]\[[code](https://github.com/geekan/MetaGPT)\]
- **RepoAgent: An LLM-Powered Open-Source Framework for Repository-level Code Documentation Generation**, _Luo et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.16667)\]\[[code](https://github.com/OpenBMB/RepoAgent)\]
- **Generative Agents: Interactive Simulacra of Human Behavior**, _Park et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2304.03442)\]\[[code](https://github.com/joonspk-research/generative_agents)\]\[[GPTeam](https://github.com/101dotxyz/GPTeam)\]
- **CogAgent: A Visual Language Model for GUI Agents**, _Hong et al._, CVPR 2024. \[[paper](https://arxiv.org/abs/2312.08914)\]\[[code](https://github.com/THUDM/CogVLM)\]
- **OpenAgents: An Open Platform for Language Agents in the Wild**, _Xie et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.10634)\]\[[code](https://github.com/xlang-ai/OpenAgents)\]
- **Reflexion: Language Agents with Verbal Reinforcement Learning**, _Shinn et al._, NeurIPS 2023. \[[paper](https://arxiv.org/abs/2303.11366)\]\[[code](https://github.com/noahshinn/reflexion)\]
- **TaskWeaver: A Code-First Agent Framework**, _Qiao et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.17541)\]\[[code](https://github.com/microsoft/TaskWeaver)\]
- **MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge**, _Fan et al._, NeurIPS 2022 Outstanding Paper. \[[paper](https://arxiv.org/abs/2206.08853)\]\[[code](https://github.com/MineDojo/MineDojo)\]
- **Voyager: An Open-Ended Embodied Agent with Large Language Models**, _Wang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.16291)\]\[[code](https://github.com/MineDojo/Voyager)\]
- **Eureka: Human-Level Reward Design via Coding Large Language Models**, _Ma et al._, ICLR 2024. \[[paper](https://arxiv.org/abs/2310.12931)\]\[[code](https://github.com/eureka-research/Eureka)\]\[[DrEureka](https://github.com/eureka-research/DrEureka)\]

- **Mind2Web: Towards a Generalist Agent for the Web**, _Deng et al._, NeurIPS 2023. \[[paper](https://arxiv.org/abs/2306.06070)\]\[[code](https://github.com/OSU-NLP-Group/Mind2Web)\]\[[AutoWebGLM](https://github.com/THUDM/AutoWebGLM)\]
- SeeAct: **GPT-4V(ision) is a Generalist Web Agent, if Grounded**, _Zheng et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.01614)\]\[[code](https://github.com/OSU-NLP-Group/SeeAct)\]

- **Foundation Models in Robotics: Applications, Challenges, and the Future**, _Firoozi et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.07843)\]\[[code](https://github.com/robotics-survey/Awesome-Robotics-Foundation-Models)\]
- **RT-1: Robotics Transformer for Real-World Control at Scale**, _Brohan et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2212.06817)\]\[[code](https://github.com/google-research/robotics_transformer)\]
- **RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control**, _Brohan et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2307.15818)\]\[[Unofficial Implementation](https://github.com/kyegomez/RT-2)\]\[[RT-H: Action Hierarchies Using Language](https://arxiv.org/abs/2403.01823)\]
- **Open X-Embodiment: Robotic Learning Datasets and RT-X Models**, _Open X-Embodiment Collaboration_, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.08864)\]\[[code](https://github.com/google-deepmind/open_x_embodiment)\]
- **Shaping the future of advanced robotics**, Google DeepMind 2024. \[[blog](https://deepmind.google/discover/blog/shaping-the-future-of-advanced-robotics/)\]
- **RoboGen: Towards Unleashing Infinite Data for Automated Robot Learning via Generative Simulation**, _Wang et al._, ICML 2024. \[[paper](https://arxiv.org/abs/2311.01455)\]\[[code](https://github.com/Genesis-Embodied-AI/RoboGen)\]
- **RL-GPT: Integrating Reinforcement Learning and Code-as-policy**, _Liu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.19299)\]
- **Genie: Generative Interactive Environments**, _Bruce et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.15391)\]
- **Mobile ALOHA: Learning Bimanual Mobile Manipulation with Low-Cost Whole-Body Teleoperation**, _Fu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.02117)\]\[[code](https://github.com/MarkFzp/mobile-aloha)\]\[[Hardware Code](https://github.com/MarkFzp/mobile-aloha)\]\[[Learning Code](https://github.com/MarkFzp/act-plus-plus)\]\[[UMI](https://github.com/real-stanford/universal_manipulation_interface)\]\[[LeRobot](https://github.com/huggingface/lerobot)\]

- \[[awesome-ai-agents](https://github.com/e2b-dev/awesome-ai-agents)\]
- \[[AutoGPT](https://github.com/Significant-Gravitas/AutoGPT)\]\[[GPT-Engineer](https://github.com/gpt-engineer-org/gpt-engineer)\]\[[AgentGPT](https://github.com/reworkd/AgentGPT)\]
- \[[BabyAGI](https://github.com/yoheinakajima/babyagi)\]\[[SuperAGI](https://github.com/TransformerOptimus/SuperAGI)\]\[[OpenAGI](https://github.com/agiresearch/OpenAGI)\]
- \[[open-interpreter](https://github.com/KillianLucas/open-interpreter)\]\[[Homepage](https://openinterpreter.com/)\]\[[rawdog](https://github.com/AbanteAI/rawdog)\]\[[OpenCodeInterpreter](https://github.com/OpenCodeInterpreter/OpenCodeInterpreter)\]
- **XAgent: An Autonomous Agent for Complex Task Solving**, \[[blog](https://blog.x-agent.net/blog/xagent/)\]\[[code](https://github.com/OpenBMB/XAgent)\]
- \[[crewAI](https://github.com/joaomdmoura/crewAI)\]

##### 3.2.2 Academic

- **Galactica: A Large Language Model for Science**, _Taylor et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2211.09085)\]\[[code](https://github.com/paperswithcode/galai)\]
- **K2: A Foundation Language Model for Geoscience Knowledge Understanding and Utilization**, _Deng et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2306.05064)\]\[[code](https://github.com/davendw49/k2)\]\[[pdf_parser](https://github.com/Acemap/pdf_parser)\]
- **GeoGalactica: A Scientific Large Language Model in Geoscience**, _Lin et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.00434)\]\[[code](https://github.com/geobrain-ai/geogalactica)\]\[[sciparser](https://github.com/davendw49/sciparser)\]
- **Scientific Large Language Models: A Survey on Biological & Chemical Domains**, _Zhang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.14656)\]\[[code](https://github.com/HICAI-ZJU/Scientific-LLM-Survey)\]
- **SciGLM: Training Scientific Language Models with Self-Reflective Instruction Annotation and Tuning**, _Zhang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.07950)\]\[[code](https://github.com/THUDM/SciGLM)\]
- **ChemLLM: A Chemical Large Language Model**, _Zhang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.06852)\]\[[model](https://huggingface.co/AI4Chem/ChemLLM-7B-Chat)\]
- \[[Awesome-Scientific-Language-Models](https://github.com/yuzhimanhua/Awesome-Scientific-Language-Models)\]\[[gpt_academic](https://github.com/binary-husky/gpt_academic)\]\[[ChatPaper](https://github.com/kaixindelele/ChatPaper)\]

##### 3.2.3 Code

- **Neural code generation**, CMU 2024 Spring. \[[link](https://cmu-codegen.github.io/s2024/)\]
- **Unifying the Perspectives of NLP and Software Engineering: A Survey on Language Models for Code**, _Zhang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.07989)\]\[[Awesome-Code-LLM](https://github.com/codefuse-ai/Awesome-Code-LLM)\]\[[MFTCoder](https://github.com/codefuse-ai/MFTCoder)\]
- **Source Code Data Augmentation for Deep Learning: A Survey**, _Zhuo et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.19915)\]\[[code](https://github.com/terryyz/DataAug4Code)\]

- Codex: **Evaluating Large Language Models Trained on Code**, _Chen et al._, arxiv 2021. \[[paper](https://arxiv.org/abs/2107.03374)\]\[[dataset](https://github.com/openai/human-eval)\]
- **Code Llama: Open Foundation Models for Code**, _Rozière et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2308.12950)\]\[[code](https://github.com/meta-llama/codellama)\]\[[model](https://huggingface.co/codellama)\]
- **CodeGemma: Open Code Models Based on Gemma**, \[[blog](https://huggingface.co/blog/codegemma)\]\[[report](https://storage.googleapis.com/deepmind-media/gemma/codegemma_report.pdf)\]
- AlphaCode: **Competition-Level Code Generation with AlphaCode**, _Li et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2203.07814)\]\[[dataset](https://github.com/google-deepmind/code_contests)\]\[[AlphaCode2_Tech_Report](https://storage.googleapis.com/deepmind-media/AlphaCode2/AlphaCode2_Tech_Report.pdf)\]
- **CodeGeeX: A Pre-Trained Model for Code Generation with Multilingual Evaluations on HumanEval-X**, _Zheng et al._, KDD 2023. \[[paper](https://arxiv.org/abs/2303.17568)\]\[[code](https://github.com/THUDM/CodeGeeX)\]\[[CodeGeeX2](https://github.com/THUDM/CodeGeeX2)\]
- **CodeGen: An Open Large Language Model for Code with Multi-Turn Program Synthesis**, _Nijkamp et al._, ICLR 2022. \[[paper](https://arxiv.org/abs/2203.13474)\]\[[code](https://github.com/salesforce/CodeGen)\]
- **CodeGen2: Lessons for Training LLMs on Programming and Natural Languages**, _Nijkamp et al._, ICLR 2023. \[[paper](https://arxiv.org/abs/2305.02309)\]\[[code](https://github.com/salesforce/CodeGen2)\]
- **CodeChain: Towards Modular Code Generation Through Chain of Self-revisions with Representative Sub-modules**, _Le et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.08992)\]\[[code](https://github.com/SalesforceAIResearch/CodeChain)\]
- **StarCoder: may the source be with you**, _Li et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.06161)\]\[[code](https://github.com/bigcode-project/starcoder)\]\[[bigcode-project](https://github.com/bigcode-project)\]\[[model](https://huggingface.co/bigcode)\]
- **StarCoder 2 and The Stack v2: The Next Generation**, _Lozhkov et al._, 2024. \[[paper](https://arxiv.org/abs/2402.19173)\]\[[code](https://github.com/bigcode-project/starcoder2)\]\[[starcoder.cpp](https://github.com/bigcode-project/starcoder.cpp)\]
- **WizardCoder: Empowering Code Large Language Models with Evol-Instruct**, _Luo et al._, ICLR 2024. \[[paper](https://arxiv.org/abs/2306.08568)\]\[[code](https://github.com/nlpxucan/WizardLM/tree/main/WizardCoder)\]
- **Magicoder: Source Code Is All You Need**, _Wei et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.02120)\]\[[code](https://github.com/ise-uiuc/magicoder)\]
- **Code Generation with AlphaCodium: From Prompt Engineering to Flow Engineering**, _Ridnik et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.08500)\]\[[code](https://github.com/Codium-ai/AlphaCodium)\]
- **DeepSeek-Coder: When the Large Language Model Meets Programming -- The Rise of Code Intelligence**, _Guo et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.14196)\]\[[code](https://github.com/deepseek-ai/DeepSeek-Coder)\]
- **If LLM Is the Wizard, Then Code Is the Wand: A Survey on How Code Empowers Large Language Models to Serve as Intelligent Agents**, _Yang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.00812)\]
- **Design2Code: How Far Are We From Automating Front-End Engineering?**, _Si et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.03163)\]\[[code](https://github.com/NoviScl/Design2Code)\]

- \[[CodeQwen1.5](https://github.com/QwenLM/CodeQwen1.5)\]\[[aiXcoder-7B](https://github.com/aixcoder-plugin/aiXcoder-7B)\]
- \[[OpenDevin](https://github.com/OpenDevin/OpenDevin)\]\[[swe-bench-technical-report](https://www.cognition-labs.com/post/swe-bench-technical-report)\]\[[devika](https://github.com/stitionai/devika)\]\[[SWE-agent](https://github.com/princeton-nlp/SWE-agent)\]\[[auto-code-rover](https://github.com/nus-apr/auto-code-rover)\]\[[developer](https://github.com/smol-ai/developer)\]

##### 3.2.4 Financial Application

- **DocLLM: A layout-aware generative language model for multimodal document understanding**, _Wang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.00908)\]
- **DocGraphLM: Documental Graph Language Model for Information Extraction**, _Wang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2401.02823)\]
- **FinBERT: A Pretrained Language Model for Financial Communications**, _Yang et al._, arxiv 2020. \[[paper](https://arxiv.org/abs/2006.08097)\]\[[Wiley paper](https://onlinelibrary.wiley.com/doi/full/10.1111/1911-3846.12832)\]\[[code](https://github.com/yya518/FinBERT)\]\[[finBERT](https://github.com/ProsusAI/finBERT)\]\[[valuesimplex/FinBERT](https://github.com/valuesimplex/FinBERT)\]
- **FinGPT: Open-Source Financial Large Language Models**, _Yang et al._, IJCAI 2023. \[[paper](https://arxiv.org/abs/2306.06031)\]\[[code](https://github.com/AI4Finance-Foundation/FinGPT)\]
- **FinGPT: Instruction Tuning Benchmark for Open-Source Large Language Models in Financial Datasets**, _Wang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.04793)\]\[[code](https://github.com/AI4Finance-Foundation/FinGPT)\]
- **Instruct-FinGPT: Financial Sentiment Analysis by Instruction Tuning of General-Purpose Large Language Models**, _Zhang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2306.12659)\]\[[code](https://github.com/AI4Finance-Foundation/FinGPT/tree/master/fingpt/FinGPT_RAG/instruct-FinGPT)\]
- **FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance**, _Liu et al._, arxiv 2020. \[[paper](https://arxiv.org/abs/2011.09607)\]\[[code](https://github.com/AI4Finance-Foundation/FinRL)\]
- **FinRL-Meta: Market Environments and Benchmarks for Data-Driven Financial Reinforcement Learning**, _Liu et al._, NeurIPS 2022. \[[paper](https://arxiv.org/abs/2211.03107)\]\[[code](https://github.com/AI4Finance-Foundation/FinRL-Meta)\]
- **DISC-FinLLM: A Chinese Financial Large Language Model based on Multiple Experts Fine-tuning**, _Chen et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.15205)\]\[[code](https://github.com/FudanDISC/DISC-FinLLM)\]
- **A Multimodal Foundation Agent for Financial Trading: Tool-Augmented, Diversified, and Generalist**, _Zhang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.18485)\]
- **XuanYuan 2.0: A Large Chinese Financial Chat Model with Hundreds of Billions Parameters**, _Zhang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.12002)\]\[[code](https://github.com/Duxiaoman-DI/XuanYuan)\]\[[PIXIU](https://github.com/The-FinAI/PIXIU)\]
- **StructGPT: A General Framework for Large Language Model to Reason over Structured Data**, _Jiang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.09645)\]\[[code](https://github.com/RUCAIBox/StructGPT)\]
- **Large Language Model for Table Processing: A Survey**, _Lu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.05121)\]\[[llm-table-survey](https://github.com/godaai/llm-table-survey)\]\[[table-transformer](https://github.com/microsoft/table-transformer)\]
- **A Survey of Large Language Models in Finance (FinLLMs)**, _Lee et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.02315)\]\[[code](https://github.com/adlnlp/FinLLMs)\]
- **Data-Copilot: Bridging Billions of Data and Humans with Autonomous Workflow**, _Zhang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2306.07209)\]\[[code](https://github.com/zwq2018/Data-Copilot)\]
- **Data Interpreter: An LLM Agent For Data Science**, _Hong et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.18679)\]\[[code](https://github.com/geekan/MetaGPT/tree/main/examples/di)\]
- \[[gpt-investor](https://github.com/mshumer/gpt-investor)\]\[[FinGLM](https://github.com/MetaGLM/FinGLM)\]

##### 3.2.5 Information Retrieval

- **ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT**, _Khattab et al._, SIGIR 2020. \[[paper](https://arxiv.org/abs/2004.12832)\]
- **ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction**, _Santhanam et al._, NAACL 2022. \[[paper](https://arxiv.org/abs/2112.01488)\]\[[code](https://github.com/stanford-futuredata/ColBERT)\]\[[RAGatouille](https://github.com/bclavie/RAGatouille)\]
- **ColBERT-XM: A Modular Multi-Vector Representation Model for Zero-Shot Multilingual Information Retrieval**, _Louis et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.15059)\]\[[code](https://github.com/ant-louis/xm-retrievers)\]\[[model](https://huggingface.co/antoinelouis/colbert-xm)\]
- **Large Language Models for Information Retrieval: A Survey**, _Zhu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2308.07107)\]\[[code](https://github.com/RUC-NLPIR/LLM4IR-Survey)\]
- **Large Language Models for Generative Information Extraction: A Survey**, _Xu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.17617)\]\[[code](https://github.com/quqxui/Awesome-LLM4IE-Papers)\]\[[UIE](https://github.com/universal-ie/UIE)\]\[[NERRE](https://github.com/LBNLP/NERRE)\]
- **UniGen: A Unified Generative Framework for Retrieval and Question Answering with Large Language Models**, _Li et al._, AAAI 2024. \[[paper](https://arxiv.org/abs/2312.11036)\]
- **INTERS: Unlocking the Power of Large Language Models in Search with Instruction Tuning**, _Zhu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.06532)\]\[[code](https://github.com/DaoD/INTERS)\]
- GenIR: **From Matching to Generation: A Survey on Generative Information Retrieval**, _Li et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2404.14851)\]\[[code](https://github.com/RUC-NLPIR/GenIR-Survey)\]

- **SIGIR-AP 2023 Tutorial: Recent Advances in Generative Information Retrieval** \[[link](https://sigir-ap2023-generative-ir.github.io/)\]
- \[[search_with_lepton](https://github.com/leptonai/search_with_lepton)\]\[[LLocalSearch](https://github.com/nilsherzig/LLocalSearch)\]\[[FreeAskInternet](https://github.com/nashsu/FreeAskInternet)\]\[[storm](https://github.com/stanford-oval/storm)\]


##### 3.2.6 Math

- **ToRA: A Tool-Integrated Reasoning Agent for Mathematical Problem Solving**, _Gou et al._, ICLR 2024. \[[paper](https://arxiv.org/abs/2309.17452)\]\[[code](https://github.com/microsoft/ToRA)\]
- **MathVista: Evaluating Math Reasoning in Visual Contexts with GPT-4V, Bard, and Other Large Multimodal Models**, _Lu et al._, ICLR 2024 Oral. \[[paper](https://arxiv.org/abs/2310.02255)\]\[[code](https://github.com/lupantech/MathVista)\]
- **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models**, _Shao et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.03300)\]\[[code](https://github.com/deepseek-ai/DeepSeek-Math)\]
- **Common 7B Language Models Already Possess Strong Math Capabilities**, _Li et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.04706)\]\[[code](https://github.com/Xwin-LM/Xwin-LM)\]
- **ChatGLM-Math: Improving Math Problem-Solving in Large Language Models with a Self-Critique Pipeline**, _Xu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2404.02893)\]\[[code](https://github.com/THUDM/ChatGLM-Math)\]
- **AlphaMath Almost Zero: process Supervision without process**, _Chen et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2405.03553)\]\[[code](https://github.com/MARIO-Math-Reasoning/Super_MARIO)\]

##### 3.2.7 Medicine and Law

- **A Survey of Large Language Models in Medicine: Progress, Application, and Challenge**, _Zhou et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.05112)\]\[[code](https://github.com/AI-in-Health/MedLLMsPracticalGuide)\]\[[LLM-for-Healthcare](https://github.com/KaiHe-better/LLM-for-Healthcare)\]
- **HuatuoGPT, towards Taming Language Model to Be a Doctor**, _Zhang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.15075)\]\[[code](https://github.com/FreedomIntelligence/HuatuoGPT)\]\[[Medical_NLP](https://github.com/FreedomIntelligence/Medical_NLP)\]\[[Zhongjing](https://github.com/SupritYoung/Zhongjing)\]\[[MedicalGPT](https://github.com/shibing624/MedicalGPT)\]
- **ChatLaw: Open-Source Legal Large Language Model with Integrated External Knowledge Bases**, _Cui et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2306.16092)\]\[[code](https://github.com/PKU-YuanGroup/ChatLaw)\]
- **DISC-LawLLM: Fine-tuning Large Language Models for Intelligent Legal Services**, _Yue et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.11325)\]\[[code](https://github.com/FudanDISC/DISC-LawLLM)\]
- **DISC-MedLLM: Bridging General Large Language Models and Real-World Medical Consultation**, _Bao et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2308.14346)\]\[[code](https://github.com/FudanDISC/DISC-MedLLM)\]
- **MedAgents: Large Language Models as Collaborators for Zero-shot Medical Reasoning**, _Tang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.10537)\]\[[code](https://github.com/gersteinlab/MedAgents)\]
- **MEDITRON-70B: Scaling Medical Pretraining for Large Language Models**, _Chen et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.16079)\]\[[meditron](https://github.com/epfLLM/meditron)\]
- Med-PaLM: **Large language models encode clinical knowledge**, _Singhal et al._, Nature 2023. \[[paper](https://www.nature.com/articles/s41586-023-06291-2)\]\[[Unofficial Implementation](https://github.com/kyegomez/Med-PaLM)\]
- **Capabilities of Gemini Models in Medicine**, _Saab et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2404.18416)\]
- AMIE: **Towards Conversational Diagnostic AI**, _Tu et al._, arxiv 2024.  \[[paper](https://arxiv.org/abs/2401.05654)\]\[[AMIE-pytorch](https://github.com/lucidrains/AMIE-pytorch)\]
- **Apollo: Lightweight Multilingual Medical LLMs towards Democratizing Medical AI to 6B People**, _Wang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.03640)\]\[[code](https://github.com/FreedomIntelligence/Apollo)\]\[[Medical_NLP](https://github.com/FreedomIntelligence/Medical_NLP)\]
- **Agent Hospital: A Simulacrum of Hospital with Evolvable Medical Agents**, _Li et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2405.02957)\]

##### 3.2.8 Recommend System

- **Recommender Systems with Generative Retrieval**, _Rajput et al._, NeurIPS 2022. \[[paper](https://arxiv.org/abs/2305.05065)\]
- **Unifying Large Language Models and Knowledge Graphs: A Roadmap**, _Pan et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2306.08302)\]
- YuLan-Rec: **User Behavior Simulation with Large Language Model based Agents**, _Wang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2306.02552)\]\[[code](https://github.com/RUC-GSAI/YuLan-Rec)\]
- **SSLRec: A Self-Supervised Learning Framework for Recommendation**, _Ren et al._, WSDM 2024 Oral. \[[paper](https://arxiv.org/abs/2308.05697)\]\[[code](https://github.com/HKUDS/SSLRec)\]\[[Awesome-SSLRec-Papers](https://github.com/HKUDS/Awesome-SSLRec-Papers)\]
- RLMRec: **Representation Learning with Large Language Models for Recommendation**, _Ren et al._, WWW 2024. \[[paper](https://arxiv.org/abs/2310.15950)\]\[[code](https://github.com/HKUDS/RLMRec)\]
- **LLMRec: Large Language Models with Graph Augmentation for Recommendation**, _Wei et al._, WSDM 2024 Oral. \[[paper](https://arxiv.org/abs/2311.00423)\]\[[code](https://github.com/HKUDS/LLMRec)\]
- **Agent4Rec_On Generative Agents in Recommendation**, _Zhang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.10108)\]\[[code](https://github.com/LehengTHU/Agent4Rec)\]
- LLM-KERec: **Breaking the Barrier: Utilizing Large Language Models for Industrial Recommendation Systems through an Inferential Knowledge Graph**, _Zhao et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.13750)\]
- **Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations**, _Zhai et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.17152)\]\[[code](https://github.com/facebookresearch/generative-recommenders)\]
- **Wukong: Towards a Scaling Law for Large-Scale Recommendation**, _Zhang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.02545)\]\[[unofficial code](https://github.com/clabrugere/wukong-recommendation)\]
- **RecAI: Leveraging Large Language Models for Next-Generation Recommender Systems**, _Lian et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.06465)\]\[[code](https://github.com/microsoft/RecAI)\]
- \[[recommenders](https://github.com/recommenders-team/recommenders)\]\[[Source code for Twitter's Recommendation Algorithm](https://github.com/twitter/the-algorithm)\]\[[Awesome-RSPapers](https://github.com/RUCAIBox/Awesome-RSPapers)\]\[[RecBole](https://github.com/RUCAIBox/RecBole)\]

##### 3.2.9 Tool Learning

- **Tool Learning with Foundation Models**, _Qin et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2304.08354)\]\[[code](https://github.com/OpenBMB/BMTools)\]
- **Toolformer: Language Models Can Teach Themselves to Use Tools**, _Schick et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2302.04761)\]\[[toolformer-pytorch](https://github.com/lucidrains/toolformer-pytorch)\]\[[toolformer](https://github.com/conceptofmind/toolformer)\]
- **ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs**, _Qin et al._, ICLR 2024 Spotlight. \[[paper](https://arxiv.org/abs/2307.16789)\]\[[code](https://github.com/OpenBMB/ToolBench)\]
- **Gorilla: Large Language Model Connected with Massive APIs**, _Patil et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.15334)\]\[[code](https://github.com/ShishirPatil/gorilla)\]
- **GPT4Tools: Teaching Large Language Model to Use Tools via Self-instruction**, _Yang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.18752)\]\[[code](https://github.com/AILab-CVC/GPT4Tools)\]
- LLMCompiler: **An LLM Compiler for Parallel Function Calling**, _Kim et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.04511)\]\[[code](https://github.com/SqueezeAILab/LLMCompiler)\]
- **Large Language Models as Tool Makers**, _Cai et al_, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.17126)\]\[[code](https://github.com/ctlllll/LLM-ToolMaker)\]
- **ToolAlpaca: Generalized Tool Learning for Language Models with 3000 Simulated Cases** _Tang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2306.05301)\]\[[code](https://github.com/tangqiaoyu/ToolAlpaca)\]\[[ToolQA](https://github.com/night-chen/ToolQA)\]\[[toolbench](https://github.com/sambanova/toolbench)\]
- **ToolChain\*: Efficient Action Space Navigation in Large Language Models with A\* Search**, _Zhuang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.13227)\]\[[code]\]
- **Chameleon: Plug-and-Play Compositional Reasoning with Large Language Models**, _Lu et al._, NeurIPS 2023. \[[paper](https://arxiv.org/abs/2304.09842)\]\[[code](https://github.com/lupantech/chameleon-llm)\]
- **ToolEyes: Fine-Grained Evaluation for Tool Learning Capabilities of Large Language Models in Real-world Scenarios**, _Ye et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.00741)\]\[[code](https://github.com/Junjie-Ye/ToolEyes)\]
- **AnyTool: Self-Reflective, Hierarchical Agents for Large-Scale API Calls**, _Du et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.04253)\]\[[code](https://github.com/dyabel/AnyTool)\]
- **LLMs in the Imaginarium: Tool Learning through Simulated Trial and Error**, _Wang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.04746)\]\[[code](https://github.com/microsoft/simulated-trial-and-error)\]
- **What Are Tools Anyway? A Survey from the Language Model Perspective**, _Wang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.15452)\]
- \[[ToolLearningPapers](https://github.com/thunlp/ToolLearningPapers)\]\[[awesome-tool-llm](https://github.com/zorazrw/awesome-tool-llm)\]

#### 3.3 LLM Technique

- **How to Train Really Large Models on Many GPUs**, _Lilian Weng_, 2021. \[[blog](https://lilianweng.github.io/posts/2021-09-25-train-large/)\]
- **Training great LLMs entirely from ground zero in the wilderness as a startup**, _Yi Tay_, 2024. \[[blog](https://www.yitay.net/blog/training-great-llms-entirely-from-ground-zero-in-the-wilderness)\]
- **Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism**, _Shoeybi et al._, arxiv 2019. \[[paper](https://arxiv.org/abs/1909.08053)\]\[[code](https://github.com/NVIDIA/Megatron-LM)\]
- **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models**, _Rajbhandari et al._, arxiv 2019. \[[paper](https://arxiv.org/abs/1910.02054)\]\[[DeepSpeed](https://github.com/microsoft/DeepSpeed)\]
- **Colossal-AI: A Unified Deep Learning System For Large-Scale Parallel Training**, _Li et al._, ICPP 2023. \[[paper](https://arxiv.org/abs/2110.14883)\]\[[code](https://github.com/hpcaitech/ColossalAI)\]
- **MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs**, _Jiang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.15627)\]
- **A Theory on Adam Instability in Large-Scale Machine Learning**, _Molybog et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2304.09871)\]
- **Loss Spike in Training Neural Networks**, _Zhang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.12133)\]
- **Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling**, _Biderman et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2304.01373)\]\[[code](https://github.com/EleutherAI/pythia)\]
- **Continual Pre-Training of Large Language Models: How to (re)warm your model**, _Gupta et al._, \[[paper](https://arxiv.org/abs/2308.04014)\]
- **FLM-101B: An Open LLM and How to Train It with $100K Budget**, _Li et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.03852)\]\[[model](https://huggingface.co/CofeAI/FLM-101B)\]
- **Instruction Tuning with GPT-4**, _Peng et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2304.03277)\]\[[code](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)\]
- **DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines**, _Khattab et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.03714)\]\[[code](https://github.com/stanfordnlp/dspy)\]
- **A Survey on Self-Evolution of Large Language Models**, _Tao et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2404.14387)\]\[[code](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/Awesome-Self-Evolution-of-LLM)\]

##### 3.3.1 Alignment

- **AI Alignment: A Comprehensive Survey**, _Ji et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.19852)\]\[[PKU-Alignment](https://github.com/PKU-Alignment)\]
- **Large Language Model Alignment: A Survey**, _Shen et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.15025)\]
- **Aligning Large Language Models with Human: A Survey**, _Wang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2307.12966)\]\[[code](https://github.com/GaryYufei/AlignLLMHumanSurvey)\]
- \[[alignment-handbook](https://github.com/huggingface/alignment-handbook)\]

- **Self-Instruct: Aligning Language Models with Self-Generated Instructions**, _Wang et al._, ACL 2023. \[[paper](https://arxiv.org/abs/2212.10560)\]\[[code](https://github.com/yizhongw/self-instruct)\]
- RLHF: \[[hf blog](https://huggingface.co/blog/rlhf)\]\[[OpenAI blog](https://openai.com/research/learning-from-human-preferences)\]\[[alignment blog](https://openai.com/blog/our-approach-to-alignment-research)\]\[[awesome-RLHF](https://github.com/opendilab/awesome-RLHF)\]
- **Secrets of RLHF in Large Language Models** \[[MOSS-RLHF](https://github.com/OpenLMLab/MOSS-RLHF)\]\[[Part I](https://arxiv.org/abs/2307.04964)\]\[[Part II](https://arxiv.org/abs/2401.06080)\]\[[OpenRLHF](https://github.com/OpenLLMAI/OpenRLHF)\]
- **Safe RLHF: Safe Reinforcement Learning from Human Feedback**, _Dai et al._, ICLR 2024 Spotlight. \[[paper](https://arxiv.org/abs/2310.12773)\]\[[code](https://github.com/PKU-Alignment/safe-rlhf)\]
- **The N+ Implementation Details of RLHF with PPO: A Case Study on TL;DR Summarization**, _Huang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.17031)\]\[[code](https://github.com/vwxyzjn/summarize_from_feedback_details)\]\[[blog](https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo)\]\[[trl](https://github.com/huggingface/trl)\]
- **LIMA: Less Is More for Alignment**, _Zhou et al._, NeurIPS 2023. \[[paper](https://arxiv.org/abs/2305.11206)\]
- DPO: **Direct Preference Optimization: Your Language Model is Secretly a Reward Model**, _Rafailov et al._, NeurIPS 2023 Runner-up Award. \[[paper](https://arxiv.org/abs/2305.18290)\]\[[Unofficial Implementation](https://github.com/eric-mitchell/direct-preference-optimization)\]\[[trl](https://github.com/huggingface/trl)\]
- BPO: **Black-Box Prompt Optimization: Aligning Large Language Models without Model Training**, _Cheng et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.04155)\]\[[code](https://github.com/thu-coai/BPO)\]
- **KTO: Model Alignment as Prospect Theoretic Optimization**, _Ethayarajh et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.01306)\]\[[code](https://github.com/ContextualAI/HALOs)\]
- **Constitutional AI: Harmlessness from AI Feedback**, _Bai et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2212.08073)\]\[[code](https://github.com/anthropics/ConstitutionalHarmlessnessPaper)\]
- **RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback**, _Lee et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.00267)\]\[[code]\]\[[awesome-RLAIF](https://github.com/mengdi-li/awesome-RLAIF)\]
- **Direct Language Model Alignment from Online AI Feedback**, _Guo et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.04792)\]
- **ReMax: A Simple, Effective, and Efficient Reinforcement Learning Method for Aligning Large Language Models**, _Li et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.10505)\]\[[code](https://github.com/liziniu/ReMax)\]\[[policy_optimization](https://github.com/liziniu/policy_optimization)\]
- **Zephyr: Direct Distillation of LM Alignment**, _Tunstall et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.16944)\]\[[code](https://github.com/huggingface/alignment-handbook)\]

- **Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision**, _Burns et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.09390)\]\[[code](https://github.com/openai/weak-to-strong)\]
- SPIN: **Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models**, _Chen et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.01335)\]\[[code](https://github.com/uclaml/SPIN)\]\[[unofficial implementation](https://github.com/thomasgauthier/LLM-self-play)\]
- CALM: **LLM Augmented LLMs: Expanding Capabilities through Composition**, _Bansal et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.02412)\]\[[CALM-pytorch](https://github.com/lucidrains/CALM-pytorch)\]
- **Self-Rewarding Language Models**, _Yuan et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.10020)\]\[[unofficial implementation](https://github.com/lucidrains/self-rewarding-lm-pytorch)\]
- Anthropic: **Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training**, _Hubinger et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.05566)\]
- **LongAlign: A Recipe for Long Context Alignment of Large Language Models**, _Bai et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.18058)\]\[[code](https://github.com/THUDM/LongAlign)\]
- **Aligner: Achieving Efficient Alignment through Weak-to-Strong Correction**, _Ji et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.02416)\]\[[code](https://github.com/Aligner2024/aligner)\]
- **A Survey on Knowledge Distillation of Large Language Models**, _Xu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.13116)\]\[[code](https://github.com/Tebmer/Awesome-Knowledge-Distillation-of-LLMs)\]
- **NeMo-Aligner: Scalable Toolkit for Efficient Model Alignment**, _Shen et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2405.01481)\]\[[code](https://github.com/NVIDIA/NeMo-Aligner)\]

##### 3.3.2 Context Length

- ALiBi: **Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation**, _Press et al._, ICLR 2022. \[[paper](https://arxiv.org/abs/2108.12409)\]\[[code](https://github.com/ofirpress/attention_with_linear_biases)\]
- **Extending Context Window of Large Language Models via Positional Interpolation**, _Chen et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2306.15595)\]
- **Scaling Transformer to 1M tokens and beyond with RMT**, _Bulatov et al._, AAAI 2024. \[[paper](https://arxiv.org/abs/2304.11062)\]\[[code](https://github.com/booydar/recurrent-memory-transformer/tree/aaai24)\]\[[LM-RMT](https://github.com/booydar/LM-RMT)\]
- **LongNet: Scaling Transformers to 1,000,000,000 Tokens**, _Ding et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2307.02486)\]\[[code](https://github.com/microsoft/torchscale/blob/main/torchscale/model/LongNet.py)\]\[[unofficial code](https://github.com/kyegomez/LongNet)\]
- **LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models**, _Chen et al._, ICLR 2024 Oral. \[[paper](https://arxiv.org/abs/2309.12307)\]\[[code](https://github.com/dvlab-research/LongLoRA)\]
- StreamingLLM: **Efficient Streaming Language Models with Attention Sinks**, _Xiao et al._, ICLR 2024. \[[paper](https://arxiv.org/abs/2309.17453)\]\[[code](https://github.com/mit-han-lab/streaming-llm)\]\[[SwiftInfer](https://github.com/hpcaitech/SwiftInfer)\]\[[SwiftInfer blog](https://hpc-ai.com/blog/colossal-ai-swiftinfer)\]
- **YaRN: Efficient Context Window Extension of Large Language Models**, _Peng et al._, ICLR 2024. \[[paper](https://arxiv.org/abs/2309.00071)\]\[[code](https://github.com/jquesnelle/yarn)\]
- **LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression**, _Jiang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.06839)\]\[[code](https://github.com/microsoft/LLMLingua)\]
- **LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens**, _Ding et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.13753)\]\[[code](https://github.com/jshuadvd/LongRoPE)\]
- **LLM Maybe LongLM: Self-Extend LLM Context Window Without Tuning**, _Jin et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.01325)\]\[[code](https://github.com/datamllab/LongLM)\]
- **The What, Why, and How of Context Length Extension Techniques in Large Language Models -- A Detailed Survey**, _Pawar et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.07872)\]
- **Data Engineering for Scaling Language Models to 128K Context**, _Fu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.10171)\]\[[code](https://github.com/FranxYao/Long-Context-Data-Engineering)\]
- CEPE: **Long-Context Language Modeling with Parallel Context Encoding**, _Yen et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.16617)\]\[[code](https://github.com/princeton-nlp/CEPE)\]
- **Counting-Stars: A Simple, Efficient, and Reasonable Strategy for Evaluating Long-Context Large Language Models**, _Song et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.11802)\]\[[code](https://github.com/nick7nlp/Counting-Stars)\]
- **Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention**, _Munkhdalai et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2404.07143)\]\[[infini-transformer-pytorch](https://github.com/lucidrains/infini-transformer-pytorch)\]\[[InfiniTransformer](https://github.com/Beomi/InfiniTransformer)\]\[[infini-mini-transformer](https://github.com/jiahe7ay/infini-mini-transformer)\]\[[megalodon](https://github.com/XuezheMax/megalodon)\]
- **Extending Llama-3's Context Ten-Fold Overnight**, _Zhang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2404.19553)\]\[[code](https://github.com/FlagOpen/FlagEmbedding/tree/master/Long_LLM/longllm_qlora)\]\[[activation_beacon](https://github.com/FlagOpen/FlagEmbedding/tree/master/Long_LLM/activation_beacon)\]

##### 3.3.3 Corpus

- \[[datatrove](https://github.com/huggingface/datatrove)\]\[[datasets](https://github.com/huggingface/datasets)\]\[[doccano](https://github.com/doccano/doccano)\]
- C4: **Documenting Large Webtext Corpora: A Case Study on the Colossal Clean Crawled Corpus**, _Dodge et al._, arxiv 2021. \[[paper](https://arxiv.org/abs/2104.08758)\]\[[dataset](https://huggingface.co/datasets/allenai/c4)\]\[[fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)\]
- **The BigScience ROOTS Corpus: A 1.6TB Composite Multilingual Dataset**, _Laurençon et al._, NeurIPS 2023. \[[paper](https://arxiv.org/abs/2303.03915)\]\[[code](https://github.com/bigscience-workshop/data-preparation)\]\[[dataset](https://huggingface.co/bigscience-data)\]
- **Data-Juicer: A One-Stop Data Processing System for Large Language Models**, _Chen et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.02033)\]\[[code](https://github.com/modelscope/data-juicer)\]
- **UltraFeedback: Boosting Language Models with High-quality Feedback**, _Cui et al._, ICML 2024. \[[paper](https://arxiv.org/abs/2310.01377)\]\[[code](https://github.com/OpenBMB/UltraFeedback)\]
- **What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning**, _Liu et al._, ICLR 2024. \[[paper](https://arxiv.org/abs/2312.15685)\]\[[code](https://github.com/hkust-nlp/deita)\]
- **WanJuan-CC: A Safe and High-Quality Open-sourced English Webtext Dataset**, _Qiu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.19282)\]\[[dataset](https://opendatalab.com/OpenDataLab/WanJuanCC)\]
- **Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research**, _Soldaini et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.00159)\]\[[code](https://github.com/allenai/dolma)\]\[[OLMo](https://github.com/allenai/OLMo)\]
- **Datasets for Large Language Models: A Comprehensive Survey**, _Liu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.18041)\]\[[Awesome-LLMs-Datasets](https://github.com/lmmlzn/Awesome-LLMs-Datasets)\]
- **DataDreamer: A Tool for Synthetic Data Generation and Reproducible LLM Workflows**, _Patel et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.10379)\]\[[code](https://github.com/datadreamer-dev/datadreamer)\]
- **Large Language Models for Data Annotation: A Survey**, _Tan et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.13446)\]\[[code](https://github.com/Zhen-Tan-dmml/LLM4Annotation)\]
- **Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance**, _Ye et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.16952)\]\[[code](https://github.com/yegcjs/mixinglaws)\]
- **COIG-CQIA: Quality is All You Need for Chinese Instruction Fine-tuning**, _Bai et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.18058)\]\[[dataset](https://huggingface.co/datasets/m-a-p/COIG-CQIA)\]
- **Best Practices and Lessons Learned on Synthetic Data for Language Models**, _Liu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2404.07503)\]

##### 3.3.4 Evaluation

- \[[Awesome-LLM-Eval](https://github.com/onejune2018/Awesome-LLM-Eval)\]\[[LLM-eval-survey](https://github.com/MLGroupJLU/LLM-eval-survey)\]
- MMLU: **Measuring Massive Multitask Language Understanding**, _Hendrycks et al._, ICLR 2021.  \[[paper](https://arxiv.org/abs/2009.03300)\]\[[code](https://github.com/hendrycks/test)\]
- HELM: **Holistic Evaluation of Language Models**, _Liang et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2211.09110)\]\[[code](https://github.com/stanford-crfm/helm)\]
- **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena**, _Zheng et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2306.05685)\]\[[code](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge)\]
- **SuperCLUE: A Comprehensive Chinese Large Language Model Benchmark**, _Xu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2307.15020)\]\[[code](https://github.com/CLUEbenchmark/SuperCLUE)\]\[[SuperCLUE-RAG](https://github.com/CLUEbenchmark/SuperCLUE-RAG)\]
- **C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for Foundation Models**, _Huang et al._, NeurIPS 2023. \[[paper](https://arxiv.org/abs/2305.08322)\]\[[code](https://github.com/hkust-nlp/ceval)\]\[[chinese-llm-benchmark](https://github.com/jeinlee1991/chinese-llm-benchmark)\]
- **CMMLU: Measuring massive multitask language understanding in Chinese**, _Li et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2306.09212)\]\[[code](https://github.com/haonan-li/CMMLU)\]
- **CMMMU: A Chinese Massive Multi-discipline Multimodal Understanding Benchmark**, _Zhang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.11944)\]\[[code](https://github.com/CMMMU-Benchmark/CMMMU)\]
- **Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference**, _Chiang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.04132)\]\[[demo](https://chat.lmsys.org/)\]
- **Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models**, _Kim et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2405.01535)\]\[[code](https://github.com/prometheus-eval/prometheus-eval)\]
- \[[Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)\]
- \[[AlpacaEval Leaderboard](https://tatsu-lab.github.io/alpaca_eval/)\]\[[alpaca_eval](https://github.com/tatsu-lab/alpaca_eval)\]
- \[[Chatbot-Arena-Leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)\]\[[blog](https://lmsys.org/blog/2023-05-03-arena/)\]\[[FastChat](https://github.com/lm-sys/FastChat)\]\[[arena-hard](https://github.com/lm-sys/arena-hard)\]
- \[[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)\]\[[OpenAI Evals](https://github.com/openai/evals)\]\[[simple-evals](https://github.com/openai/simple-evals)\]
- \[[OpenCompass](https://github.com/open-compass/opencompass)\]
- \[[llm-colosseum](https://github.com/OpenGenerativeAI/llm-colosseum)\]

##### 3.3.5 Hallucination

- **Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models**, _Zhang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.01219)\]\[[code](https://github.com/HillZhang1999/llm-hallucination-survey)\]
- **A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions**, _Huang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.05232)\]\[[code](https://github.com/LuckyyySTA/Awesome-LLM-hallucination)\]\[[Awesome-MLLM-Hallucination](https://github.com/showlab/Awesome-MLLM-Hallucination)\]
- **The Dawn After the Dark: An Empirical Study on Factuality Hallucination in Large Language Models**, _Li et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.03205)\]\[[code](https://github.com/RUCAIBox/HaluEval-2.0)\]
- **Chain-of-Verification Reduces Hallucination in Large Language Models**, _Dhuliawala et al._, arxiv 2023. \[[paper]\(https://arxiv.org/abs/2309.11495)]\[[code](https://github.com/lastmile-ai/aiconfig/tree/main/cookbooks/Chain-of-Verification)\]
- **HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models**, _Guan et al._, CVPR 2024. \[[paper](https://arxiv.org/abs/2310.14566)\]\[[code](https://github.com/tianyi-lab/HallusionBench)\]
- **Woodpecker: Hallucination Correction for Multimodal Large Language Models**, _Yin et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.16045)\]\[[code](https://github.com/BradyFU/Woodpecker)\]
- **TrustLLM: Trustworthiness in Large Language Models**, _Sun et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.05561)\]\[[code](https://github.com/HowieHwong/TrustLLM)\]
- SAFE: **Long-form factuality in large language models**, _Wei et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.18802)\]\[[code](https://github.com/google-deepmind/long-form-factuality)\]

##### 3.3.6 Inference

- **How to make LLMs go fast**, 2023. \[[blog](https://vgel.me/posts/faster-inference/)\]
- **Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems**, _Miao et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.15234)\]\[[Awesome-Quantization-Papers](https://github.com/Zhen-Dong/Awesome-Quantization-Papers)\]\[[awesome-model-quantization](https://github.com/htqin/awesome-model-quantization)\]
- **LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale**, _Dettmers et al._, NeurIPS 2022. \[[paper](https://arxiv.org/abs/2208.07339)\]\[[code](https://github.com/TimDettmers/bitsandbytes)\]
- **LLM-FP4: 4-Bit Floating-Point Quantized Transformers**, _Liu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.16836)\]\[[code](https://github.com/nbasyl/LLM-FP4)\]
- **OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models**, _Shao et al._, ICLR 2024 Spotlight. \[[paper](https://arxiv.org/abs/2308.13137)\]\[[code](https://github.com/OpenGVLab/OmniQuant)\]
- **BitNet: Scaling 1-bit Transformers for Large Language Models**, _Wang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.11453)\]\[[code](https://github.com/microsoft/unilm/tree/master/bitnet)\]\[[unofficial implementation](https://github.com/kyegomez/BitNet)\]
- **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers**, _Frantar et al._, ICLR 2023. \[[paper](https://arxiv.org/abs/2210.17323)\]\[[code](https://github.com/IST-DASLab/gptq)\]\[[AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)\]
- **QMoE: Practical Sub-1-Bit Compression of Trillion-Parameter Models**, _Frantar et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.16795)\]\[[code](https://github.com/IST-DASLab/qmoe)\]
- **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration**, _Lin et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2306.00978)\]\[[code](https://github.com/mit-han-lab/llm-awq)\]\[[AutoAWQ](https://github.com/casper-hansen/AutoAWQ)\]
- **LLM in a flash: Efficient Large Language Model Inference with Limited Memory**, _Alizadeh et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.11514)\]\[[air_llm](https://github.com/lyogavin/Anima/tree/main/air_llm)\]
- **LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models**, _Jiang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.05736)\]\[[code](https://github.com/microsoft/LLMLingua)\]
- **FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU**, _Sheng et al._, ICML 2023. \[[paper](https://arxiv.org/abs/2303.06865)\]\[[code](https://github.com/FMInference/FlexGen)\]
- **PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU**, _Song et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.12456)\]\[[code](https://github.com/SJTU-IPADS/PowerInfer)\]\[[llama.cpp](https://github.com/ggerganov/llama.cpp)\]\[[Anima](https://github.com/lyogavin/Anima)\]
- **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**, _Dao et al._, NeurIPS 2022. \[[paper](https://arxiv.org/abs/2205.14135)\]\[[code](https://github.com/Dao-AILab/flash-attention)\]
- **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning**, _Tri Dao_, arxiv 2023. \[[paper](https://arxiv.org/abs/2307.08691)\]\[[code](https://github.com/Dao-AILab/flash-attention)\]
- vllm: **Efficient Memory Management for Large Language Model Serving with PagedAttention**, _Kwon et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.06180)\]\[[code](https://github.com/vllm-project/vllm)\]
- **Fast and Expressive LLM Inference with RadixAttention and SGLang**, _Zheng et al._, Stanford blog 2024. \[[blog](https://lmsys.org/blog/2024-01-17-sglang/)\]\[[code](https://github.com/sgl-project/sglang/)\]
- **Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads**, _Cai et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.10774)\]\[[code](https://github.com/FasterDecoding/Medusa)\]
- **EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty**, _Li et al._, ICML 2024. \[[paper](https://arxiv.org/abs/2401.15077)\]\[[code](https://github.com/SafeAILab/EAGLE)\]
- **APAR: LLMs Can Do Auto-Parallel Auto-Regressive Decoding**, _Liu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.06761)\]\[[code]\]\[[Ouroboros](https://github.com/thunlp/Ouroboros)\]

- \[[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)\]\[[FasterTransformer](https://github.com/NVIDIA/FasterTransformer)\]\[[TritonServer](https://github.com/triton-inference-server/server)\]\[[GenerativeAIExamples](https://github.com/NVIDIA/GenerativeAIExamples)\]
- \[[DeepSpeed-MII](https://github.com/microsoft/DeepSpeed-MII)\]\[[DeepSpeed-FastGen](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen)\]\[[ONNX Runtime](https://github.com/microsoft/onnxruntime)\]\[[onnx](https://github.com/onnx/onnx)\]
- \[[text-generation-inference](https://github.com/huggingface/text-generation-inference)\]\[[quantization](https://huggingface.co/docs/transformers/main/en/quantization)\]\[[quanto](https://github.com/huggingface/quanto)\]
- \[[OpenLLM](https://github.com/bentoml/OpenLLM)\]\[[mlc-llm](https://github.com/mlc-ai/mlc-llm)\]
- \[[LMDeploy](https://github.com/InternLM/lmdeploy)\]
- \[[ggml](https://github.com/ggerganov/ggml)\]\[[exllamav2](https://github.com/turboderp/exllamav2)\]\[[llama.cpp](https://github.com/ggerganov/llama.cpp)\]\[[gpt-fast](https://github.com/pytorch-labs/gpt-fast)\]\[[fastllm](https://github.com/ztxz16/fastllm)\]\[[CTranslate2](https://github.com/OpenNMT/CTranslate2)\]\[[ipex-llm](https://github.com/intel-analytics/ipex-llm)\]\[[rtp-llm](https://github.com/alibaba/rtp-llm)\]
- \[[ChuanhuChatGPT](https://github.com/GaiZhenbiao/ChuanhuChatGPT)\]\[[ChatGPT-Next-Web](https://github.com/ChatGPTNextWeb/ChatGPT-Next-Web)\]\[[OpenLLM](https://github.com/bentoml/OpenLLM)\]

##### 3.3.7 MoE

- **Mixture of Experts Explained**, _Sanseviero et al._, Hugging Face Blog 2023. \[[blog](https://huggingface.co/blog/moe)\]
- **Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer**, _Shazeer et al._, arxiv 2017. \[[paper](https://arxiv.org/abs/1701.06538)\]\[[Re-Implementation](https://github.com/davidmrau/mixture-of-experts)\]
- **GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding**, _Lepikhin et al._, arxiv 2020. \[[paper](https://arxiv.org/abs/2006.16668)\]\[[mixture-of-experts](https://github.com/lucidrains/mixture-of-experts)\]
- **MegaBlocks: Efficient Sparse Training with Mixture-of-Experts**, _Gale et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2211.15841)\]\[[code](https://github.com/stanford-futuredata/megablocks)\]
- **Mixture-of-Experts Meets Instruction Tuning:A Winning Combination for Large Language Models**, _Shen et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.14705)\]\[[code]\]
- **Fast Inference of Mixture-of-Experts Language Models with Offloading**, _Eliseev and Mazur_, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.17238)\]\[[code](https://github.com/dvmazur/mixtral-offloading)\]
- Mixtral-8×7B: **Mixtral of Experts**, _Jiang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2401.04088)\]\[[code](https://github.com/mistralai/mistral-src)\]\[[megablocks-public](https://github.com/mistralai/megablocks-public)\]\[[model](https://huggingface.co/mistralai)\]\[[blog](https://mistral.ai/news/mixtral-of-experts/)\]\[[Chinese-Mixtral-8x7B](https://github.com/HIT-SCIR/Chinese-Mixtral-8x7B)\]\[[Chinese-Mixtral](https://github.com/ymcui/Chinese-Mixtral)\]
- **DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models**, _Dai et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.06066)\]\[[code](https://github.com/deepseek-ai/DeepSeek-MoE)\]
- **DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model**, _DeepSeek-AI_, arxiv 2024. \[[paper](https://arxiv.org/abs/2405.04434)\]\[[code](https://github.com/deepseek-ai/DeepSeek-V2)\]
- **Evolutionary Optimization of Model Merging Recipes**, _Akiba et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.13187)\]\[[code](https://github.com/SakanaAI/evolutionary-model-merge)\]

- \[[llama-moe](https://github.com/pjlab-sys4nlp/llama-moe)\]\[[Aurora](https://github.com/WangRongsheng/Aurora)\]\[[OpenMoE](https://github.com/XueFuzhao/OpenMoE)\]\[[makeMoE](https://github.com/AviSoori1x/makeMoE)\]

##### 3.3.8 PEFT (Parameter-efficient Fine-tuning)

- \[[DeepSpeed](https://github.com/microsoft/DeepSpeed)\]\[[DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples)\]\[[blog](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)\]
- \[[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)\]\[[Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed)\]\[[Megatron-DeepSpeed](https://github.com/bigscience-workshop/Megatron-DeepSpeed)\]
- \[[torchtune](https://github.com/pytorch/torchtune)\]\[[torchtitan](https://github.com/pytorch/torchtitan)\]
- \[[PEFT](https://github.com/huggingface/peft)\]\[[trl](https://github.com/huggingface/trl)\]\[[accelerate](https://github.com/huggingface/accelerate)\]\[[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)\]\[[xtuner](https://github.com/InternLM/xtuner)\]\[[MFTCoder](https://github.com/codefuse-ai/MFTCoder)\]\[[llm-foundry](https://github.com/mosaicml/llm-foundry)\]\[[swift](https://github.com/modelscope/swift)\]
- \[[mergekit](https://github.com/cg123/mergekit)\]\[[Model Merging](https://huggingface.co/collections/osanseviero/model-merging-65097893623330a3a51ead66)\]\[[OpenChatKit](https://github.com/togethercomputer/OpenChatKit)\]

- **LoRA: Low-Rank Adaptation of Large Language Models**, _Hu et al._, arxiv 2021. \[[paper](https://arxiv.org/abs/2106.09685)\]\[[code](https://github.com/microsoft/LoRA)\]\[[LoRA From Scratch](https://lightning.ai/lightning-ai/studios/code-lora-from-scratch)\]\[[lora](https://github.com/cloneofsimo/lora)\]\[[dora](https://github.com/catid/dora)\]
- **QLoRA: Efficient Finetuning of Quantized LLMs**, _Dettmers et al._, NeurIPS 2023 Oral. \[[paper](https://arxiv.org/abs/2305.14314)\]\[[code](https://github.com/artidoro/qlora)\]\[[bitsandbytes](https://github.com/TimDettmers/bitsandbytes)\]\[[unsloth](https://github.com/unslothai/unsloth)\]
- **S-LoRA: Serving Thousands of Concurrent LoRA Adapters**, _Sheng et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.03285)\]\[[code](https://github.com/S-LoRA/S-LoRA)\]
- **GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection**, _Zhao et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.03507)\]\[[code](https://github.com/jiaweizzhao/galore)\]
- **Prefix-Tuning: Optimizing Continuous Prompts for Generation**, _Li et al._, ACL 2021. \[[paper](https://arxiv.org/abs/2101.00190)\]\[[code](https://github.com/XiangLi1999/PrefixTuning)\]
- Adapter: **Parameter-Efficient Transfer Learning for NLP**, _Houlsby et al._, ICML 2019. \[[paper](https://arxiv.org/abs/1902.00751)\]\[[code](https://github.com/google-research/adapter-bert)\]\[[unify-parameter-efficient-tuning](https://github.com/jxhe/unify-parameter-efficient-tuning)\]
- **Adapters: A Unified Library for Parameter-Efficient and Modular Transfer Learning**, _Poth et al._, EMNLP 2023. \[[paper](https://arxiv.org/abs/2311.11077)\]\[[code](https://github.com/adapter-hub/adapters)\]
- **LLM-Adapters: An Adapter Family for Parameter-Efficient Fine-Tuning of Large Language Models**, _Hu et al._, EMNLP 2023. \[[paper](https://arxiv.org/abs/2304.01933)\]\[[code](https://github.com/AGI-Edgerunners/LLM-Adapters)\]
- **LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention**, _Zhang et al._, ICLR 2024. \[[paper](https://arxiv.org/abs/2303.16199)\]\[[code](https://github.com/OpenGVLab/LLaMA-Adapter)\]
- **LLaMA Pro: Progressive LLaMA with Block Expansion**, _Wu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.02415)\]\[[code](https://github.com/TencentARC/LLaMA-Pro)\]
- P-Tuning: **GPT Understands, Too**, _Liu et al._, arxiv 2021. \[[paper](https://arxiv.org/abs/2103.10385)\]\[[code](https://github.com/THUDM/P-tuning)\]
- **P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks**, _Liu et al._, ACL 2022. \[[paper](https://arxiv.org/abs/2110.07602)\]\[[code](https://github.com/THUDM/P-tuning-v2)\]
- **Towards a Unified View of Parameter-Efficient Transfer Learning**, _He et al._, ICLR 2022. \[[paper](https://arxiv.org/abs/2110.04366)\]\[[code](https://github.com/jxhe/unify-parameter-efficient-tuning)\]
- **Mixed Precision Training**, _Micikevicius et al._, ICLR 2018. \[[paper](https://arxiv.org/abs/1710.03740)\]
- **8-bit Optimizers via Block-wise Quantization** _Dettmers et al._, ICLR 2022. \[[paper](https://arxiv.org/abs/2110.02861)\]\[[code](https://github.com/timdettmers/bitsandbytes)\]
- **FP8-LM: Training FP8 Large Language Models** _Peng et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.18313)\]\[[code](https://github.com/Azure/MS-AMP)\]
- **Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive Survey**, _Han et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.14608)\]
- **LISA: Layerwise Importance Sampling for Memory-Efficient Large Language Model Fine-Tuning**, _Pan et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.17919)\]\[[code](https://github.com/OptimalScale/LMFlow)\]
- **LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models**, _Zheng et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.13372)\]\[[code](https://github.com/hiyouga/LLaMA-Factory)\]
- **ReFT: Representation Finetuning for Language Models**, _Wu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2404.03592)\]\[[code](https://github.com/stanfordnlp/pyreft)\]

##### 3.3.9 Prompt Learning

- **OpenPrompt: An Open-source Framework for Prompt-learning**, _Ding et al._, arxiv 2021. \[[paper](https://arxiv.org/abs/2111.01998)\]\[[code](https://github.com/thunlp/OpenPrompt)\]
- **Learning to Generate Prompts for Dialogue Generation through Reinforcement Learning**, _Su et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2206.03931)\]
- **Large Language Models Are Human-Level Prompt Engineers**, _Zhou et al._, ICLR 2023. \[[paper](https://arxiv.org/abs/2211.01910)\]\[[code](https://github.com/keirp/automatic_prompt_engineer)\]
- **Large Language Models as Optimizers**, _Yang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.03409)\]\[[code](https://github.com/google-deepmind/opro)\]
- **Principled Instructions Are All You Need for Questioning LLaMA-1/2, GPT-3.5/4**, _Bsharat et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.16171)\]\[[code](https://github.com/VILA-Lab/ATLAS)\]
- **Meta-Prompting: Enhancing Language Models with Task-Agnostic Scaffolding**, _Suzgun and Kalai_, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.12954)\]\[[code](https://github.com/suzgunmirac/meta-prompting)\]
- AutoPrompt: **Intent-based Prompt Calibration: Enhancing prompt optimization with synthetic boundary cases**, _Levi et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.03099)\]\[[code](https://github.com/Eladlev/AutoPrompt)\]\[[automatic_prompt_engineer](https://github.com/keirp/automatic_prompt_engineer)\]
- \[[PromptPapers](https://github.com/thunlp/PromptPapers)\]\[[ChatGPT Prompt Engineering for Developers](https://prompt-engineering.xiniushu.com/)\]\[[Prompt Engineering Guide](https://www.promptingguide.ai/zh)\]\[[k12promptguide](https://www.k12promptguide.com/)\]\[[gpt-prompt-engineer](https://github.com/mshumer/gpt-prompt-engineer)\]\[[awesome-chatgpt-prompts](https://github.com/f/awesome-chatgpt-prompts)\]\[[awesome-chatgpt-prompts-zh](https://github.com/PlexPt/awesome-chatgpt-prompts-zh)\]

- **The Power of Scale for Parameter-Efficient Prompt Tuning**, _Lester et al._, EMNLP 2021. \[[paper](https://arxiv.org/abs/2104.08691)\]\[[code](https://github.com/google-research/prompt-tuning)\]\[[soft-prompt-tuning](https://github.com/kipgparker/soft-prompt-tuning)\]\[[Prompt-Tuning](https://github.com/mkshing/Prompt-Tuning)\]
- **A Survey on In-context Learning**, _Dong et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2301.00234)\]
- **Rethinking the Role of Demonstrations: What Makes In-Context Learning Work**, _Min et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2202.12837)\]\[[code](https://github.com/Alrope123/rethinking-demonstrations)\]
- **Larger language models do in-context learning differently**, _Wei et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2303.03846)\]
- **PAL: Program-aided Language Models**, _Gao et al._, ICML 2023. \[[paper](https://arxiv.org/abs/2211.10435)\]\[[code](https://github.com/reasoning-machines/pal)\]
- **A Comprehensive Survey on Instruction Following**, _Lou et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2303.10475)\]\[[code](https://github.com/RenzeLou/awesome-instruction-learning)\]
- RLHF: **Fine-Tuning Language Models from Human Preferences**, _Ziegler et al._, arxiv 2019. \[[paper](https://arxiv.org/abs/1909.08593)\]\[[code](https://github.com/openai/lm-human-preferences)\]
- RLHF: **Learning to summarize from human feedback**, _Stiennon et al._, NeurIPS 2020. \[[paper](https://arxiv.org/abs/2009.01325)\]\[[code](https://github.com/openai/summarize-from-feedback)\]
- RLHF: **Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback**, _Bai et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2204.05862)\]\[[code](https://github.com/anthropics/hh-rlhf)\]
- **Finetuned Language Models Are Zero-Shot Learners**, _Wei et al._, ICLR 2022. \[[paper](https://arxiv.org/abs/2109.01652)\]
- **Instruction Tuning for Large Language Models: A Survey**, _Zhang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2308.10792)\]\[[code](https://github.com/xiaoya-li/Instruction-Tuning-Survey)\]
- **What learning algorithm is in-context learning? Investigations with linear models**, _Akyürek et al._, ICLR 2023. \[[paper](https://arxiv.org/abs/2211.15661)\]
- **Why Can GPT Learn In-Context? Language Models Implicitly Perform Gradient Descent as Meta-Optimizers**, _Dai et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2212.10559)\]\[[code](https://github.com/microsoft/LMOps/tree/main/understand_icl)\]

##### 3.3.10 RAG (Retrieval Augmented Generation)

- **Retrieval-Augmented Generation for Large Language Models: A Survey**, _Gao et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.10997)\]\[[code](https://github.com/Tongji-KGLLM/RAG-Survey)\]
- **Retrieval-Augmented Generation for AI-Generated Content: A Survey**, _Zhao et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.19473)\]\[[code](https://github.com/hymie122/RAG-Survey)\]
- **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**, _Lewis et al._, NeurIPS 2020. \[[paper](https://arxiv.org/abs/2005.11401)\]\[[code](https://github.com/huggingface/transformers/tree/main/examples/research_projects/rag)\]\[[model](https://huggingface.co/facebook/rag-token-nq)\]\[[docs](https://huggingface.co/docs/transformers/main/model_doc/rag)\]\[[FAISS](https://github.com/facebookresearch/faiss)\]
- **Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection**, _Asai et al._, ICLR 2024 Oral. \[[paper](https://arxiv.org/abs/2310.11511)\]\[[code](https://github.com/AkariAsai/self-rag)\]\[[CRAG](https://github.com/HuskyInSalt/CRAG)\]
- **Dense Passage Retrieval for Open-Domain Question Answering**, _Karpukhin et al._, EMNLP 2020. \[[paper](https://arxiv.org/abs/2004.04906)\]\[[code](https://github.com/facebookresearch/DPR)\]
- **Internet-Augmented Dialogue Generation** _Komeili et al._, arxiv 2021. \[[paper](https://arxiv.org/abs/2107.07566)\]
- RETRO: **Improving language models by retrieving from trillions of tokens**, _Borgeaud et al._, arxiv 2021. \[[paper](https://arxiv.org/abs/2112.04426)\]\[[RETRO-pytorch](https://github.com/lucidrains/RETRO-pytorch)\]
- **FreshLLMs: Refreshing Large Language Models with Search Engine Augmentation**, _Vu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.03214)\]\[[code](https://github.com/freshllms/freshqa)\]
- **Chain-of-Note: Enhancing Robustness in Retrieval-Augmented Language Models**, _Yu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.09210)\]
- **Learning to Filter Context for Retrieval-Augmented Generation**, _Wang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.08377)\]\[[code](https://github.com/zorazrw/filco)\]
- **When Large Language Models Meet Vector Databases: A Survey**, _Jing et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.01763)\]
- **RAFT: Adapting Language Model to Domain Specific RAG**, _Zhang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.10131)\]\[[code](https://github.com/ShishirPatil/gorilla/tree/main/raft)\]
- **RA-ISF: Learning to Answer and Understand from Retrieval Augmentation via Iterative Self-Feedback**, _Liu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.06840)\]\[[code](https://github.com/OceannTwT/ra-isf)\]
- **RQ-RAG: Learning to Refine Queries for Retrieval Augmented Generation**, _Chan et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2404.00610)\]\[[code](https://github.com/chanchimin/RQ-RAG)\]\[[Adaptive-RAG](https://github.com/starsuzi/Adaptive-RAG)\]

- **ACL 2023 Tutorial: Retrieval-based Language Models and Applications**, _Asai et al._, ACL 2023. \[[link](https://acl2023-retrieval-lm.github.io/)\]
- \[[Advanced RAG Techniques: an Illustrated Overview](https://pub.towardsai.net/advanced-rag-techniques-an-illustrated-overview-04d193d8fec6)\]\[[Chinese Version](https://zhuanlan.zhihu.com/p/674755232)\]
- \[[LangChain](https://github.com/langchain-ai/langchain)\]\[[blog](https://blog.langchain.dev/deconstructing-rag/)\]
- \[[LlamaIndex](https://github.com/run-llama/llama_index)\]\[[A Cheat Sheet and Some Recipes For Building Advanced RAG](https://blog.llamaindex.ai/a-cheat-sheet-and-some-recipes-for-building-advanced-rag-803a9d94c41b)\]
- \[[chatgpt-retrieval-plugin](https://github.com/openai/chatgpt-retrieval-plugin)\]
- \[[haystack](https://github.com/deepset-ai/haystack)\]\[[Langchain-Chatchat](https://github.com/chatchat-space/Langchain-Chatchat)\]
- **Browse the web with GPT-4V and Vimium** \[[vimGPT](https://github.com/ishan0102/vimGPT)\]
- \[[QAnything](https://github.com/netease-youdao/QAnything)\]\[[ragflow](https://github.com/infiniflow/ragflow)\]\[[fastRAG](https://github.com/IntelLabs/fastRAG)\]
- \[[trt-llm-rag-windows](https://github.com/NVIDIA/trt-llm-rag-windows)\]\[[history_rag](https://github.com/wxywb/history_rag)\]\[[gpt-crawler](https://github.com/BuilderIO/gpt-crawler)\]\[[R2R](https://github.com/SciPhi-AI/R2R)\]\[[rag-notebook-to-microservices](https://github.com/wenqiglantz/rag-notebook-to-microservices)\]\[[MaxKB](https://github.com/1Panel-dev/MaxKB)\]\[[Verba](https://github.com/weaviate/Verba)\]

###### Text Embedding

- **Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks**, _Reimers et al._, EMNLP 2019. \[[paper](https://arxiv.org/abs/1908.10084)\]\[[code](https://github.com/UKPLab/sentence-transformers)\]\[[model](https://huggingface.co/sentence-transformers)\]\[[model](https://huggingface.co/sentence-transformers)\]\[[vec2text](https://github.com/jxmorris12/vec2text)\]
- **SimCSE: Simple Contrastive Learning of Sentence Embeddings**, _Gao et al._, EMNLP 2021. \[[paper](https://arxiv.org/abs/2104.08821)\]\[[code](https://github.com/princeton-nlp/SimCSE)\]
- OpenAI: **Text and Code Embeddings by Contrastive Pre-Training**, _Neelakantan et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2201.10005)\]\[[blog](https://openai.com/blog/introducing-text-and-code-embeddings)\]
- MRL: **Matryoshka Representation Learning**, _Kusupati et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2205.13147)\]\[[code](https://github.com/RAIVNLab/MRL)\]
- BGE: **C-Pack: Packaged Resources To Advance General Chinese Embedding**, _Xiao et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.07597)\]\[[code](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/baai_general_embedding)\]\[[FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)\]
- LLM-Embedder: **Retrieve Anything To Augment Large Language Models**, _Zhang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.07554)\]\[[code](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_embedder)\]\[[FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)\]
- **BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation**, _Chen et al._, arxiv 2024. \[[paper](https://export.arxiv.org/abs/2402.03216)\]\[[code](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/BGE_M3)\]\[[FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)\]
- \[[m3e-base](https://huggingface.co/moka-ai/m3e-base)\]
- **Jina Embeddings 2: 8192-Token General-Purpose Text Embeddings for Long Documents**, _Günther et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.19923)\]\[[model](https://huggingface.co/jinaai/jina-embeddings-v2-base-en)\]
- GTE: **Towards General Text Embeddings with Multi-stage Contrastive Learning**, _Li et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2308.03281)\]\[[model](https://huggingface.co/thenlper/gte-large-zh)\]
- \[[BCEmbedding](https://github.com/netease-youdao/BCEmbedding)\]\[[bce-embedding-base_v1](https://huggingface.co/maidalun1020/bce-embedding-base_v1)\]\[[bce-reranker-base_v1](https://huggingface.co/maidalun1020/bce-reranker-base_v1)\]
- \[[CohereV3](https://huggingface.co/Cohere/Cohere-embed-multilingual-v3.0)\]
- **One Embedder, Any Task: Instruction-Finetuned Text Embeddings**, _Su et al._, ACL 2023. \[[paper](https://arxiv.org/abs/2212.09741)\]\[[code](https://github.com/xlang-ai/instructor-embedding)\]
- E5: **Improving Text Embeddings with Large Language Models**, _Wang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.00368)\]\[[code](https://github.com/microsoft/unilm/tree/master/e5)\]\[[model](https://huggingface.co/intfloat/e5-mistral-7b-instruct)\]\[[llm2vec](https://github.com/McGill-NLP/llm2vec)\]
- **Nomic Embed: Training a Reproducible Long Context Text Embedder**, _Nussbaum et al._, Nomic AI 2024. \[[paper](https://arxiv.org/abs/2402.01613)\]\[[code](https://github.com/nomic-ai/contrastors)\]
- GritLM: **Generative Representational Instruction Tuning**, _Muennighoff et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.09906)\]\[[code](https://github.com/ContextualAI/gritlm)\]

##### 3.3.11 Reasoning and Planning

- Few-Shot-CoT: **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models**, _Wei et al._, NeurIPS 2022. \[[paper](https://arxiv.org/abs/2201.11903)\]\[[chain-of-thought-hub](https://github.com/FranxYao/chain-of-thought-hub)\]
- **Self-Consistency Improves Chain of Thought Reasoning in Language Models**, _Wang et al._, ICLR 2023. \[[paper](https://arxiv.org/abs/2203.11171)\]
- Zero-Shot-CoT: **Large Language Models are Zero-Shot Reasoners**, _Kojima et al._, NeurIPS 2022. \[[paper](https://arxiv.org/abs/2205.11916)\]\[[code](https://github.com/kojima-takeshi188/zero_shot_cot)\]
- Auto-CoT: **Automatic Chain of Thought Prompting in Large Language Models**, _Zhang et al._, ICLR 2023. \[[paper](https://arxiv.org/abs/2210.03493)\]\[[code](https://github.com/amazon-science/auto-cot)\]
- **Multimodal Chain-of-Thought Reasoning in Language Models**, _Zhang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2302.00923)\]\[[code](https://github.com/amazon-science/mm-cot)\]
- **Chain-of-Thought Reasoning Without Prompting**, _Wang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.10200)\]
- **ReAct: Synergizing Reasoning and Acting in Language Models**, _Yao et al._, ICLR 2023. \[[paper](https://arxiv.org/abs/2210.03629)\]\[[code](https://github.com/ysymyth/ReAct)\]
- **MM-REACT: Prompting ChatGPT for Multimodal Reasoning and Action**, _Yang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2303.11381)\]\[[code](https://github.com/microsoft/MM-REACT)\]
- **Tree of Thoughts: Deliberate Problem Solving with Large Language Models**, _Yao et al._, NeurIPS 2023. \[[paper](https://arxiv.org/abs/2305.10601)\]\[[code](https://github.com/princeton-nlp/tree-of-thought-llm)\]\[[Plug in and Play Implementation](https://github.com/kyegomez/tree-of-thoughts)\]\[[tree-of-thought-prompting](https://github.com/dave1010/tree-of-thought-prompting)\]
- **Graph of Thoughts: Solving Elaborate Problems with Large Language Models**, _Besta et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2308.09687)\]\[[code](https://github.com/spcl/graph-of-thoughts)\]
- **Cumulative Reasoning with Large Language Models**, _Zhang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2308.04371)\]\[[code](https://github.com/iiis-ai/cumulative-reasoning)\]
- **Algorithm of Thoughts: Enhancing Exploration of Ideas in Large Language Models**, _Sel et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2308.10379)\]\[[unofficial code](https://github.com/kyegomez/Algorithm-Of-Thoughts)\]
- **Everything of Thoughts: Defying the Law of Penrose Triangle for Thought Generation**, _Ding et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.04254)\]\[[code](https://github.com/microsoft/Everything-of-Thoughts-XoT)\]
- **Diffusion of Thoughts: Chain-of-Thought Reasoning in Diffusion Language Models**, _Ye et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.07754)\]\[[code](https://github.com/HKUNLP/diffusion-of-thoughts)\]
- **Least-to-Most Prompting Enables Complex Reasoning in Large Language Models**, _Zhou et al._, ICLR 2023. \[[paper](https://arxiv.org/abs/2205.10625)\]
- DEPS: **Describe, Explain, Plan and Select: Interactive Planning with Large Language Models Enables Open-World Multi-Task Agents**, _Wang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2302.01560)\]\[[code](https://github.com/CraftJarvis/MC-Planner)\]
- RAP: **Reasoning with Language Model is Planning with World Model**, _Hao et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.14992)\]\[[code](https://github.com/Ber666/llm-reasoners)\]
- LEMA: **Learning From Mistakes Makes LLM Better Reasoner**, _An et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.20689)\]\[[code](https://github.com/microsoft/LEMA)\]
- **Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks**, _Chen et al._, TMLR 2023. \[[paper](https://arxiv.org/abs/2211.12588)\]\[[code](https://github.com/wenhuchen/Program-of-Thoughts)\]
- **Chain of Code: Reasoning with a Language Model-Augmented Code Emulator**, _Li et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.04474)\]\[[code]\]
- **The Impact of Reasoning Step Length on Large Language Models**, _Jin et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.04925)\]\[[code](https://github.com/jmyissb/The-Impact-of-Reasoning-Step-Length-on-Large-Language-Models)\]
- **Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models**, _Wang et al._, ACL 2023. \[[paper](https://arxiv.org/abs/2305.04091)\]\[[code](https://github.com/AGI-Edgerunners/Plan-and-Solve-Prompting)\]
- **Improving Factuality and Reasoning in Language Models through Multiagent Debate**, _Du et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.14325)\]\[[code](https://github.com/composable-models/llm_multiagent_debate)\]\[[Multi-Agents-Debate](https://github.com/Skytliang/Multi-Agents-Debate)\]
- **Self-Refine: Iterative Refinement with Self-Feedback**, _Madaan et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2303.17651)\]\[[code](https://github.com/madaan/self-refine)\]
- **CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing**, _Gou et al._, ICLR 2024. \[[paper](https://arxiv.org/abs/2305.11738)\]\[[code](https://github.com/microsoft/ProphetNet/tree/master/CRITIC)\]
- **Self-Discover: Large Language Models Self-Compose Reasoning Structures**, _Zhou et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.03620)\]\[[unofficial implementation](https://github.com/catid/self-discover)\]\[[SELF-DISCOVER](https://github.com/kailashsp/SELF-DISCOVER)\]
- **RAT: Retrieval Augmented Thoughts Elicit Context-Aware Reasoning in Long-Horizon Generation**, _Wang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.05313)\]\[[code](https://github.com/CraftJarvis/RAT)\]
- **KnowAgent: Knowledge-Augmented Planning for LLM-Based Agents**, _Zhu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.03101)\]\[[code](https://github.com/zjunlp/KnowAgent)\]\[[KnowLM](https://github.com/zjunlp/KnowLM)\]
- **Advancing LLM Reasoning Generalists with Preference Trees**, _Yuan et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2404.02078)\]\[[code](https://github.com/OpenBMB/Eurus)\]

- ReST-EM: **Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models**, _Singh et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.06585)\]\[[unofficial code](https://github.com/lucidrains/ReST-EM-pytorch)\]
- **ReST meets ReAct: Self-Improvement for Multi-Step Reasoning LLM Agent**, _Aksitov et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.10003)\]\[[code]\]
- **Orca 2: Teaching Small Language Models How to Reason**, _Mitra et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.11045)\]\[[code]\]
- Searchformer: **Beyond A*: Better Planning with Transformers via Search Dynamics Bootstrapping**, _Lehnert et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.14083)\]
- **How Far Are We from Intelligent Visual Deductive Reasoning?**, _Zhang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.04732)\]\[[code](https://github.com/apple/ml-rpm-bench)\]

###### Survey
- \[[Prompt4ReasoningPapers](https://github.com/zjunlp/Prompt4ReasoningPapers)\]


#### 3.4 LLM Theory

- **Scaling Laws for Neural Language Models**, _Kaplan et al._, arxiv 2020. \[[paper](https://arxiv.org/abs/2001.08361)\]\[[unofficial code](https://github.com/shehper/scaling_laws)\]
- **Emergent Abilities of Large Language Models**, _Wei et al._, TMRL 2022. \[[paper](https://arxiv.org/abs/2206.07682)\]
- Chinchilla: **Training Compute-Optimal Large Language Models**, _Hoffmann et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2203.15556)\]
- **Scaling Laws for Autoregressive Generative Modeling**, _Henighan et al._, arxiv 2020. \[[paper](https://arxiv.org/abs/2010.14701)\]
- **Are Emergent Abilities of Large Language Models a Mirage**, _Schaeffer et al._, NeurIPS 2023 Outstanding Paper. \[[paper](https://arxiv.org/abs/2304.15004)\]
- S2A: **System 2 Attention (is something you might need too)**, _Weston et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.11829)\]
- **Scaling Laws for Downstream Task Performance of Large Language Models**, _Isik et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.04177)\]
- **Scalable Pre-training of Large Autoregressive Image Models**, _El-Nouby et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.08541)\]\[[code](https://github.com/apple/ml-aim)\]
- **When Scaling Meets LLM Finetuning: The Effect of Data, Model and Finetuning Method**, _Zhang et al._, ICLR 2024. \[[paper](https://arxiv.org/abs/2402.17193)\]
- **Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws**, _Allen-Zhu et al_, arxiv 2024. \[[paper](https://arxiv.org/abs/2404.05405)\]
- **Language Modeling Is Compression**, _Delétang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.10668)\]
- **Language Models Represent Space and Time**, _Gurnee et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.02207)\]\[[code](https://github.com/wesg52/world-models)\]

- **Language models can explain neurons in language models**, _OpenAI_, 2023. \[[blog](https://openai.com/research/language-models-can-explain-neurons-in-language-models)\]\[[code](https://github.com/openai/automated-interpretability)\]\[[transformer-debugger](https://github.com/openai/transformer-debugger)\]
- **Towards Monosemanticity: Decomposing Language Models With Dictionary Learning**, _Anthropic_, 2023. \[[blog](https://www.anthropic.com/news/towards-monosemanticity-decomposing-language-models-with-dictionary-learning)\]
- **Usable XAI: 10 Strategies Towards Exploiting Explainability in the LLM Era**, _Wu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.08946)\]\[[code](https://github.com/JacksonWuxs/UsableXAI_LLM)\]
- **LM Transparency Tool: Interactive Tool for Analyzing Transformer Language Models**, _Tufanov et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2404.07004)\]\[[code](https://github.com/facebookresearch/llm-transparency-tool)\]

- ROME: **Locating and Editing Factual Associations in GPT**, _Meng et al._, NeurIPS 2022. \[[paper](https://arxiv.org/abs/2202.05262)\]\[[code](https://github.com/kmeng01/rome)\]\[[FastEdit](https://github.com/hiyouga/FastEdit)\]
- **Editing Large Language Models: Problems, Methods, and Opportunities**, _Yao et al._, EMNLP 2023. \[[paper](https://arxiv.org/abs/2305.13172)\]\[[code](https://github.com/zjunlp/EasyEdit)\]
- **A Comprehensive Study of Knowledge Editing for Large Language Models**, _Zhang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.01286)\]\[[code](https://github.com/zjunlp/EasyEdit)\]

#### 3.5 Chinese Model

- \[[Awesome-Chinese-LLM](https://github.com/HqWu-HITCS/Awesome-Chinese-LLM)\]\[[awesome-LLMs-In-China](https://github.com/wgwang/awesome-LLMs-In-China)\]
- **GLM: General Language Model Pretraining with Autoregressive Blank Infilling**, _Du et al._, ACL 2022. \[[paper](https://arxiv.org/abs/2103.10360)\]\[[code](https://github.com/THUDM/GLM)\]\[[ChatGLM3](https://github.com/THUDM/ChatGLM3)\]\[[AgentTuning](https://github.com/THUDM/AgentTuning)\]
- **GLM-130B: An Open Bilingual Pre-trained Model**, _Zeng et al._, ICLR 2023. \[[paper](https://arxiv.org/abs/2210.02414v2)\]\[[code](https://github.com/THUDM/GLM-130B/)\]
- **Baichuan 2: Open Large-scale Language Models**, _Yang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.10305)\]\[[code](https://github.com/baichuan-inc/Baichuan2)\]
- **Qwen Technical Report**, _Bai et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.16609)\]\[[code](https://github.com/QwenLM/Qwen)\]\[[Qwen1.5](https://github.com/QwenLM/Qwen1.5)\]\[[Qwen-Agent](https://github.com/QwenLM/Qwen-Agent)\]
- **Yi: Open Foundation Models by 01.AI**, _Young et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.04652)\]\[[code](https://github.com/01-ai/Yi)\]
- **InternLM2 Technical Report**, _Cai et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.17297)\]\[[code](https://github.com/InternLM/InternLM)\]
- **DeepSeek LLM: Scaling Open-Source Language Models with Longtermism**, _Bi et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.02954)\]\[[DeepSeek-LLM](https://github.com/deepseek-ai/DeepSeek-LLM)\]\[[DeepSeek-Coder)](https://github.com/deepseek-ai/DeepSeek-Coder)\]
- **TeleChat Technical Report**, _Wang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.03804)\]\[[code](https://github.com/Tele-AI/Telechat)\]\[[Tele-FLM](https://huggingface.co/CofeAI/Tele-FLM)\]
- **Efficient and Effective Text Encoding for Chinese LLaMA and Alpaca**, Cui et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2304.08177)\]\[[code](https://github.com/ymcui/Chinese-LLaMA-Alpaca)\]\[[Chinese-LLaMA-Alpaca-2](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2)\]\[[Chinese-LLaMA-Alpaca-3](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3)\]\[[baby-llama2-chinese](https://github.com/DLLXW/baby-llama2-chinese)\]
- **Rethinking Optimization and Architecture for Tiny Language Models**, _Tang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.02791)\]\[[code](https://github.com/YuchuanTian/RethinkTinyLM)\]
- \[[MOSS](https://github.com/OpenLMLab/MOSS)\]\[[MOSS-RLHF](https://github.com/OpenLMLab/MOSS-RLHF)\]
- \[[MiniCPM](https://github.com/OpenBMB/MiniCPM)\]\[[Skywork](https://github.com/SkyworkAI/Skywork)\]\[[Orion](https://github.com/OrionStarAI/Orion)\]\[[BELLE](https://github.com/LianjiaTech/BELLE)\]\[[Yuan-2.0](https://github.com/IEIT-Yuan/Yuan-2.0)\]\[[Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM)\]
- \[[LlamaFamily/Llama-Chinese](https://github.com/LlamaFamily/Llama-Chinese)\]\[[LinkSoul-AI/Chinese-Llama-2-7b](https://github.com/LinkSoul-AI/Chinese-Llama-2-7b)\]\[[llama3-Chinese-chat](https://github.com/CrazyBoyM/llama3-Chinese-chat)\]\[[phi3-Chinese](https://github.com/CrazyBoyM/phi3-Chinese)\]
- \[[Firefly](https://github.com/yangjianxin1/Firefly)\]\[[GPT2-chitchat](https://github.com/yangjianxin1/GPT2-chitchat)\]
- Alpaca-CoT: **An Empirical Study of Instruction-tuning Large Language Models in Chinese**, _Si et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.07328)\]\[[code](https://github.com/PhoebusSi/Alpaca-CoT)\]

<details>
---
<details>
<summary>CV</summary>

## CV

- **CS231n: Deep Learning for Computer Vision** \[[link](http://cs231n.stanford.edu/)\]

### 1. Basic for CV
- AlexNet: **ImageNet Classification with Deep Convolutional Neural Networks**, _Krizhevsky et al._, NIPS 2012. \[[paper](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)\]
- VGG: **Very Deep Convolutional Networks for Large-Scale Image Recognition**, _Simonyan et al._, ICLR 2015. \[[paper](https://arxiv.org/abs/1409.1556)\]
- GoogLeNet: **Going Deeper with Convolutions**, _Szegedy et al._, CVPR 2015. \[[paper](https://arxiv.org/abs/1409.4842)\]
- ResNet: **Deep Residual Learning for Image Recognition**, _He et al._, CVPR 2016 Best Paper. \[[paper](https://arxiv.org/abs/1512.03385)\]\[[code](https://github.com/KaimingHe/deep-residual-networks)\]
- DenseNet: **Densely Connected Convolutional Networks**, _Huang et al._, CVPR 2017 Oral. \[[paper](https://arxiv.org/abs/1608.06993)\]\[[code](https://github.com/liuzhuang13/DenseNet)\]
- **EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks**, _Tan et al._, ICML 2019. \[[paper](https://arxiv.org/abs/1905.11946)\]\[[code](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)\]\[[EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)\]
- BYOL: **Bootstrap your own latent: A new approach to self-supervised Learning**, _Grill et al._, arxiv 2020. \[[paper](https://arxiv.org/abs/2006.07733)\]\[[code](https://github.com/google-deepmind/deepmind-research/tree/master/byol)\]\[[byol-pytorch](https://github.com/lucidrains/byol-pytorch)\]

### 2. Contrastive Learning

- MoCo: **Momentum Contrast for Unsupervised Visual Representation Learning**, _He et al._, CVPR 2020. \[[paper](https://arxiv.org/abs/1911.05722)\]\[[code](https://github.com/facebookresearch/moco)\]
- SimCLR: **A Simple Framework for Contrastive Learning of Visual Representations**, _Chen et al._, PMLR 2020. \[[paper](https://arxiv.org/abs/2002.05709)\]\[[code](https://github.com/google-research/simclr)\]
- **DINOv2: Learning Robust Visual Features without Supervision**, _Oquab et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2304.07193)\]\[[code](https://github.com/facebookresearch/dinov2)\]
- **FeatUp: A Model-Agnostic Framework for Features at Any Resolution**, _Fu et al._, ICLR 2024. \[[paper](https://arxiv.org/abs/2403.10516v1)\]\[[code](https://github.com/mhamilton723/FeatUp)\]

- InfoNCE Loss: **Representation Learning with Contrastive Predictive Coding**, _Oord et al._, arxiv 2018. \[[paper](https://arxiv.org/abs/1807.03748)\]\[[unofficial code](https://github.com/jefflai108/Contrastive-Predictive-Coding-PyTorch)\]

### 3. CV Application

- **NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis**, _Mildenhall et al._, ECCV 2020. \[[paper](https://arxiv.org/abs/2003.08934)\]\[[code](https://github.com/bmild/nerf)\]\[[nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch)\]\[[NeRF-Factory](https://github.com/kakaobrain/NeRF-Factory)\]
- GFP-GAN: **Towards Real-World Blind Face Restoration with Generative Facial Prior**, _Wang et al._, CVPR 2021. \[[paper](https://arxiv.org/abs/2101.04061)\]\[[code](https://github.com/TencentARC/GFPGAN)\]
- CodeFormer: **Towards Robust Blind Face Restoration with Codebook Lookup Transformer**, _Zhou et al._, NeurIPS 2022. \[[paper](https://arxiv.org/abs/2206.11253)\]\[[code](https://github.com/sczhou/CodeFormer)\]\[[APISR](https://github.com/Kiteretsu77/APISR)\]
- **BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers**, _Li et al._, ECCV 2022. \[[paper](https://arxiv.org/abs/2203.17270)\]\[[code](https://github.com/fundamentalvision/BEVFormer)\]\[[occupancy_networks](https://github.com/autonomousvision/occupancy_networks)\]\[[VoxFormer](https://github.com/NVlabs/VoxFormer)\]\[[TPVFormer](https://github.com/wzzheng/TPVFormer)\]
- UniAD: **Planning-oriented Autonomous Driving**, _Hu et al._, CVPR 2023 Best Paper. \[[paper](https://arxiv.org/abs/2212.10156)\]\[[code](https://github.com/OpenDriveLab/UniAD)\]

- **FaceChain: A Playground for Identity-Preserving Portrait Generation**, _Liu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2308.14256)\]\[[code](https://github.com/modelscope/facechain)\]
- MGIE: **Guiding Instruction-based Image Editing via Multimodal Large Language Models**, _Fu et al._, ICLR 2024 Spotlight. \[[paper](https://arxiv.org/abs/2309.17102)\]\[[code](https://github.com/apple/ml-mgie)\]
- **PhotoMaker: Customizing Realistic Human Photos via Stacked ID Embedding**, _Li et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.04461)\]\[[code](https://github.com/TencentARC/PhotoMaker)\]
- **InstantID: Zero-shot Identity-Preserving Generation in Seconds**, _Wang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.07519)\]\[[code](https://github.com/InstantID/InstantID)\]\[[InstantStyle](https://github.com/InstantStyle/InstantStyle)\]
- **ReplaceAnything as you want: Ultra-high quality content replacement**, \[[link](https://aigcdesigngroup.github.io/replace-anything/)\]\[[IDM-VTON](https://github.com/yisol/IDM-VTON)\]
- LayerDiffusion: **Transparent Image Layer Diffusion using Latent Transparency**, _Zhang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.17113)\]\[[code](https://github.com/layerdiffusion/LayerDiffusion)\]\[[sd-forge-layerdiffusion](https://github.com/layerdiffusion/sd-forge-layerdiffusion)\]\[[IC-Light](https://github.com/lllyasviel/IC-Light)\]

- \[[deepfakes/faceswap](https://github.com/deepfakes/faceswap)\]\[[DeepFaceLab](https://github.com/iperov/DeepFaceLab)\]\[[DeepFaceLive](https://github.com/iperov/DeepFaceLive)\]
- \[[IOPaint](https://github.com/Sanster/IOPaint)\]\[[SPADE](https://github.com/NVlabs/SPADE)\]\[[EasyOCR](https://github.com/JaidedAI/EasyOCR)\]

### 4. Foundation Model

- ViT: **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**, _Dosovitskiy et al._, ICLR 2021. \[[paper](https://arxiv.org/abs/2010.11929)\]\[[code](https://github.com/google-research/vision_transformer)\]\[[Pytorch Implementation](https://github.com/lucidrains/vit-pytorch)\]\[[efficientvit](https://github.com/mit-han-lab/efficientvit)\]\[[ViT-Adapter](https://github.com/czczup/ViT-Adapter)\]
- ViT-Adapter: **Vision Transformer Adapter for Dense Predictions**, _Chen et al._, ICLR 2023 Spotlight. \[[paper](https://arxiv.org/abs/2205.08534)\]\[[code](https://github.com/czczup/ViT-Adapter)\]
- DeiT: **Training data-efficient image transformers & distillation through attention**, _Touvron et al._, ICML 2021. \[[paper](https://arxiv.org/abs/2012.12877)\]\[[code](https://github.com/facebookresearch/deit)\]
- **ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision**, _Kim et al._, ICML 2021. \[[paper](https://arxiv.org/abs/2102.03334)\]\[[code](https://github.com/dandelin/vilt)\]
- **Swin Transformer: Hierarchical Vision Transformer using Shifted Windows**, _Liu et al._, ICCV 2021. \[[paper](https://arxiv.org/abs/2103.14030)\]\[[code](https://github.com/microsoft/Swin-Transformer)\]
- MAE: **Masked Autoencoders Are Scalable Vision Learners**, _He et al._, CVPR 2022. \[[paper](https://arxiv.org/abs/2111.06377)\]\[[code](https://github.com/facebookresearch/mae)\]
- LVM: **Sequential Modeling Enables Scalable Learning for Large Vision Models**, _Bai et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.00785)\]\[[code](https://github.com/ytongbai/LVM)\]
- GLEE: **General Object Foundation Model for Images and Videos at Scale**, _Wu wt al._, CVPR 2024. \[[paper](https://arxiv.org/abs/2312.09158)\]\[[code](https://github.com/FoundationVision/GLEE)\]
- **Tokenize Anything via Prompting**, _Pan et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.09128)\]\[[code](https://github.com/baaivision/tokenize-anything)\]
- **Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model** _Zhu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.09417)\]\[[code](https://github.com/hustvl/Vim)\]\[[VMamba](https://github.com/MzeroMiko/VMamba)\]
- **Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data**, _Yang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.10891)\]\[[code](https://github.com/LiheYoung/Depth-Anything)\]
- **Vision Superalignment: Weak-to-Strong Generalization for Vision Foundation Models**, _Guo et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.03749)\]\[[code](https://github.com/ggjy/vision_weak_to_strong)\]

- \[[pytorch-image-models](https://github.com/huggingface/pytorch-image-models)\]\[[Pointcept](https://github.com/Pointcept/Pointcept)\]

### 5. Generative Model (GAN and VAE)

- GAN: **Generative Adversarial Networks**, _Goodfellow et al._, arxiv 2014. \[[paper](https://arxiv.org/abs/1406.2661)\]\[[code](https://github.com/goodfeli/adversarial)\]\[[Pytorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN)\]
- StyleGAN3: **Alias-Free Generative Adversarial Networks**, _Karras etal._, NeurIPS 2021. \[[paper](https://arxiv.org/abs/2106.12423)\]\[[code](https://github.com/NVlabs/stylegan3)\]
- GigaGAN: **Scaling up GANs for Text-to-Image Synthesis**, _Kang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2303.05511)\]\[[code](https://github.com/lucidrains/gigagan-pytorch)\]
- \[[pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)\]\[[img2img-turbo](https://github.com/GaParmar/img2img-turbo)\]
- VAE: **Auto-Encoding Variational Bayes**, _Kingma et al._, arxiv 2013. \[[paper](https://arxiv.org/abs/1312.6114)\]\[[code](https://github.com/jaanli/variational-autoencoder)\]\[[Pytorch-VAE](https://github.com/AntixK/PyTorch-VAE)\]
- VQ-VAE: **Neural Discrete Representation Learning**, _Oord et al._, NIPS 2017. \[[paper](https://arxiv.org/abs/1711.00937)\]\[[code](https://github.com/AntixK/PyTorch-VAE/blob/master/models/vq_vae.py)\]\[[vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch)\]
- VQ-VAE-2: **Generating Diverse High-Fidelity Images with VQ-VAE-2**, _Razavi et al._, arxiv 2019. \[[paper](https://arxiv.org/abs/1906.00446)\]\[[code](https://github.com/rosinality/vq-vae-2-pytorch)\]
- VQGAN: **Taming Transformers for High-Resolution Image Synthesis**, _Esser et al._, CVPR 2021. \[[paper](https://arxiv.org/abs/2012.09841)\]\[[code](https://github.com/CompVis/taming-transformers)\]
- **Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction**, _Tian et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2404.02905)\]\[[code](https://github.com/FoundationVision/VAR)\]

### 6. Image Editing

- **InstructPix2Pix: Learning to Follow Image Editing Instructions**, _Brooks et al._, CVPR 2023 Highlight. \[[paper](https://arxiv.org/abs/2211.09800)\]\[[code](https://github.com/timothybrooks/instruct-pix2pix)\]
- **Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold**, _Pan et al._, SIGGRAPH 2023. \[[paper](https://arxiv.org/abs/2305.10973)\]\[[code](https://github.com/XingangPan/DragGAN)\]
- **DragDiffusion: Harnessing Diffusion Models for Interactive Point-based Image Editing**, _Shi et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2306.14435)\]\[[code](https://github.com/Yujun-Shi/DragDiffusion)\]
- **DragonDiffusion: Enabling Drag-style Manipulation on Diffusion Models**, _Mou et al._, ICLR 2024 Spolight. \[[paper](https://arxiv.org/abs/2307.02421)\]\[[code](https://github.com/MC-E/DragonDiffusion)\]
- **LEDITS++: Limitless Image Editing using Text-to-Image Models**, _Brack et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.16711)\]\[[code](https://huggingface.co/spaces/editing-images/leditsplusplus/tree/main)\]\[[demo](https://huggingface.co/spaces/editing-images/leditsplusplus)\]
- **Diffusion Model-Based Image Editing: A Survey**, _Huang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.17525)\]\[[code](https://github.com/SiatMMLab/Awesome-Diffusion-Model-Based-Image-Editing-Methods)\]

### 7. Object Detection

- DETR: **End-to-End Object Detection with Transformers**, _Carion et al._, arxiv 2020. \[[paper](https://arxiv.org/abs/2005.12872)\]\[[code](https://github.com/facebookresearch/detr)\]
- Focus-DERT: **Less is More_Focus Attention for Efficient DETR**, _Zheng et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2307.12612)\]\[[code](https://github.com/huawei-noah/noah-research)\]
- **U2-Net_Going Deeper with Nested U-Structure for Salient Object Detection**, _Qin et al._, arxiv 2020. \[[paper](https://arxiv.org/abs/2005.09007)\]\[[code](https://github.com/xuebinqin/U-2-Net)\]
- YOLO: **You Only Look Once: Unified, Real-Time Object Detection** _Redmon et al._, arxiv 2015. \[[paper](https://arxiv.org/abs/1506.02640)\]
- **YOLOX: Exceeding YOLO Series in 2021**, _Ge et al._, arxiv 2021. \[[paper](https://arxiv.org/abs/2107.08430)\]\[[code](https://github.com/Megvii-BaseDetection/YOLOX)\]
- **Gold-YOLO: Efficient Object Detector via Gather-and-Distribute Mechanism**, _Wang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.11331)\]\[[code](https://github.com/huawei-noah/Efficient-Computing/tree/master/Detection/Gold-YOLO)\]
- **YOLO-World: Real-Time Open-Vocabulary Object Detection**, _Cheng et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.17270)\]\[[code](https://github.com/ailab-cvc/yolo-world)\]
- **YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information**, _Wang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.13616)\]\[[code](https://github.com/WongKinYiu/yolov9)\]
- **T-Rex2: Towards Generic Object Detection via Text-Visual Prompt Synergy**, _Jiang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.14610)\]\[[code](https://github.com/IDEA-Research/T-Rex)\]

- \[[detectron2](https://github.com/facebookresearch/detectron2)\]\[[yolov5](https://github.com/ultralytics/yolov5)\]\[[mmdetection](https://github.com/open-mmlab/mmdetection)\]\[[detrex](https://github.com/IDEA-Research/detrex)\]

### 8. Semantic Segmentation

- **U-Net: Convolutional Networks for Biomedical Image Segmentation**, _Ronneberger et al._, MICCAI 2015. \[[paper](https://arxiv.org/abs/1505.04597)\]\[[code](https://github.com/milesial/Pytorch-UNet)\]
- **Segment Anything**, _Kirillov et al._, ICCV 2023. \[[paper](https://arxiv.org/abs/2304.02643)\]\[[code](https://github.com/facebookresearch/segment-anything)\]
- **EfficientSAM: Leveraged Masked Image Pretraining for Efficient Segment Anything**, _Xiong et al._, CVPR 2024. \[[paper](https://arxiv.org/abs/2312.00863)\]\[[code](https://github.com/yformer/EfficientSAM)\]
- **Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks**, _Ren et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.14159)\]\[[code](https://github.com/IDEA-Research/Grounded-Segment-Anything)\]

- \[[mmsegmentation](https://github.com/open-mmlab/mmsegmentation)\]\[[mmdeploy](https://github.com/open-mmlab/mmdeploy)\]

### 9. Video

- **VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training**, _Tong et al._, NeurIPS 2022 Spotlight. \[[paper](https://arxiv.org/abs/2203.12602)\]\[[code](https://github.com/MCG-NJU/VideoMAE)\]
- **MagicVideo-V2: Multi-Stage High-Aesthetic Video Generation**, _Wang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.04468)\]
- \[[V-JEPA](https://github.com/facebookresearch/jepa)\]\[[I-JEPA](https://github.com/facebookresearch/ijepa)\]
- **VideoMamba: State Space Model for Efficient Video Understanding**, _Li et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.06977)\]\[[code](https://github.com/OpenGVLab/VideoMamba)\]
- **VideoChat: Chat-Centric Video Understanding**, _Li et al._, CVPR 2024 Highlight. \[[paper](https://arxiv.org/abs/2305.06355)\]\[[code](https://github.com/OpenGVLab/Ask-Anything)\]

### 10. Survey for CV

- **ConvNet vs Transformer, Supervised vs CLIP: Beyond ImageNet Accuracy**, _Vishniakov et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.09215)\]\[[code](https://github.com/kirill-vish/Beyond-INet)\]
- **Parameter-Efficient Fine-Tuning for Pre-Trained Vision Models: A Survey**, _Xin et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.02242)\]\[[code](https://github.com/synbol/Awesome-Parameter-Efficient-Transfer-Learning)\]

<details>
---
<details>
<summary>Multimodal</summary>

## Multimodal

### 1. Audio

- Whisper: **Robust Speech Recognition via Large-Scale Weak Supervision**, _Radford et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2212.04356)\]\[[code](https://github.com/openai/whisper)\]\[[whisper.cpp](https://github.com/ggerganov/whisper.cpp)\]\[[faster-whisper](https://github.com/SYSTRAN/faster-whisper)\]\[[WhisperFusion](https://github.com/collabora/WhisperFusion)\]
- **WhisperX: Time-Accurate Speech Transcription of Long-Form Audio**, _Bain et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2303.00747)\]\[[code](https://github.com/m-bain/whisperX)\]
- **Distil-Whisper: Robust Knowledge Distillation via Large-Scale Pseudo Labelling**，_Gandhi et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.00430)\]\[[code](https://github.com/huggingface/distil-whisper)\]
- **Speculative Decoding for 2x Faster Whisper Inference**, _Sanchit Gandhi_, HuggingFace Blog 2023. \[[blog](https://huggingface.co/blog/whisper-speculative-decoding)\]\[[paper](https://arxiv.org/abs/2211.17192)\]
- VALL-E: **Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers**, _Wang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2301.02111)\]\[[code](https://github.com/microsoft/unilm/tree/master/valle)\]
- VALL-E-X: **Speak Foreign Languages with Your Own Voice: Cross-Lingual Neural Codec Language Modeling**, _Zhang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2303.03926)\]\[[code](https://github.com/microsoft/unilm/tree/master/valle)\]
- **Seamless: Multilingual Expressive and Streaming Speech Translation**, _Seamless Communication et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.05187)\]\[[code](https://github.com/facebookresearch/seamless_communication)\]\[[audiocraft](https://github.com/facebookresearch/audiocraft)\]
- **SeamlessM4T: Massively Multilingual & Multimodal Machine Translation**, _Seamless Communication et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2308.11596)\]\[[code](https://github.com/facebookresearch/seamless_communication)\]
- **StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models**, _Li et al._, NeurIPS 2023. \[[paper](https://arxiv.org/abs/2306.07691)\]\[[code](https://github.com/yl4579/StyleTTS2)\]
- **Amphion: An Open-Source Audio, Music and Speech Generation Toolkit**, _Zhang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.09911)\]\[[code](https://github.com/open-mmlab/Amphion)\]
- VITS: **Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech**, _Kim et al._, ICML 2021. \[[paper](https://arxiv.org/abs/2106.06103)\]\[[code](https://github.com/jaywalnut310/vits)\]\[[Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)\]\[[so-vits-svc-fork](https://github.com/voicepaw/so-vits-svc-fork)\]\[[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)\]\[[VITS-fast-fine-tuning](https://github.com/Plachtaa/VITS-fast-fine-tuning)\]
- **OpenVoice: Versatile Instant Voice Cloning**, _Qin et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.01479)\]\[[code](https://github.com/myshell-ai/OpenVoice)\]\[[MockingBird](https://github.com/babysor/MockingBird)\]\[[clone-voice](https://github.com/jianchang512/clone-voice)\]
- **NaturalSpeech 3: Zero-Shot Speech Synthesis with Factorized Codec and Diffusion Models**, _Ju et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.03100)\]
- **VoiceCraft: Zero-Shot Speech Editing and Text-to-Speech in the Wild**, _Peng et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.16973)\]\[[code](https://github.com/jasonppy/voicecraft)\]
- **WavLLM: Towards Robust and Adaptive Speech Large Language Model**, _Hu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2404.00656)\]\[[code](https://github.com/microsoft/SpeechT5/tree/main/WavLLM)\]

- **Github Repositories**
- \[[coqui-ai/TTS](https://github.com/coqui-ai/TTS)\]\[[suno-ai/bark](https://github.com/suno-ai/bark)\]\[[WhisperSpeech](https://github.com/collabora/WhisperSpeech)\]\[[MeloTTS](https://github.com/myshell-ai/MeloTTS)\]\[[parler-tts](https://github.com/huggingface/parler-tts)\]
- [https://github.com/netease-youdao/EmotiVoice](https://github.com/netease-youdao/EmotiVoice)
- [https://github.com/pyannote/pyannote-audio](https://github.com/pyannote/pyannote-audio)
- [https://github.com/alibaba-damo-academy/FunASR](https://github.com/alibaba-damo-academy/FunASR)
- [https://github.com/OpenTalker/video-retalking](https://github.com/OpenTalker/video-retalking)
- [https://github.com/Rudrabha/Wav2Lip](https://github.com/Rudrabha/Wav2Lip)
- [https://github.com/Zz-ww/SadTalker-Video-Lip-Sync](https://github.com/Zz-ww/SadTalker-Video-Lip-Sync)
- [https://github.com/OpenTalker/SadTalker](https://github.com/OpenTalker/SadTalker)
- \[[Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)\]

### 2. Blip

- ALBEF: **Align before Fuse: Vision and Language Representation Learning with Momentum Distillation**, _Li et al._, NeurIPS 2021. \[[paper](https://arxiv.org/abs/2107.07651)\]\[[code](https://github.com/salesforce/ALBEF)\]
- **BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation**, _Li et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2201.12086)\]\[[code](https://github.com/salesforce/BLIP)\]
- **BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models**, _Li et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2301.12597)\]\[[code](https://github.com/salesforce/LAVIS/tree/main/projects/blip2)\]
- **InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning**, _Dai et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.06500)\]\[[code](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip)\]
- **X-InstructBLIP: A Framework for aligning X-Modal instruction-aware representations to LLMs and Emergent Cross-modal Reasoning**, _Panagopoulou et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.18799)\]\[[code](https://github.com/artemisp/LAVIS-XInstructBLIP)\]
- **LAVIS: A Library for Language-Vision Intelligence**, _Li et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2209.09019)\]\[[code](https://github.com/salesforce/LAVIS)\]
- **VLMo: Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts**, _Bao et al._, NeurIPS 2022. \[[paper](https://arxiv.org/abs/2111.02358)\]\[[code](https://github.com/microsoft/unilm/tree/master/vlmo)\]
- **BEiT: BERT Pre-Training of Image Transformers**, _Bao et al._, ICLR 2022 Oral presentation. \[[paper](https://arxiv.org/abs/2106.08254)\]\[[code](https://github.com/microsoft/unilm/tree/master/beit)\]
- BeiT-V3: **Image as a Foreign Language: BEiT Pretraining for All Vision and Vision-Language Tasks**, _Wang et al._, CVPR 2023. \[[paper](https://arxiv.org/abs/2208.10442)\]\[[code](https://github.com/microsoft/unilm/tree/master/beit3)\]

### 3. Clip

- CLIP: **Learning Transferable Visual Models From Natural Language Supervision**, _Radford et al._, ICML 2021. \[[paper](https://arxiv.org/abs/2103.00020)\]\[[code](https://github.com/OpenAI/CLIP)\]\[[clip-as-service](https://github.com/jina-ai/clip-as-service)\]\[[open_clip](https://github.com/mlfoundations/open_clip)\]
- DALL-E2: **Hierarchical Text-Conditional Image Generation with CLIP Latents**, _Ramesh et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2204.06125)\]\[[code](https://github.com/lucidrains/DALLE2-pytorch)\]
- **HiCLIP: Contrastive Language-Image Pretraining with Hierarchy-aware Attention**, _Geng et al._, ICLR 2023. \[[paper](https://arxiv.org/abs/2303.02995)\]\[[code](https://github.com/jeykigung/HiCLIP)\]
- **Chinese CLIP: Contrastive Vision-Language Pretraining in Chinese**, _Yang et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2211.01335)\]\[[code](https://github.com/OFA-Sys/Chinese-CLIP)\]
- MetaCLIP: **Demystifying CLIP Data**, _Xu et al._, ICLR 2024 Spotlight. \[[paper](https://arxiv.org/abs/2309.16671)\]\[[code](https://github.com/facebookresearch/MetaCLIP)\]
- **Alpha-CLIP: A CLIP Model Focusing on Wherever You Want**, _Sun et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.03818)\]\[[code](https://github.com/SunzeY/AlphaCLIP)\]
- MMVP: **Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs**, _Tong et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.06209)\]\[[code](https://github.com/tsb0601/MMVP)\]
- **MobileCLIP: Fast Image-Text Models through Multi-Modal Reinforced Training**, _Vasu et al._, CVPR 20224. \[[paper](https://arxiv.org/abs/2311.17049)\]\[[code](https://github.com/apple/ml-mobileclip)\]
- **Long-CLIP: Unlocking the Long-Text Capability of CLIP**, _Zhang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.15378)\]\[[code](https://github.com/beichenzbc/Long-CLIP)\]

### 4. Diffusion Model

- **Tutorial on Diffusion Models for Imaging and Vision**, _Stanley H. Chan_, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.18103)\]
- **Denoising Diffusion Probabilistic Models**，_Ho et al._, NeurIPS 2020. \[[paper](https://arxiv.org/abs/2006.11239)\]\[[code](https://github.com/hojonathanho/diffusion)\]\[[Pytorch Implementation](https://github.com/lucidrains/denoising-diffusion-pytorch)\]\[[RDDM](https://github.com/nachifur/RDDM)\]
- **Improved Denoising Diffusion Probabilistic Models**, _Nichol and Dhariwal_, ICML 2021. \[[paper](https://arxiv.org/abs/2102.09672)\]\[[code](https://github.com/openai/improved-diffusion)\]
- **Diffusion Models Beat GANs on Image Synthesis**, _Dhariwal and Nichol_, NeurIPS 2021. \[[paper](https://arxiv.org/abs/2105.05233)\]\[[code](https://github.com/openai/guided-diffusion)\]
- **Classifier-Free Diffusion Guidance**, _Ho and Salimans_, NeurIPS 2021. \[[paper](https://arxiv.org/abs/2207.12598)\]\[[code](https://github.com/lucidrains/classifier-free-guidance-pytorch)\]
- **GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models**, _Nichol et al._, arxiv 2021. \[[paper](https://arxiv.org/abs/2112.10741)\]\[[code](https://github.com/openai/glide-text2im)\]
- DALL-E2: **Hierarchical Text-Conditional Image Generation with CLIP Latents**, _Ramesh et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2204.06125)\]\[[code](https://github.com/lucidrains/DALLE2-pytorch)\]\[[dalle-mini](https://github.com/borisdayma/dalle-mini)\]
- Stable-Diffusion: **High-Resolution Image Synthesis with Latent Diffusion Models**, _Rombach et al._, CVPR 2022. \[[paper](https://arxiv.org/abs/2112.10752)\]\[[code](https://github.com/CompVis/latent-diffusion)\]\[[CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)\]\[[Stability-AI/stablediffusion](https://github.com/Stability-AI/stablediffusion)\]\[[ml-stable-diffusion](https://github.com/apple/ml-stable-diffusion)\]
- **SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis**, _Podell et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2307.01952)\]\[[code](https://github.com/Stability-AI/generative-models)\]\[[SDXL-Lightning](https://huggingface.co/ByteDance/SDXL-Lightning)\]
- **Introducing Stable Cascade**, _Stability AI_, 2024. \[[link](https://stability.ai/news/introducing-stable-cascade)\]\[[code](https://github.com/Stability-AI/StableCascade)\]\[[model](https://huggingface.co/stabilityai/stable-cascade)\]
- **SDXL-Turbo: Adversarial Diffusion Distillation**, _Sauer et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.17042)\]\[[code](https://github.com/Stability-AI/generative-models)\]
- LCM: **Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference**, _Luo et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.04378)\]\[[code](https://github.com/luosiallen/latent-consistency-model)\]\[[Hyper-SD](https://huggingface.co/ByteDance/Hyper-SD)\]
- **LCM-LoRA: A Universal Stable-Diffusion Acceleration Module**, _Luo et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.05556)\]\[[code](https://github.com/luosiallen/latent-consistency-model)\]
- Stable Diffusion 3: **Scaling Rectified Flow Transformers for High-Resolution Image Synthesis**, _Esser et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.03206)\]\[[mmdit](https://github.com/lucidrains/mmdit)\]
- SD3-Turbo: **Fast High-Resolution Image Synthesis with Latent Adversarial Diffusion Distillation**, _Sauer et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.12015)\]
- **StreamDiffusion: A Pipeline-level Solution for Real-time Interactive Generation**, _Kodaira et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.12491)\]\[[code](https://github.com/cumulo-autumn/StreamDiffusion)\]
- **DiffuseKronA: A Parameter Efficient Fine-tuning Method for Personalized Diffusion Models**, _Marjit et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.17412)\]\[[code](https://github.com/IBM/DiffuseKronA)\]
- **Video Diffusion Models**, _Ho et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2204.03458)\]\[[code](https://github.com/lucidrains/video-diffusion-pytorch)\]
- **Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets**, _Blattmann et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.15127)\]\[[code](https://github.com/Stability-AI/generative-models)\]
- **Consistency Models**, _Song et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2303.01469)\]\[[code](https://github.com/openai/consistency_models)\]\[[Consistency Decoder](https://github.com/openai/consistencydecoder)\]
- **A Survey on Video Diffusion Models**, _Xing et al._, srxiv 2023. \[[paper](https://arxiv.org/abs/2310.10647)\]\[[code](https://github.com/ChenHsing/Awesome-Video-Diffusion-Models)\]
- **Diffusion Models: A Comprehensive Survey of Methods and Applications**, _Yang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2209.00796)\]\[[code](https://github.com/YangLing0818/Diffusion-Models-Papers-Survey-Taxonomy)\]
- **Language Model Beats Diffusion -- Tokenizer is Key to Visual Generation**, _Yu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.05737)\]
- **The Chosen One: Consistent Characters in Text-to-Image Diffusion Models**, _Avrahami et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.10093)\]\[[code](https://github.com/ZichengDuan/TheChosenOne)\]
- U-ViT: **All are Worth Words: A ViT Backbone for Diffusion Models**, _Bao et al._, CVPR 2023. \[[paper](https://arxiv.org/abs/2209.12152)\]\[[code](https://github.com/baofff/U-ViT)\]
- **UniDiffuser: One Transformer Fits All Distributions in Multi-Modal Diffusion**, _Bao et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2303.06555)\]\[[code](https://github.com/thu-ml/unidiffuser)\]
- l-DAE: **Deconstructing Denoising Diffusion Models for Self-Supervised Learning**, _Chen et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.14404)\]
- DiT: **Scalable Diffusion Models with Transformers**, _Peebles et al._, ICCV 2023 Oral. \[[paper](https://arxiv.org/abs/2212.09748)\]\[[code](https://github.com/facebookresearch/DiT)\]\[[OpenDiT](https://github.com/NUS-HPC-AI-Lab/OpenDiT)\]\[[MDT](https://github.com/sail-sg/MDT)\]
- **SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers**, _Ma et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.08740)\]\[[code](https://github.com/willisma/SiT)\]
- **Hyper-SD: Trajectory Segmented Consistency Model for Efficient Image Synthesis**, _Ren et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2404.13686)\]\[[model](https://huggingface.co/ByteDance/Hyper-SD)\]

- **Github Repositories**
- \[[Awesome-Diffusion-Models](https://github.com/diff-usion/Awesome-Diffusion-Models)\]\[[Awesome-Video-Diffusion](https://github.com/showlab/Awesome-Video-Diffusion)\]
- \[[stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)\]\[[stable-diffusion-webui-colab](https://github.com/camenduru/stable-diffusion-webui-colab)\]\[[sd-webui-controlnet](https://github.com/Mikubill/sd-webui-controlnet)\]\[[stable-diffusion-webui-forge](https://github.com/lllyasviel/stable-diffusion-webui-forge)\]
- \[[Fooocus](https://github.com/lllyasviel/Fooocus)\]
- \[[ComfyUI](https://github.com/comfyanonymous/ComfyUI)\]\[[streamlit](https://github.com/streamlit/streamlit)\]\[[gradio](https://github.com/gradio-app/gradio)\]\[[ComfyUI-Workflows-ZHO](https://github.com/ZHO-ZHO-ZHO/ComfyUI-Workflows-ZHO)\]
- \[[diffusers](https://github.com/huggingface/diffusers)\]

### 5. Multimodal LLM

- LLaVA: **Visual Instruction Tuning**, _Liu et al._, NeurIPS 2023 Oral. \[[paper](https://arxiv.org/abs/2304.08485)\]\[[code](https://github.com/haotian-liu/LLaVA)\]\[[vip-llava](https://github.com/mu-cai/vip-llava)\]\[[LLaVA-pp](https://github.com/mbzuai-oryx/LLaVA-pp)\]
- LLaVA-1.5: **Improved Baselines with Visual Instruction Tuning**, _Liu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.03744)\]\[[code](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md)\]
- **LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day**, _Li et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2306.00890)\]\[[code](https://github.com/microsoft/LLaVA-Med)\]
- **Video-LLaVA: Learning United Visual Representation by Alignment Before Projection**, _Lin et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.10122)\]\[[code](https://github.com/PKU-YuanGroup/Video-LLaVA)\]
- **MoE-LLaVA: Mixture of Experts for Large Vision-Language Models**, _Lin et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.15947)\]\[[code](https://github.com/PKU-YuanGroup/MoE-LLaVA)\]
- **MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models**, _Zhu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2304.10592)\]\[[code](https://github.com/Vision-CAIR/MiniGPT-4)\]
- **MiniGPT-v2: large language model as a unified interface for vision-language multi-task learning**, _Chen et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.09478)\]\[[code](https://github.com/Vision-CAIR/MiniGPT-4)\]
- **MiniGPT4-Video: Advancing Multimodal LLMs for Video Understanding with Interleaved Visual-Textual Tokens**, _Ataallah et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2404.03413)\]\[[code](https://github.com/Vision-CAIR/MiniGPT4-video)\]
- **MiniGPT-5: Interleaved Vision-and-Language Generation via Generative Vokens**, _Zheng et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.02239)\]\[[code](https://github.com/eric-ai-lab/MiniGPT-5)\]

- **Flamingo: a Visual Language Model for Few-Shot Learning**, _Alayrac et al._, NeurIPS 2022. \[[paper](https://arxiv.org/abs/2204.14198)\]\[[open-flamingo](https://github.com/mlfoundations/open_flamingo)\]\[[flamingo-pytorch](https://github.com/lucidrains/flamingo-pytorch)\]
- **Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding**, _Zhang et al._, EMNLP 2023. \[[paper](https://arxiv.org/abs/2306.02858)\]\[[code](https://github.com/DAMO-NLP-SG/Video-LLaMA)\]
- **BuboGPT: Enabling Visual Grounding in Multi-Modal LLMs**, _Zhao et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2307.08581)\]\[[code](https://github.com/magic-research/bubogpt)\]\[[AnyGPT](https://github.com/OpenMOSS/AnyGPT)\]
- **CogVLM: Visual Expert for Pretrained Language Models**, _Wang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.03079)\]\[[code](https://github.com/THUDM/CogVLM)\]\[[VisualGLM-6B](https://github.com/THUDM/VisualGLM-6B)\]\[[OmniLMM](https://github.com/OpenBMB/OmniLMM)\]
- **DreamLLM: Synergistic Multimodal Comprehension and Creation**, _Dong et al._, ICLR 2024 Spotlight. \[[paper](https://arxiv.org/abs/2309.11499)\]\[[code](https://github.com/RunpeiDong/DreamLLM)\]
- **NExT-GPT: Any-to-Any Multimodal LLM**, _Wu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.05519)\]\[[code](https://github.com/NExT-GPT/NExT-GPT)\]
- SoM: **Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V**, _Yang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.11441)\]\[[code](https://github.com/microsoft/SoM)\]
- **Ferret: Refer and Ground Anything Anywhere at Any Granularity**, _You et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.07704)\]\[[code](https://github.com/apple/ml-ferret)\]\[[Ferret-UI](https://arxiv.org/abs/2404.05719)\]
- **Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond**, _Bai et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2308.12966)\]\[[code](https://github.com/QwenLM/Qwen-VL)\]
- **InternLM-XComposer: A Vision-Language Large Model for Advanced Text-image Comprehension and Composition**, _Zhang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.15112)\]\[[code](https://github.com/InternLM/InternLM-XComposer)\]
- **InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks**, _Chen et al._, CVPR 2024. \[[paper](https://arxiv.org/abs/2312.14238)\]\[[code](https://github.com/OpenGVLab/InternVL)\]\[[InternVideo](https://github.com/OpenGVLab/InternVideo)\]
- **DeepSeek-VL: Towards Real-World Vision-Language Understanding**, _Lu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.05525)\]\[[code](https://github.com/deepseek-ai/DeepSeek-VL)\]
- **ShareGPT4V: Improving Large Multi-Modal Models with Better Captions**, _Chen et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.12793)\]\[[code](https://github.com/InternLM/InternLM-XComposer/tree/main/projects/ShareGPT4V)\]
- **TinyGPT-V: Efficient Multimodal Large Language Model via Small Backbones**, _Yuan et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.16862)\]\[[code](https://github.com/DLYuanGod/TinyGPT-V)\]
- **Monkey: Image Resolution and Text Label Are Important Things for Large Multi-modal Models**, _Li et al._, CVPR 2024. \[[paper](https://arxiv.org/abs/2311.06607)\]\[[code](https://github.com/Yuliang-Liu/Monkey)\]
- **Vary: Scaling up the Vision Vocabulary for Large Vision-Language Models**, _Wei et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.06109)\]\[[code](https://github.com/Ucas-HaoranWei/Vary)\]
- Vary-toy: **Small Language Model Meets with Reinforced Vision Vocabulary**, _Wei et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.12503)\]\[[code](https://github.com/Ucas-HaoranWei/Vary-toy)\]
- LWM: **World Model on Million-Length Video And Language With RingAttention**, _Liu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.08268)\]\[[code](https://github.com/LargeWorldModel/LWM)\]

- \[[MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V)\]\[[moondream](https://github.com/vikhyat/moondream)\]\[[MobileVLM](https://github.com/Meituan-AutoML/MobileVLM)\]\[[OmniFusion](https://github.com/AIRI-Institute/OmniFusion)\]\[[Bunny](https://github.com/BAAI-DCAI/Bunny)\]

### 6. Text2Image

- DALL-E: **Zero-Shot Text-to-Image Generation**, _Ramesh et al._, arxiv 2021. \[[paper](https://arxiv.org/abs/2102.12092)\]\[[code](https://github.com/openai/DALL-E)\]
- DALL-E3: **Improving Image Generation with Better Captions**, _Betker et al._, OpenAI 2023. \[[paper](https://cdn.openai.com/papers/dall-e-3.pdf)\]\[[code](https://github.com/openai/consistencydecoder)\]\[[blog](https://openai.com/dall-e-3)\]
- ControlNet: **Adding Conditional Control to Text-to-Image Diffusion Models**, _Zhang et al._, ICCV 2023 Marr Prize. \[[paper](https://arxiv.org/abs/2302.05543)\]\[[code](https://github.com/lllyasviel/ControlNet)\]
- **T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models**, _Mou et al._, AAAI 2024. \[[paper](https://arxiv.org/abs/2302.08453)\]\[[code](https://github.com/TencentARC/T2I-Adapter)\]
- **AnyText: Multilingual Visual Text Generation And Editing**, _Tuo et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.03054)\]\[[code](https://github.com/tyxsspa/AnyText)\]
- RPG: **Mastering Text-to-Image Diffusion: Recaptioning, Planning, and Generating with Multimodal LLMs**, _Yang et al._, ICML 2024. \[[paper](https://arxiv.org/abs/2401.11708)\]\[[code](https://github.com/YangLing0818/RPG-DiffusionMaster)\]

- **LAION-5B: An open large-scale dataset for training next generation image-text models**, _Schuhmann et al._, NeurIPS 2022. \[[paper](https://arxiv.org/abs/2210.08402)\]\[[code](https://github.com/LAION-AI/laion-datasets)\]\[[blog](https://laion.ai/blog/laion-5b/)\]
- DeepFloyd IF: **Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding**, _Saharia et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2205.11487)\]\[[code](https://github.com/deep-floyd/IF)\]
- Imagen: **Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding**, _Saharia et al._, NeurIPS 2022. \[[paper](https://arxiv.org/abs/2205.11487)\]\[[unofficial code](https://github.com/lucidrains/imagen-pytorch)\]
- **Instruct-Imagen: Image Generation with Multi-modal Instruction**, _Hu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.01952)\]
- **TextDiffuser: Diffusion Models as Text Painters**, _Chen et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.10855)\]\[[code](https://github.com/microsoft/unilm/tree/master/textdiffuser)\]
- **TextDiffuser-2: Unleashing the Power of Language Models for Text Rendering**, _Chen et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.16465)\]\[[code](https://github.com/microsoft/unilm/tree/master/textdiffuser-2)\]
- **PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis**, _Chen et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.00426)\]\[[code](https://github.com/PixArt-alpha/PixArt-alpha)\]
- **PIXART-δ: Fast and Controllable Image Generation with Latent Consistency Models**, _Chen et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.05252)\]\[[code](https://github.com/PixArt-alpha/PixArt-alpha)\]
- **PixArt-Σ: Weak-to-Strong Training of Diffusion Transformer for 4K Text-to-Image Generation**, _Chen et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.04692)\]\[[code](https://github.com/PixArt-alpha/PixArt-sigma)\]
- **IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models**, _Ye et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2308.06721)\]\[[code](https://github.com/tencent-ailab/IP-Adapter)\]
- **Controllable Generation with Text-to-Image Diffusion Models: A Survey**, _Cao et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.04279)\]\[[code](https://github.com/PRIV-Creation/Awesome-Controllable-T2I-Diffusion-Models)\]
- **StoryDiffusion: Consistent Self-Attention for Long-Range Image and Video Generation**, _Zhou et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2405.01434)\]\[[code](https://github.com/HVision-NKU/StoryDiffusion)\]

### 7. Text2Video

- **Animate Anyone: Consistent and Controllable Image-to-Video Synthesis for Character Animation**, _Hu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.17117)\]\[[code](https://github.com/HumanAIGC/AnimateAnyone)\]\[[Open-AnimateAnyone](https://github.com/guoqincode/Open-AnimateAnyone)\]\[[Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone)\]
- **EMO: Emote Portrait Alive - Generating Expressive Portrait Videos with Audio2Video Diffusion Model under Weak Conditions**, _Tian et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.17485)\]\[[code](https://github.com/HumanAIGC/EMO)\]
- **AniPortrait: Audio-Driven Synthesis of Photorealistic Portrait Animation**, _Wei wt al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.17694)\]\[[code](https://github.com/Zejun-Yang/AniPortrait)\]
- **DreaMoving: A Human Video Generation Framework based on Diffusion Models**, _Feng et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.05107)\]\[[code](https://github.com/dreamoving/dreamoving-project)\]
- **MagicAnimate:Temporally Consistent Human Image Animation using Diffusion Model**, _Xu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.16498)\]\[[code](https://github.com/magic-research/magic-animate)\]\[[champ](https://github.com/fudan-generative-vision/champ)\]
- **FlowVid: Taming Imperfect Optical Flows for Consistent Video-to-Video Synthesis**, _Liang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.17681)\]\[[code](https://github.com/Jeff-LiangF/FlowVid)\]

- \[[Awesome-Video-Diffusion](https://github.com/showlab/Awesome-Video-Diffusion)\]
- **Video Diffusion Models**, _Ho et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2204.03458)\]\[[video-diffusion-pytorch](https://github.com/lucidrains/video-diffusion-pytorch)\]
- **Make-A-Video: Text-to-Video Generation without Text-Video Data**, _Singer et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2209.14792)\]\[[make-a-video-pytorch](https://github.com/lucidrains/make-a-video-pytorch)\]
- **Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation**, _Wu et al._, ICCV 2023. \[[paper](https://arxiv.org/abs/2212.11565)\]\[[code](https://github.com/showlab/Tune-A-Video)\]
- **Text2Video-Zero: Text-to-Image Diffusion Models are Zero-Shot Video Generators**, _Khachatryan et al._, ICCV 2023 Oral. \[[paper](https://arxiv.org/abs/2303.13439)\]\[[code](https://github.com/Picsart-AI-Research/Text2Video-Zero)\]\[[StreamingT2V](https://github.com/Picsart-AI-Research/StreamingT2V)\]
- **CogVideo: Large-scale Pretraining for Text-to-Video Generation via Transformers**, _Hong et al._, ICLR 2023. \[[paper](https://arxiv.org/abs/2205.15868)\]\[[code](https://github.com/THUDM/CogVideo)\]

- **Follow Your Pose: Pose-Guided Text-to-Video Generation using Pose-Free Videos**, _Ma et al._, AAAI 2024. \[[paper](https://arxiv.org/abs/2304.01186)\]\[[code](https://github.com/mayuelala/FollowYourPose)\]
- **Follow-Your-Click: Open-domain Regional Image Animation via Short Prompts**, _Ma et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.08268)\]\[[code](https://github.com/mayuelala/FollowYourClick)\]
- **AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning**, _Guo et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2307.04725)\]\[[code](https://github.com/guoyww/AnimateDiff)\]\[[AnimateDiff-Lightning](https://huggingface.co/ByteDance/AnimateDiff-Lightning)\]
- **StableVideo: Text-driven Consistency-aware Diffusion Video Editing**, _Chai et al._, ICCV 2023. \[[paper](https://arxiv.org/abs/2308.09592)\]\[[code](https://github.com/rese1f/StableVideo)\]
- **I2VGen-XL: High-Quality Image-to-Video Synthesis via Cascaded Diffusion Models**, _Zhang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.04145)\]\[[code](https://github.com/ali-vilab/VGen)\]
- TF-T2V: **A Recipe for Scaling up Text-to-Video Generation with Text-free Videos**, _Wang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.15770)\]\[[code](https://github.com/ali-vilab/VGen)\]
- **Lumiere: A Space-Time Diffusion Model for Video Generation**, _Bar-Tal et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.12945)\]\[[lumiere-pytorch](https://github.com/lucidrains/lumiere-pytorch)\]
- **Sora: Creating video from text**, _OpenAI_, 2024. \[[blog](https://openai.com/sora)\]\[[Open-Sora](https://github.com/hpcaitech/Open-Sora)\]\[[Open-Sora-Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan)\]\[[minisora](https://github.com/mini-sora/minisora)\]\[[SoraWebui](https://github.com/SoraWebui/SoraWebui)\]\[[MuseV](https://github.com/TMElyralab/MuseV)\]\[[PhysDreamer](https://github.com/a1600012888/PhysDreamer)\]
- **Sora: A Review on Background, Technology, Limitations, and Opportunities of Large Vision Models**, _Liu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.17177)\]\[[code](https://github.com/lichao-sun/SoraReview)\]
- **Mora: Enabling Generalist Video Generation via A Multi-Agent Framework**, _Yuan et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.13248)\]\[[code](https://github.com/lichao-sun/Mora)\]
- **Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution**, _Dehghani et al._, NeurIPS 2024. \[[paper](https://arxiv.org/abs/2307.06304)\]\[[unofficial code](https://github.com/kyegomez/NaViT)\]
- **VideoPoet: A Large Language Model for Zero-Shot Video Generation**, _Kondratyuk et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.14125)\]
- **Latte: Latent Diffusion Transformer for Video Generation**, _Ma et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.03048)\]\[[code](https://github.com/Vchitect/Latte)\]\[[LaVIT](https://github.com/jy0205/LaVIT)\]
- **Snap Video: Scaled Spatiotemporal Transformers for Text-to-Video Synthesis**, _Menapace et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.14797)\]
- \[[MoneyPrinterTurbo](https://github.com/harry0703/MoneyPrinterTurbo)\]\[[videos](https://github.com/3b1b/videos)\]

### 8. Survey for Multimodal

- **A Survey on Multimodal Large Language Models**, _Yin et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2306.13549)\]\[[code](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models)\]
- **Multimodal Foundation Models: From Specialists to General-Purpose Assistants**, _Li et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.10020)\]\[[cvinw_readings](https://github.com/computer-vision-in-the-wild/cvinw_readings)\]
- **From GPT-4 to Gemini and Beyond: Assessing the Landscape of MLLMs on Generalizability, Trustworthiness and Causality through Four Modalities**, _Lu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.15071)\]\[[Leaderboards](https://openlamm.github.io/Leaderboards)\]

### 9. Other

- **Fuyu-8B: A Multimodal Architecture for AI Agents** _Bavishi et al._, Adept blog 2023. \[[blog](https://www.adept.ai/blog/fuyu-8b)\]\[[model](https://huggingface.co/adept/fuyu-8b)\]
- **Otter: A Multi-Modal Model with In-Context Instruction Tuning**, _Li et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.03726)\]\[[code](https://github.com/Luodian/Otter)\]
- **OtterHD: A High-Resolution Multi-modality Model**, _Li et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.04219)\]\[[code](https://github.com/Luodian/Otter)\]\[[model](https://huggingface.co/Otter-AI/OtterHD-8B)\]
- CM3leon: **Scaling Autoregressive Multi-Modal Models_Pretraining and Instruction Tuning**, _Yu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.02591)\]\[[Unofficial Implementation](https://github.com/kyegomez/CM3Leon)\]
- **MM-Interleaved: Interleaved Image-Text Generative Modeling via Multi-modal Feature Synchronizer**, _Tian et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.10208)\]\[[code](https://github.com/OpenGVLab/MM-Interleaved)\]
- **CogCoM: Train Large Vision-Language Models Diving into Details through Chain of Manipulations**, _Qi et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.04236)\]\[[code](https://github.com/THUDM/CogCoM)\]
- **SPHINX-X: Scaling Data and Parameters for a Family of Multi-modal Large Language Models**, _Gao et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.05935)\]\[[code](https://github.com/Alpha-VLLM/LLaMA2-Accessory)\]
- LWM: **World Model on Million-Length Video And Language With RingAttention**, _Liu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.08268)\]\[[code](https://github.com/LargeWorldModel/LWM)\]

<details>
---
<details>
<summary>Reinforcement Learning</summary>

## Reinforcement Learning

### 1.Basic for RL

- **Deep Reinforcement Learning: Pong from Pixels**, _Andrej Karpathy_, 2016. \[[blog](https://karpathy.github.io/2016/05/31/rl/)\]\[[reinforcement-learning-an-introduction](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)\]\[[easy-rl](https://github.com/datawhalechina/easy-rl)\]\[[deep-rl-course](https://huggingface.co/learn/deep-rl-course/)\]
- DQN: **Playing Atari with Deep Reinforcement Learning**, _Mnih et al._, arxiv 2013. \[[paper](https://arxiv.org/abs/1312.5602)\]\[[code](https://github.com/higgsfield/RL-Adventure/blob/master/1.dqn.ipynb)\]
- DQNNaturePaper: **Human-level control through deep reinforcement learning**, _Mnih et al._, Nature 2015. \[[paper](https://www.nature.com/articles/nature14236)\]\[[DQN-tensorflow](https://github.com/devsisters/DQN-tensorflow)\]\[[DQN_pytorch](https://github.com/dxyang/DQN_pytorch)\]
- DDQN: **Deep Reinforcement Learning with Double Q-learning**, _Hasselt et al._, AAAI 2016. \[[paper](https://arxiv.org/abs/1509.06461)\]\[[RL-Adventure](https://github.com/higgsfield/RL-Adventure)\]\[[deep-q-learning](https://github.com/keon/deep-q-learning)\]\[[Deep-RL-Keras](https://github.com/germain-hug/Deep-RL-Keras)\]
- **Rainbow: Combining Improvements in Deep Reinforcement Learning**, _Hesssel et al._, AAAI 2018. \[[paper](https://arxiv.org/abs/1710.02298)\]\[[Rainbow](https://github.com/Kaixhin/Rainbow)\]
- DDPG: **Continuous control with deep reinforcement learning**, _Lillicrap et al._, ICLR 2016. \[[paper](https://arxiv.org/abs/1509.02971)\]\[[pytorch-ddpg](https://github.com/ghliu/pytorch-ddpg)\]

- PPO: **Proximal Policy Optimization Algorithms**, _Schulman et al._, arxiv 2017. \[[paper](https://arxiv.org/abs/1707.06347)\]\[[code](https://github.com/openai/baselines)\]\[[PPO-PyTorch](https://github.com/nikhilbarhate99/PPO-PyTorch)\]\[[implementation-matters](https://github.com/MadryLab/implementation-matters)\]\[[PPOxFamily](https://github.com/opendilab/PPOxFamily)\]

- **Diffusion Models for Reinforcement Learning: A Survey**, _Zhu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.01223)\]\[[code](https://github.com/apexrl/Diff4RLSurvey)\]\[[diffusion_policy](https://github.com/real-stanford/diffusion_policy)\]
- **The Definitive Guide to Policy Gradients in Deep Reinforcement Learning: Theory, Algorithms and Implementations**, _Matthias Lehmann_, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.13662)\]\[[code](https://github.com/Matt00n/PolicyGradientsJax)\]

- \[[tianshou](https://github.com/thu-ml/tianshou)\]\[[rlkit](https://github.com/rail-berkeley/rlkit)\]

### 2. LLM for decision making 

- **Decision Transformer_Reinforcement Learning via Sequence Modeling**, _Chen et al._, NeurIPS 2021. \[[paper](https://arxiv.org/abs/2106.01345)\]\[[code](https://github.com/kzl/decision-transformer)\]
- Trajectory Transformer: **Offline Reinforcement Learning as One Big Sequence Modeling Problem**, _Janner et al._, NeurIPS 2021. \[[paper](https://arxiv.org/abs/2106.02039)\]\[[code](https://github.com/JannerM/trajectory-transformer)\]
- **Guiding Pretraining in Reinforcement Learning with Large Language Models**, _Du et al._, ICML 2023. \[[paper](https://arxiv.org/abs/2302.06692)\]\[[code](https://github.com/yuqingd/ellm)\]
- **Introspective Tips: Large Language Model for In-Context Decision Making**, _Chen et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.11598)\]
- **Q-Transformer: Scalable Offline Reinforcement Learning via Autoregressive Q-Functions**, _Chebotar et al._, CoRL 2023. \[[paper](https://arxiv.org/abs/2309.10150)\]\[[Unofficial Implementation](https://github.com/lucidrains/q-transformer)\]
- **Survey on Large Language Model-Enhanced Reinforcement Learning: Concept, Taxonomy, and Methods**, _Cao et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2404.00282)\]

<details>
---
<details>
<summary>GNN</summary>

## GNN
 
- \[[GNNPapers](https://github.com/thunlp/GNNPapers)\]\[[dgl](https://github.com/dmlc/dgl)\]
- **A Gentle Introduction to Graph Neural Networks**, _Sanchez-Lengeling et al._, Distill 2021. \[[paper](https://distill.pub/2021/gnn-intro/)\]
- **CS224W: Machine Learning with Graphs**, Stanford. \[[link](http://web.stanford.edu/class/cs224w)\]

- GCN: **Semi-Supervised Classification with Graph Convolutional Networks**, _Kipf and Welling_, ICLR 2017. \[[paper](https://arxiv.org/abs/1609.02907)\]\[[code](https://github.com/tkipf/gcn)\]\[[pygcn](https://github.com/tkipf/pygcn)\]
- GAE: **Variational Graph Auto-Encoders**, _Kipf and Welling_, arxiv 2016. \[[paper](https://arxiv.org/abs/1611.07308)\]\[[code](https://github.com/tkipf/gae)\]\[[gae-pytorch](https://github.com/zfjsail/gae-pytorch)\]
- GAT: **Graph Attention Networks**, _Veličković et al._, ICLR 2018. \[[paper](https://arxiv.org/abs/1710.10903)\]\[[code](https://github.com/PetarV-/GAT)\]\[[pyGAT](https://github.com/Diego999/pyGAT)\]\[[pytorch-GAT](https://github.com/gordicaleksa/pytorch-GAT)\]
- GIN: **How Powerful are Graph Neural Networks?**, _Xu et al._, ICLR 2019. \[[paper](https://arxiv.org/abs/1810.00826)\]\[[code](https://github.com/weihua916/powerful-gnns)\]

- **Do Transformers Really Perform Bad for Graph Representation**, _Ying et al._, NeurIPS 2021. \[[paper](https://arxiv.org/abs/2106.05234)\]\[[code](https://github.com/Microsoft/Graphormer)\]
- **GraphGPT: Graph Instruction Tuning for Large Language Models**, _Tang et al._, SIGIR 2024. \[[paper](https://arxiv.org/abs/2310.13023)\]\[[code](https://github.com/HKUDS/GraphGPT)\]
- **OpenGraph: Towards Open Graph Foundation Models**, _Xia et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.01121)\]\[[code](https://github.com/HKUDS/OpenGraph)\]

- \[[pytorch_geometric](https://github.com/pyg-team/pytorch_geometric)\]

### Survey for GNN

<details>
---
<details>
<summary>Transformer Architecture</summary>

## Transformer Architecture

- **Attention is All you Need**, _Vaswani et al._, NIPS 2017. \[[paper](https://arxiv.org/abs/1706.03762)\]\[[code](https://github.com/jadore801120/attention-is-all-you-need-pytorch)\]\[[transformer-debugger](https://github.com/openai/transformer-debugger)\]\[[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)\]\[[The Random Transformer](https://osanseviero.github.io/hackerllama/blog/posts/random_transformer/)\]\[[The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)\]\[[Transformers-Tutorials](https://github.com/NielsRogge/Transformers-Tutorials)\]\[[x-transformers](https://github.com/lucidrains/x-transformers)\]
- RoPE: **RoFormer: Enhanced Transformer with Rotary Position Embedding**, _Su et al._, arxiv 2021. \[[paper](https://arxiv.org/abs/2104.09864)\]\[[code](https://github.com/ZhuiyiTechnology/roformer)\]\[[rotary-embedding-torch](https://github.com/lucidrains/rotary-embedding-torch)\]\[[blog](https://kexue.fm/archives/9675)\]
- **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints**, _Ainslie et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.13245)\]\[[unofficial code](https://github.com/fkodom/grouped-query-attention-pytorch)\]
- **RWKV: Reinventing RNNs for the Transformer Era**, _Peng et al._, EMNLP 2023. \[[paper](https://arxiv.org/abs/2305.13048)\]\[[code](https://github.com/BlinkDL/RWKV-LM)\]\[[ChatRWKV](https://github.com/BlinkDL/ChatRWKV)\]\[[rwkv.cpp](https://github.com/RWKV/rwkv.cpp)\]
- **Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence**, _Peng et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2404.05892)\]\[[code](https://github.com/RWKV/RWKV-LM)\]
- **Mamba: Linear-Time Sequence Modeling with Selective State Spaces**, _Gu and Dao_, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.00752)\]\[[code](https://github.com/state-spaces/mamba)\]\[[mamba-minimal](https://github.com/johnma2006/mamba-minimal)\]\[[Awesome-Mamba-Papers](https://github.com/yyyujintang/Awesome-Mamba-Papers)\]
- **Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models**, _De et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.19427)\]\[[recurrentgemma](https://github.com/google-deepmind/recurrentgemma)\]
- **Jamba: A Hybrid Transformer-Mamba Language Model**, _Lieber et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.19887)\]\[[model](https://huggingface.co/ai21labs/Jamba-v0.1)\]
- **Neural Network Diffusion**, _Wang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.13144)\]\[[code](https://github.com/NUS-HPC-AI-Lab/Neural-Network-Diffusion)\]\[[GPD](https://github.com/tsinghua-fib-lab/GPD)\]
- **KAN: Kolmogorov-Arnold Networks**, _Liu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2404.19756)\]\[[code](https://github.com/KindXiaoming/pykan)\]\[[efficient-kan](https://github.com/Blealtan/efficient-kan)\]\[[kan-gpt](https://github.com/AdityaNG/kan-gpt)\]

<details>
