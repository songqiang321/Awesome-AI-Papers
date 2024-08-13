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

## NLP

### 1. Word2Vec

- **Efficient Estimation of Word Representations in Vector Space**, _Mikolov et al._, arxiv 2013. \[[paper](https://arxiv.org/abs/1301.3781)\]
- **Distributed Representations of Words and Phrases and their Compositionality**, _Mikolov et al._, NIPS 2013. \[[paper](https://arxiv.org/abs/1310.4546)\]
- **Distributed representations of sentences and documents**, _Le and Mikolov_, ICML 2014. \[[paper](https://arxiv.org/abs/1405.4053)\]
- **Word2vec Explained: deriving Mikolov et al.'s negative-sampling word-embedding method**, _Goldberg and Levy_, arxiv 2014. \[[paper](https://arxiv.org/abs/1402.3722)\]
- **word2vec Parameter Learning Explained**, _Rong_, arxiv 2014. \[[paper](https://arxiv.org/abs/1411.2738)\]
- **Glove: Global vectors for word representation.**，_Pennington et al._, EMNLP 2014. \[[paper](https://aclanthology.org/D14-1162/)\]\[[code](https://github.com/stanfordnlp/GloVe)\]
- fastText: **Bag of Tricks for Efficient Text Classification**, _Joulin et al._, arxiv 2016. \[[paper](https://arxiv.org/abs/1607.01759)\]\[[code](https://github.com/facebookresearch/fastText)\]
- ELMo: **Deep Contextualized Word Representations**, _Peters et al._, NAACL 2018. \[[paper](https://arxiv.org/abs/1802.05365)\]
- BPE: **Neural Machine Translation of Rare Words with Subword Units**, _Sennrich et al._, ACL 2016. \[[paper](https://arxiv.org/abs/1508.07909)\]\[[code](https://github.com/rsennrich/subword-nmt)\]
- Byte-Level BPE: **Neural Machine Translation with Byte-Level Subwords**, _Wang et al._, arxiv 2019. \[[paper](https://arxiv.org/abs/1909.03341)\]\[[code](https://github.com/facebookresearch/fairseq/tree/main/examples/byte_level_bpe)\]

### 2. Seq2Seq

- **Generating Sequences With Recurrent Neural Networks**, _Graves_, arxiv 2013. \[[paper](https://arxiv.org/abs/1308.0850)\]
- **Sequence to Sequence Learning with Neural Networks**, _Sutskever et al._, NeruIPS 2014. \[[paper](https://arxiv.org/abs/1409.3215)\]
- **Neural Machine Translation by Jointly Learning to Align and Translate**, _Bahdanau et al._, ICLR 2015. \[[paper](https://arxiv.org/abs/1409.0473)\]\[[code](https://github.com/lisa-groundhog/GroundHog)\]
- **On the Properties of Neural Machine Translation: Encoder-Decoder Approaches**, _Cho et al._, arxiv 2014. \[[paper](https://arxiv.org/abs/1409.1259)\]
- **Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation**, _Cho et al._, arxiv 2014. \[[paper](https://arxiv.org/abs/1406.1078)\]
- \[[fairseq](https://github.com/facebookresearch/fairseq)\]\[[fairseq2](https://github.com/facebookresearch/fairseq2)\]\[[pytorch-seq2seq](https://github.com/IBM/pytorch-seq2seq)\]

### 3. Pretraining

- **Attention Is All You Need**, _Vaswani et al._, NIPS 2017. \[[paper](https://arxiv.org/abs/1706.03762)\]\[[code](https://github.com/tensorflow/tensor2tensor)\]
- GPT: **Improving language understanding by generative pre-training**, _Radford et al._, preprint 2018.  \[[paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)\]\[[code](https://github.com/openai/finetune-transformer-lm)\]
- GPT-2: **Language Models are Unsupervised Multitask Learners**, _Radford et al._, OpenAI blog 2019. \[[paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)\]\[[code](https://github.com/openai/gpt-2)\]\[[llm.c](https://github.com/karpathy/llm.c)\]
- GPT-3: **Language Models are Few-Shot Learners**, _Brown et al._, NeurIPS 2020. \[[paper](https://arxiv.org/abs/2005.14165)\]\[[code](https://github.com/openai/gpt-3)\]\[[nanoGPT](https://github.com/karpathy/nanoGPT)\]\[[build-nanogpt](https://github.com/karpathy/build-nanogpt)\]\[[gpt-fast](https://github.com/pytorch-labs/gpt-fast)\]\[[modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)\]
- InstructGPT: **Training language models to follow instructions with human feedback**, _Ouyang et al._, NeurIPS 2022. \[[paper](https://arxiv.org/abs/2203.02155)\]\[[MOSS-RLHF](https://github.com/OpenLMLab/MOSS-RLHF)\]
- **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**, _Devlin et al._, NAACL 2019 Best Paper. \[[paper](https://arxiv.org/abs/1810.04805)\]\[[code](https://github.com/google-research/bert)\]\[[BERT-pytorch](https://github.com/codertimo/BERT-pytorch)\]\[[bert4torch](https://github.com/Tongjilibo/bert4torch)\]\[[bert4keras](https://github.com/bojone/bert4keras)\]
- **RoBERTa: A Robustly Optimized BERT Pretraining Approach**, _Liu et al._, arxiv 2019. \[[paper](https://arxiv.org/abs/1907.11692)\]\[[code](https://github.com/facebookresearch/fairseq/tree/main/examples/roberta)\]\[[Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)\]
- **What Does BERT Look At: An Analysis of BERT's Attention**, _Clark et al._, arxiv 2019. \[[paper](https://arxiv.org/abs/1906.04341)\]\[[code](https://github.com/clarkkev/attention-analysis)\]
- **DeBERTa: Decoding-enhanced BERT with Disentangled Attention**, _He et al._, ICLR 2021. \[[paper](https://arxiv.org/abs/2006.03654)\]\[[code](https://github.com/microsoft/DeBERTa)\]
- **DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter** _Sanh et al._, arxiv 2019. \[[paper](https://arxiv.org/abs/1910.01108)\]\[[code](https://github.com/huggingface/transformers)\]\[[albert_pytorch](https://github.com/lonePatient/albert_pytorch)\]
- **BERT Rediscovers the Classical NLP Pipeline**, _Tenney et al._, arxiv 2019. \[[paper](https://arxiv.org/abs/1905.05950)\]\[[code](https://github.com/nyu-mll/jiant)\]
- **How to Fine-Tune BERT for Text Classification?**, _Sun et al._, arxiv 2019. \[[paper](https://arxiv.org/abs/1905.05583)\]\[[code](https://github.com/xuyige/BERT4doc-Classification)\]
- **TinyStories: How Small Can Language Models Be and Still Speak Coherent English**, _Eldan and Li_, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.07759)\]\[[dataset](https://huggingface.co/datasets/roneneldan/TinyStories)\]\[[phi-3](https://huggingface.co/collections/microsoft/phi-3-6626e15e9585a200d2d761e3)\]

- \[[LLM101n](https://github.com/karpathy/LLM101n)\]\[[EurekaLabsAI](https://github.com/EurekaLabsAI)\]\[[llm-course](https://github.com/mlabonne/llm-course)\]\[[intro-llm](https://intro-llm.github.io/)\]\[[llm-cookbook](https://github.com/datawhalechina/llm-cookbook)\]\[[hugging-llm](https://github.com/datawhalechina/hugging-llm)\]\[[generative-ai-for-beginners](https://github.com/microsoft/generative-ai-for-beginners)\]\[[awesome-generative-ai-guide](https://github.com/aishwaryanr/awesome-generative-ai-guide)\]\[[LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)\]\[[llm-action](https://github.com/liguodongiot/llm-action)\]
- \[[cs230-code-examples](https://github.com/cs230-stanford/cs230-code-examples)\]\[[victoresque/pytorch-template](https://github.com/victoresque/pytorch-template)\]\[[songquanpeng/pytorch-template](https://github.com/songquanpeng/pytorch-template)\]
- \[[tokenizer_summary](https://huggingface.co/docs/transformers/tokenizer_summary)\]\[[minbpe](https://github.com/karpathy/minbpe)\]\[[tokenizers](https://github.com/huggingface/tokenizers)\]\[[tiktoken](https://github.com/openai/tiktoken)\]\[[SentencePiece](https://github.com/google/sentencepiece)\]

#### 3.1 Large Language Model

- **A Survey of Large Language Models**, _Zhao etal._, arxiv 2023. \[[paper](https://arxiv.org/abs/2303.18223)\]\[[code](https://github.com/RUCAIBox/LLMSurvey)\]\[[LLMBox](https://github.com/RUCAIBox/LLMBox)\]\[[LLMBook-zh](https://llmbook-zh.github.io/)\]\[[LLMsPracticalGuide](https://github.com/Mooler0410/LLMsPracticalGuide)\]
- **Efficient Large Language Models: A Survey**, _Wan et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.03863)\]\[[code](https://github.com/AIoT-MLSys-Lab/Efficient-LLMs-Survey)\]
- **Challenges and Applications of Large Language Models**, _Kaddour et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2307.10169)\]
- **A Comprehensive Survey on Pretrained Foundation Models: A History from BERT to ChatGPT**, _Zhou et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2302.09419)\]
- **From Google Gemini to OpenAI Q* (Q-Star): A Survey of Reshaping the Generative Artificial Intelligence (AI) Research Landscape**, _Mclntosh et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.10868)\]\[[AGI-survey](https://github.com/ulab-uiuc/AGI-survey)\]
- **A Survey of Resource-efficient LLM and Multimodal Foundation Models**, _Xu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.08092)\]\[[code](https://github.com/UbiquitousLearning/Efficient_Foundation_Model_Survey)\]
- **Large Language Models: A Survey**, _Minaee et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.06196)\]
- Anthropic: **Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback**, _Bai et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2204.05862)\]\[[code](https://github.com/anthropics/hh-rlhf)\]
- Anthropic: **Constitutional AI: Harmlessness from AI Feedback**, _Bai et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2212.08073)\]\[[code](https://github.com/anthropics/ConstitutionalHarmlessnessPaper)\]
- Anthropic: **Model Card and Evaluations for Claude Models**, Anthropic, 2023. \[[paper](https://www-files.anthropic.com/production/images/Model-Card-Claude-2.pdf)\]
- Anthropic: **The Claude 3 Model Family: Opus, Sonnet, Haiku**, Anthropic, 2024. \[[paper](https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf)\]\[[Claude 3.5](https://www-cdn.anthropic.com/fed9cc193a14b84131812372d8d5857f8f304c52/Model_Card_Claude_3_Addendum.pdf)\]
- **BLOOM: A 176B-Parameter Open-Access Multilingual Language Model**, _BigScience Workshop_, arxiv 2022. \[[paper](https://arxiv.org/abs/2211.05100)\]\[[code](https://github.com/bigscience-workshop)\]\[[model](https://huggingface.co/bigscience)\]
- **OPT: Open Pre-trained Transformer Language Models**, _Zhang et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2205.01068)\]\[[code](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT)\]
- Chinchilla: **Training Compute-Optimal Large Language Models**, _Hoffmann et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2203.15556)\]
- Gopher: **Scaling Language Models: Methods, Analysis & Insights from Training Gopher**, _Rae et al._, arxiv 2021. \[[paper](https://arxiv.org/abs/2112.11446)\]
- **GPT-NeoX-20B: An Open-Source Autoregressive Language Model**, _Black et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2204.06745)\]\[[code](https://github.com/EleutherAI/gpt-neox)\]
- **Gemini: A Family of Highly Capable Multimodal Models**, _Gemini Team, Google_, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.11805)\]\[[Gemini 1.0](https://storage.googleapis.com/deepmind-media/gemini/gemini_1_report.pdf)\]\[[Gemini 1.5](https://arxiv.org/abs/2403.05530)\]\[[Unofficial Implementation](https://github.com/kyegomez/Gemini)\]\[[MiniGemini](https://github.com/dvlab-research/MGM)\]
- **Gemma: Open Models Based on Gemini Research and Technology**, _Google DeepMind_, 2024. \[[paper](https://arxiv.org/abs/2403.08295)\]\[[code](https://github.com/google/gemma_pytorch)\]\[[google-deepmind/gemma](https://github.com/google-deepmind/gemma)\]\[[gemma.cpp](https://github.com/google/gemma.cpp)\]\[[model](https://ai.google.dev/gemma)\]\[[paligemma](https://github.com/google-research/big_vision/tree/main/big_vision/configs/proj/paligemma)\]\[[gemma-cookbook](https://github.com/google-gemini/gemma-cookbook)\]
- **Gemma 2: Improving Open Language Models at a Practical Size**, _Google Team_, 2024. \[[paper](https://arxiv.org/abs/2408.00118)\]\[[blog](https://blog.google/technology/developers/google-gemma-2/)\]\[[Advancing Responsible AI with Gemma](https://developers.googleblog.com/en/smaller-safer-more-transparent-advancing-responsible-ai-with-gemma/)\]\[[Gemma Scope](https://arxiv.org/abs/2408.05147)\]\[[ShieldGemma](https://arxiv.org/abs/2407.21772)\]\[[Gemma-2-9B-Chinese-Chat](https://huggingface.co/shenzhi-wang/Gemma-2-9B-Chinese-Chat)\]
- **GPT-4 Technical Report**, _OpenAI_, arxiv 2023. \[[paper](https://arxiv.org/abs/2303.08774)\]
- **GPT-4V(ision) System Card**, _OpenAI_, OpenAI blog 2023. \[[paper](https://openai.com/research/gpt-4v-system-card)\]
- **Sparks of Artificial General Intelligence_Early experiments with GPT-4**, _Bubeck et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2303.12712)\]
- **The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)**, _Yang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.17421)\]\[[guidance](https://github.com/guidance-ai/guidance)\]
- **LaMDA: Language Models for Dialog Applications**, _Thoppilan et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2201.08239)\]\[[LaMDA-rlhf-pytorch](https://github.com/conceptofmind/LaMDA-rlhf-pytorch)\]
- **LLaMA: Open and Efficient Foundation Language Models**, _Touvron et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2302.13971)\]\[[code](https://github.com/meta-llama/llama/tree/llama_v1)\]\[[llama.cpp](https://github.com/ggerganov/llama.cpp)\]\[[ollama](https://github.com/jmorganca/ollama)\]\[[llamafile](https://github.com/Mozilla-Ocho/llamafile)\]
- **Llama 2: Open Foundation and Fine-Tuned Chat Models**, _Touvron et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2307.09288)\]\[[code](https://github.com/meta-llama/llama)\]\[[llama2.c](https://github.com/karpathy/llama2.c)\]\[[lit-llama](https://github.com/Lightning-AI/lit-llama)\]\[[litgpt](https://github.com/Lightning-AI/litgpt)\]
- **The Llama 3 Herd of Models**, _Llama Team, AI @ Meta_, 2024. \[[blog](https://ai.meta.com/blog/meta-llama-3-1/)\]\[[paper](https://arxiv.org/abs/2407.21783)\]\[[llama3](https://github.com/meta-llama/llama3)\]\[[llama-models](https://github.com/meta-llama/llama-models)\]\[[llama-recipes](https://github.com/meta-llama/llama-recipes)\]\[[llama-agentic-system](https://github.com/meta-llama/llama-agentic-system)\]\[[llama3-from-scratch](https://github.com/naklecha/llama3-from-scratch)\]\[[nano-llama31](https://github.com/karpathy/nano-llama31)\]
- **TinyLlama: An Open-Source Small Language Model**, _Zhang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.02385)\]\[[code](https://github.com/jzhang38/TinyLlama)\]\[[LiteLlama](https://huggingface.co/ahxt/LiteLlama-460M-1T)\]\[[MobiLlama](https://github.com/mbzuai-oryx/MobiLlama)\]
- **Stanford Alpaca: An Instruction-following LLaMA Model**, _Taori et al._, Stanford blog 2023. \[[paper](https://crfm.stanford.edu/2023/03/13/alpaca.html)\]\[[code](https://github.com/tatsu-lab/stanford_alpaca)\]\[[Alpaca-Lora](https://github.com/tloen/alpaca-lora)\]\[[OpenAlpaca](https://github.com/yxuansu/OpenAlpaca)\]
- **Mistral 7B**, _Jiang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.06825)\]\[[code](https://github.com/mistralai/mistral-inference)\]\[[model](https://huggingface.co/mistralai)\]\[[mistral-finetune](https://github.com/mistralai/mistral-finetune)\]
- **OLMo: Accelerating the Science of Language Models**, _Groeneveld et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.00838)\]\[[code](https://github.com/allenai/OLMo)\]\[[Dolma Dataset](https://github.com/allenai/dolma)\]
- Minerva: **Solving Quantitative Reasoning Problems with Language Models**, _Lewkowycz et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2206.14858)\]
- **PaLM: Scaling Language Modeling with Pathways**, _Chowdhery et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2204.02311)\]\[[PaLM-pytorch](https://github.com/lucidrains/PaLM-pytorch)\]\[[PaLM-rlhf-pytorch](https://github.com/lucidrains/PaLM-rlhf-pytorch)\]\[[PaLM](https://github.com/conceptofmind/PaLM)\]
- **PaLM 2 Technical Report**, _Anil et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.10403)\]
- **PaLM-E: An Embodied Multimodal Language Model**, _Driess et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2303.03378)\]\[[code](https://github.com/kyegomez/PALM-E)\]
- T5: **Exploring the limits of transfer learning with a unified text-to-text transformer**, _Raffel et al._, Journal of Machine Learning Research 2020. \[[paper](https://arxiv.org/abs/1910.10683)\]\[[code](https://github.com/google-research/text-to-text-transfer-transformer)\]\[[t5-pytorch](https://github.com/conceptofmind/t5-pytorch)\]\[[t5-pegasus-pytorch](https://github.com/renmada/t5-pegasus-pytorch)\]
- **BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension**, _Lewis et al._, ACL 2020. \[[paper](https://arxiv.org/abs/1910.13461)\]\[[code](https://github.com/facebookresearch/fairseq/tree/main/examples/bart)\]
- FLAN: **Finetuned Language Models Are Zero-Shot Learners**, _Wei et al._, ICLR 2022. \[[paper](https://arxiv.org/abs/2109.01652)\]\[[code](https://github.com/google-research/flan)\]
- Scaling Flan: **Scaling Instruction-Finetuned Language Models**, _Chung et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2210.11416)\]\[[model](https://github.com/google-research/t5x/blob/main/docs/models.md#flan-t5-checkpoints)\]
- **Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context**, _Dai et al._, ACL 2019. \[[paper](https://arxiv.org/abs/1901.02860)\]\[[code](https://github.com/kimiyoung/transformer-xl)\]
- **XLNet: Generalized Autoregressive Pretraining for Language Understanding**, _Yang et al._, NeurIPS 2019. \[[paper](https://arxiv.org/abs/1906.08237)\]\[[code](https://github.com/zihangdai/xlnet)\]
- **WebGPT: Browser-assisted question-answering with human feedback**, _Nakano et al._, arxiv 2021. \[[paper](https://arxiv.org/abs/2112.09332)\]\[[MS-MARCO-Web-Search](https://github.com/microsoft/MS-MARCO-Web-Search)\]
- **Open Release of Grok-1**, _xAI_, 2024. \[[blog](https://x.ai/blog/grok-os)\]\[[code](https://github.com/xai-org/grok-1)\]\[[model](https://huggingface.co/xai-org/grok-1)\]\[[modelscope](https://modelscope.cn/models/AI-ModelScope/grok-1/summary)\]\[[hpcai-tech/grok-1](https://huggingface.co/hpcai-tech/grok-1)\]\[[dbrx](https://github.com/databricks/dbrx)\]\[[Command R+](https://huggingface.co/CohereForAI/c4ai-command-r-plus)\]\[[snowflake-arctic](https://github.com/Snowflake-Labs/snowflake-arctic)\]

#### 3.2 LLM Application

- **A Watermark for Large Language Models**, _Kirchenbauer et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2301.10226)\]\[[code](https://github.com/jwkirchenbauer/lm-watermarking)\]\[[MarkLLM](https://github.com/THU-BPM/MarkLLM)\]
- **SeqXGPT: Sentence-Level AI-Generated Text Detection**, _Wang et al._, EMNLP 2023. \[[paper](https://arxiv.org/abs/2310.08903)\]\[[code](https://github.com/Jihuai-wpy/SeqXGPT)\]\[[llm-detect-ai](https://github.com/yanqiangmiffy/llm-detect-ai)\]\[[detect-gpt](https://github.com/eric-mitchell/detect-gpt)\]\[[fast-detect-gpt](https://github.com/baoguangsheng/fast-detect-gpt)\]
- **AlpaGasus: Training A Better Alpaca with Fewer Data**, _Chen et al._, ICLR 2024. \[[paper](https://arxiv.org/abs/2307.08701)\]\[[code](https://github.com/gpt4life/alpagasus)\]
- **AutoMix: Automatically Mixing Language Models**, _Madaan et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.12963)\]\[[code](https://github.com/automix-llm/automix)\]
- **ChipNeMo: Domain-Adapted LLMs for Chip Design**, _Liu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.00176)\]\[[semikong](https://github.com/aitomatic/semikong)\]
- **GAIA: A Benchmark for General AI Assistants**, _Mialon et al._, ICLR 2024. \[[paper](https://arxiv.org/abs/2311.12983)\]\[[code](https://huggingface.co/gaia-benchmark)\]
- **HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face**, _Shen et al._, NeurIPS 2023. \[[paper](https://arxiv.org/abs/2303.17580)\]\[[code](https://github.com/microsoft/JARVIS)\]
- **MemGPT: Towards LLMs as Operating Systems**, _Packer et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.08560)\]\[[code](https://github.com/cpacker/MemGPT)\]
- **UFO: A UI-Focused Agent for Windows OS Interaction**, _Zhang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.07939)\]\[[code](https://github.com/microsoft/UFO)\]
- **OS-Copilot: Towards Generalist Computer Agents with Self-Improvement**, _Wu et al._, ICLR 2024. \[[paper](https://arxiv.org/abs/2402.07456)\]\[[code](https://github.com/OS-Copilot/FRIDAY)\]
- **AIOS: LLM Agent Operating System**, _Mei et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.16971)\]\[[code](https://github.com/agiresearch/AIOS)\]
- **DB-GPT: Empowering Database Interactions with Private Large Language Models**, _Xue et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.17449)\]\[[code](https://github.com/eosphoros-ai/DB-GPT)\]\[[DocsGPT](https://github.com/arc53/DocsGPT)\]\[[privateGPT](https://github.com/imartinez/privateGPT)\]\[[localGPT](https://github.com/PromtEngineer/localGPT)\]
- **OpenChat: Advancing Open-source Language Models with Mixed-Quality Data**, _Wang et al._, ICLR 2024. \[[paper](https://arxiv.org/abs/2309.11235)\]\[[code](https://github.com/imoneoi/openchat)\]
- **OpenCodeInterpreter: Integrating Code Generation with Execution and Refinement**, _Zheng et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.14658)\]\[[code](https://github.com/OpenCodeInterpreter/OpenCodeInterpreter)\]\[[code-interpreter](https://github.com/e2b-dev/code-interpreter)\]
- **Orca: Progressive Learning from Complex Explanation Traces of GPT-4**, _Mukherjee et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2306.02707)\]
- **PDFTriage: Question Answering over Long, Structured Documents**, _Saad-Falcon et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.08872)\]\[[code]\]
- **Prompt2Model: Generating Deployable Models from Natural Language Instructions**, _Viswanathan et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2308.12261)\]\[[code](https://github.com/neulab/prompt2model)\]
- **Shepherd: A Critic for Language Model Generation**, _Wang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2308.04592)\]\[[code](https://github.com/facebookresearch/Shepherd)\]
- **Alpaca: A Strong, Replicable Instruction-Following Model**, _Taori et al._, Stanford Blog 2023. \[[paper](https://crfm.stanford.edu/2023/03/13/alpaca.html)\]\[[code](https://github.com/tatsu-lab/stanford_alpaca)\]
- **Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90\%* ChatGPT Quality**, _Chiang et al._, 2023. \[[blog](https://lmsys.org/blog/2023-03-30-vicuna/)\]
- **WizardLM: Empowering Large Language Models to Follow Complex Instructions**, _Xu et al._, ICLR 2024. \[[paper](https://arxiv.org/abs/2304.12244)\]\[[code](https://github.com/nlpxucan/WizardLM)\]
- **WebGLM: Towards An Efficient Web-Enhanced Question Answering System with Human Preferences**, _Liu et al._, KDD 2023. \[[paper](https://arxiv.org/abs/2306.07906)\]\[[code](https://github.com/THUDM/WebGLM)\]\[[AutoWebGLM](https://github.com/THUDM/AutoWebGLM)\]\[[AutoCrawler](https://github.com/EZ-hwh/AutoCrawler)\]\[[gpt-crawler](https://github.com/BuilderIO/gpt-crawler)\]\[[webllama](https://github.com/McGill-NLP/webllama)\]\[[gpt-researcher](https://github.com/assafelovic/gpt-researcher)\]\[[skyvern](https://github.com/Skyvern-AI/skyvern)\]\[[Scrapegraph-ai](https://github.com/VinciGit00/Scrapegraph-ai)\]\[[crawl4ai](https://github.com/unclecode/crawl4ai)\]\[[crawlee-python](https://github.com/apify/crawlee-python)\]\[[Agent-E](https://github.com/EmergenceAI/Agent-E)\]
- **LLM4Decompile: Decompiling Binary Code with Large Language Models**, _Tan et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.05286)\] \[[code](https://github.com/albertan017/LLM4Decompile)\]
- **MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases**, _Liu et al._, ICML 2024. \[[paper](https://arxiv.org/abs/2402.14905)\]\[[code](https://github.com/facebookresearch/MobileLLM)\]
- **The Oscars of AI Theater: A Survey on Role-Playing with Language Models**, _Chen et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2407.11484)\]\[[code](https://github.com/nuochenpku/Awesome-Role-Play-Papers)\]\[[RPBench-Auto](https://boson.ai/rpbench-blog/)\]
- **Apple Intelligence Foundation Language Models**, _Gunter et al._, arxiv 2024. \[[blog](https://machinelearning.apple.com/research/introducing-apple-foundation-models)\]\[[paper](https://arxiv.org/abs/2407.21075)\]

- \[[ray](https://github.com/ray-project/ray)\]\[[dask](https://github.com/dask/dask)\]\[[TaskingAI](https://github.com/TaskingAI/TaskingAI)\]\[[gpt4all](https://github.com/nomic-ai/gpt4all)\]\[[ollama](https://github.com/jmorganca/ollama)\]\[[llama.cpp](https://github.com/ggerganov/llama.cpp)\]\[[dify](https://github.com/langgenius/dify)\]\[[mindsdb](https://github.com/mindsdb/mindsdb)\]\[[bisheng](https://github.com/dataelement/bisheng)\]\[[phidata](https://github.com/phidatahq/phidata)\]\[[guidance](https://github.com/guidance-ai/guidance)\]\[[outlines](https://github.com/outlines-dev/outlines)\]\[[jsonformer](https://github.com/1rgs/jsonformer)\]\[[fabric](https://github.com/danielmiessler/fabric)\]\[[mem0](https://github.com/mem0ai/mem0)\]
- \[[awesome-llm-apps](https://github.com/Shubhamsaboo/awesome-llm-apps)\]\[[fastc](https://github.com/EveripediaNetwork/fastc)\]
- \[[chatgpt-on-wechat](https://github.com/zhayujie/chatgpt-on-wechat)\]\[[HuixiangDou](https://github.com/InternLM/HuixiangDou)\]\[[Streamer-Sales](https://github.com/PeterH0323/Streamer-Sales)\]\[[metahuman-stream](https://github.com/lipku/metahuman-stream)\]\[[aiavatarkit](https://github.com/uezo/aiavatarkit)\]

##### 3.2.1 AI Agent

- **LLM Powered Autonomous Agents**, _Lilian Weng_, 2023. \[[blog](https://lilianweng.github.io/posts/2023-06-23-agent/)\]\[[LLMAgentPapers](https://github.com/zjunlp/LLMAgentPapers)\]\[[LLM-Agents-Papers](https://github.com/AGI-Edgerunners/LLM-Agents-Papers)\]\[[awesome-language-agents](https://github.com/ysymyth/awesome-language-agents)\]\[[Awesome-Papers-Autonomous-Agent](https://github.com/lafmdp/Awesome-Papers-Autonomous-Agent)\]
- **A Survey on Large Language Model based Autonomous Agents**, _Wang et al._, \[[paper](https://arxiv.org/abs/2308.11432)\]\[[code](https://github.com/Paitesanshi/LLM-Agent-Survey)\]
- **The Rise and Potential of Large Language Model Based Agents: A Survey**, _Xi et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.07864)\]\[[code](https://github.com/WooooDyy/LLM-Agent-Paper-List)\]
- **Agent AI: Surveying the Horizons of Multimodal Interaction**, _Durante et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.03568)\]
- **Position Paper: Agent AI Towards a Holistic Intelligence**, _Huang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.00833)\]

- **AgentBench: Evaluating LLMs as Agents**, _Liu et al._, ICLR 2024. \[[paper](https://arxiv.org/abs/2308.03688)\]\[[code](https://github.com/THUDM/AgentBench)\]\[[OSWorld](https://github.com/xlang-ai/OSWorld)\]\[[AgentGym](https://github.com/WooooDyy/AgentGym)\]
- **Agents: An Open-source Framework for Autonomous Language Agents**, _Zhou et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.07870)\]\[[code](https://github.com/aiwaves-cn/agents)\]
- **AutoAgents: A Framework for Automatic Agent Generation**, _Chen et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.17288)\]\[[code](https://github.com/Link-AGI/AutoAgents)\]
- **AgentTuning: Enabling Generalized Agent Abilities for LLMs**, _Zeng et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.12823)\]\[[code](https://github.com/THUDM/AgentTuning)\]
- **AgentVerse: Facilitating Multi-Agent Collaboration and Exploring Emergent Behaviors**, _Chen et al._, ICLR 2024. \[[paper](https://arxiv.org/abs/2308.10848)\]\[[code](https://github.com/OpenBMB/AgentVerse/)\]
- **AppAgent: Multimodal Agents as Smartphone Users**, _Zhang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.13771)\]\[[code](https://github.com/mnotgod96/AppAgent)\]\[[digirl](https://github.com/DigiRL-agent/digirl)\]
- **Mobile-Agent: Autonomous Multi-Modal Mobile Device Agent with Visual Perception**, _Wang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.16158)\]\[[code](https://github.com/X-PLUG/MobileAgent)\]\[[Mobile-Agent-v2](https://arxiv.org/abs/2406.01014)\]
- **Personal LLM Agents: Insights and Survey about the Capability, Efficiency and Security**, _Li et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.05459)\]\[[code](https://github.com/MobileLLM/Personal_LLM_Agents_Survey)\]
- **AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation**, _Wu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2308.08155)\]\[[code](https://github.com/microsoft/autogen)\]
- **CAMEL: Communicative Agents for "Mind" Exploration of Large Language Model Society**, _Li et al._, NeurIPS 2023. \[[paper](https://arxiv.org/abs/2303.17760)\]\[[code](https://github.com/camel-ai/camel)\]\[[crab](https://github.com/camel-ai/crab)\]
- ChatDev: **Communicative Agents for Software Development**, _Qian et al._, ACL 2024. \[[paper](https://arxiv.org/abs/2307.07924)\]\[[code](https://github.com/OpenBMB/ChatDev)\]\[[gpt-pilot](https://github.com/Pythagora-io/gpt-pilot)\]
- **MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework**, _Hong et al._, ICLR 2024 Oral. \[[paper](https://arxiv.org/abs/2308.00352)\]\[[code](https://github.com/geekan/MetaGPT)\]
- **ProAgent: From Robotic Process Automation to Agentic Process Automation**, _Ye et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.10751)\]\[[code](https://github.com/OpenBMB/ProAgent)\]
- **RepoAgent: An LLM-Powered Open-Source Framework for Repository-level Code Documentation Generation**, _Luo et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.16667)\]\[[code](https://github.com/OpenBMB/RepoAgent)\]
- **Generative Agents: Interactive Simulacra of Human Behavior**, _Park et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2304.03442)\]\[[code](https://github.com/joonspk-research/generative_agents)\]\[[GPTeam](https://github.com/101dotxyz/GPTeam)\]
- **CogAgent: A Visual Language Model for GUI Agents**, _Hong et al._, CVPR 2024. \[[paper](https://arxiv.org/abs/2312.08914)\]\[[code](https://github.com/THUDM/CogVLM)\]
- **OpenAgents: An Open Platform for Language Agents in the Wild**, _Xie et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.10634)\]\[[code](https://github.com/xlang-ai/OpenAgents)\]
- **TaskWeaver: A Code-First Agent Framework**, _Qiao et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.17541)\]\[[code](https://github.com/microsoft/TaskWeaver)\]
- **MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge**, _Fan et al._, NeurIPS 2022 Outstanding Paper. \[[paper](https://arxiv.org/abs/2206.08853)\]\[[code](https://github.com/MineDojo/MineDojo)\]
- **Voyager: An Open-Ended Embodied Agent with Large Language Models**, _Wang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.16291)\]\[[code](https://github.com/MineDojo/Voyager)\]
- **Eureka: Human-Level Reward Design via Coding Large Language Models**, _Ma et al._, ICLR 2024. \[[paper](https://arxiv.org/abs/2310.12931)\]\[[code](https://github.com/eureka-research/Eureka)\]\[[DrEureka](https://github.com/eureka-research/DrEureka)\]
- **LEGENT: Open Platform for Embodied Agents**, _Cheng et al._, ACL 2024. \[[paper](https://arxiv.org/abs/2404.18243)\]\[[code](https://github.com/thunlp/LEGENT)\]

- **Mind2Web: Towards a Generalist Agent for the Web**, _Deng et al._, NeurIPS 2023. \[[paper](https://arxiv.org/abs/2306.06070)\]\[[code](https://github.com/OSU-NLP-Group/Mind2Web)\]\[[AutoWebGLM](https://github.com/THUDM/AutoWebGLM)\]
- SeeAct: **GPT-4V(ision) is a Generalist Web Agent, if Grounded**, _Zheng et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.01614)\]\[[code](https://github.com/OSU-NLP-Group/SeeAct)\]
- **Cradle: Empowering Foundation Agents Towards General Computer Control**, _Tan et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.03186)\]\[[code](https://github.com/BAAI-Agents/Cradle)\]
- **AgentScope: A Flexible yet Robust Multi-Agent Platform**, _Gao et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.14034)\]\[[code](https://github.com/modelscope/agentscope)\]\[[modelscope-agent](https://github.com/modelscope/modelscope-agent)\]
- **AgentGym: Evolving Large Language Model-based Agents across Diverse Environments**, _Xi et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2406.04151)\]\[[code](https://github.com/WooooDyy/AgentGym)\]
- **Internet of Agents: Weaving a Web of Heterogeneous Agents for Collaborative Intelligence**, _Chen et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2407.07061)\]\[[code](https://github.com/OpenBMB/IoA)\]
- CLASI: **Towards Achieving Human Parity on End-to-end Simultaneous Speech Translation via LLM Agent**, _ByteDance Research_, 2024. \[[paper](https://arxiv.org/abs/2407.21646)\]\[[translation-agent](https://github.com/andrewyng/translation-agent)\]

- **Foundation Models in Robotics: Applications, Challenges, and the Future**, _Firoozi et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.07843)\]\[[code](https://github.com/robotics-survey/Awesome-Robotics-Foundation-Models)\]
- **Aligning Cyber Space with Physical World: A Comprehensive Survey on Embodied AI**, _Liu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2407.06886)\]\[[code](https://github.com/HCPLab-SYSU/Embodied_AI_Paper_List)\]
- **RT-1: Robotics Transformer for Real-World Control at Scale**, _Brohan et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2212.06817)\]\[[code](https://github.com/google-research/robotics_transformer)\]\[[IRASim](https://github.com/bytedance/IRASim)\]
- **RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control**, _Brohan et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2307.15818)\]\[[Unofficial Implementation](https://github.com/kyegomez/RT-2)\]\[[RT-H: Action Hierarchies Using Language](https://arxiv.org/abs/2403.01823)\]
- **Open X-Embodiment: Robotic Learning Datasets and RT-X Models**, _Open X-Embodiment Collaboration_, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.08864)\]\[[code](https://github.com/google-deepmind/open_x_embodiment)\]
- **Shaping the future of advanced robotics**, Google DeepMind 2024. \[[blog](https://deepmind.google/discover/blog/shaping-the-future-of-advanced-robotics/)\]
- **RoboGen: Towards Unleashing Infinite Data for Automated Robot Learning via Generative Simulation**, _Wang et al._, ICML 2024. \[[paper](https://arxiv.org/abs/2311.01455)\]\[[code](https://github.com/Genesis-Embodied-AI/RoboGen)\]
- **RL-GPT: Integrating Reinforcement Learning and Code-as-policy**, _Liu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.19299)\]
- **Genie: Generative Interactive Environments**, _Bruce et al._, ICML 2024 Best Paper. \[[paper](https://arxiv.org/abs/2402.15391)\]
- **Mobile ALOHA: Learning Bimanual Mobile Manipulation with Low-Cost Whole-Body Teleoperation**, _Fu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.02117)\]\[[code](https://github.com/MarkFzp/mobile-aloha)\]\[[Hardware Code](https://github.com/MarkFzp/mobile-aloha)\]\[[Learning Code](https://github.com/MarkFzp/act-plus-plus)\]\[[UMI](https://github.com/real-stanford/universal_manipulation_interface)\]\[[humanplus](https://github.com/MarkFzp/humanplus)\]\[[TeleVision](https://github.com/OpenTeleVision/TeleVision)\]\[[Surgical Robot Transformer](https://surgical-robot-transformer.github.io/)\]\[[lifelike-agility-and-play](https://github.com/Tencent-RoboticsX/lifelike-agility-and-play)\]
- **Octo: An Open-Source Generalist Robot Policy**, _Ghosh et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2405.12213)\]\[[code](https://github.com/octo-models/octo)\]
- **GRUtopia: Dream General Robots in a City at Scale**, _Wang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2407.10943)\]\[[code](https://github.com/OpenRobotLab/GRUtopia)\]

- \[[LeRobot](https://github.com/huggingface/lerobot)\]\[[DORA](https://github.com/dora-rs/dora)\]\[[awesome-ai-agents](https://github.com/e2b-dev/awesome-ai-agents)\]\[[IsaacLab](https://github.com/isaac-sim/IsaacLab)\]
- \[[AutoGPT](https://github.com/Significant-Gravitas/AutoGPT)\]\[[GPT-Engineer](https://github.com/gpt-engineer-org/gpt-engineer)\]\[[AgentGPT](https://github.com/reworkd/AgentGPT)\]
- \[[BabyAGI](https://github.com/yoheinakajima/babyagi)\]\[[SuperAGI](https://github.com/TransformerOptimus/SuperAGI)\]\[[OpenAGI](https://github.com/agiresearch/OpenAGI)\]
- \[[open-interpreter](https://github.com/KillianLucas/open-interpreter)\]\[[Homepage](https://openinterpreter.com/)\]\[[rawdog](https://github.com/AbanteAI/rawdog)\]\[[OpenCodeInterpreter](https://github.com/OpenCodeInterpreter/OpenCodeInterpreter)\]
- **XAgent: An Autonomous Agent for Complex Task Solving**, \[[blog](https://blog.x-agent.net/blog/xagent/)\]\[[code](https://github.com/OpenBMB/XAgent)\]
- \[[crewAI](https://github.com/joaomdmoura/crewAI)\]\[[PraisonAI](https://github.com/MervinPraison/PraisonAI)\]\[[llama-agents](https://github.com/run-llama/llama-agents)\]\[[phidata](https://github.com/phidatahq/phidata)\]\[[gpt-computer-assistant](https://github.com/onuratakan/gpt-computer-assistant)\]
- \[[translation-agent](https://github.com/andrewyng/translation-agent)\]\[[agent-zero](https://github.com/frdel/agent-zero)\]\[[AgentK](https://github.com/mikekelly/AgentK)\]\[[Twitter Personality](https://github.com/wordware-ai/twitter)\]

##### 3.2.2 Academic

- **Galactica: A Large Language Model for Science**, _Taylor et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2211.09085)\]\[[code](https://github.com/paperswithcode/galai)\]
- **K2: A Foundation Language Model for Geoscience Knowledge Understanding and Utilization**, _Deng et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2306.05064)\]\[[code](https://github.com/davendw49/k2)\]\[[pdf_parser](https://github.com/Acemap/pdf_parser)\]
- **GeoGalactica: A Scientific Large Language Model in Geoscience**, _Lin et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.00434)\]\[[code](https://github.com/geobrain-ai/geogalactica)\]\[[sciparser](https://github.com/davendw49/sciparser)\]
- **Scientific Large Language Models: A Survey on Biological & Chemical Domains**, _Zhang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.14656)\]\[[code](https://github.com/HICAI-ZJU/Scientific-LLM-Survey)\]
- **SciGLM: Training Scientific Language Models with Self-Reflective Instruction Annotation and Tuning**, _Zhang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.07950)\]\[[code](https://github.com/THUDM/SciGLM)\]
- **ChemLLM: A Chemical Large Language Model**, _Zhang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.06852)\]\[[model](https://huggingface.co/AI4Chem/ChemLLM-7B-Chat)\]
- **LangCell: Language-Cell Pre-training for Cell Identity Understanding**, _Zhao et al._, ICML 2024. \[[paper](https://arxiv.org/abs/2405.06708)\]\[[code](https://github.com/PharMolix/LangCell)\]\[[scFoundation](https://github.com/biomap-research/scFoundation)\]
- **SPIQA: A Dataset for Multimodal Question Answering on Scientific Papers**, _Pramanick et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2407.09413)\]\[[code](https://github.com/google/spiqa)\]
- STORM: **Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models**, _Shao et al._, NAACL 2024. \[[paper](https://arxiv.org/abs/2402.14207)\]\[[code](https://github.com/stanford-oval/storm)\]
- **Automated Peer Reviewing in Paper SEA: Standardization, Evaluation, and Analysis**, _Yu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2407.12857)\]\[[code](https://github.com/ecnu-sea/sea)\]
- **The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery**, _Lu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2408.06292)\]\[[code](https://github.com/SakanaAI/AI-Scientist)\]

- \[[Awesome-Scientific-Language-Models](https://github.com/yuzhimanhua/Awesome-Scientific-Language-Models)\]\[[gpt_academic](https://github.com/binary-husky/gpt_academic)\]\[[ChatPaper](https://github.com/kaixindelele/ChatPaper)\]\[[scispacy](https://github.com/allenai/scispacy)\]\[[awesome-ai4s](https://github.com/hyperai/awesome-ai4s)\]

##### 3.2.3 Code

- **Neural code generation**, CMU 2024 Spring. \[[link](https://cmu-codegen.github.io/s2024/)\]
- **Unifying the Perspectives of NLP and Software Engineering: A Survey on Language Models for Code**, _Zhang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.07989)\]\[[Awesome-Code-LLM](https://github.com/codefuse-ai/Awesome-Code-LLM)\]\[[MFTCoder](https://github.com/codefuse-ai/MFTCoder)\]\[[Awesome-Code-LLM](https://github.com/huybery/Awesome-Code-LLM)\]
- **Source Code Data Augmentation for Deep Learning: A Survey**, _Zhuo et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.19915)\]\[[code](https://github.com/terryyz/DataAug4Code)\]

- Codex: **Evaluating Large Language Models Trained on Code**, _Chen et al._, arxiv 2021. \[[paper](https://arxiv.org/abs/2107.03374)\]\[[human-eval](https://github.com/openai/human-eval)\]\[[CriticGPT](https://openai.com/index/finding-gpt4s-mistakes-with-gpt-4/)\]\[[On scalable oversight with weak LLMs judging strong LLMs](https://arxiv.org/abs/2407.04622)\]
- **Code Llama: Open Foundation Models for Code**, _Rozière et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2308.12950)\]\[[code](https://github.com/meta-llama/codellama)\]\[[model](https://huggingface.co/codellama)\]\[[llamacoder](https://github.com/Nutlope/llamacoder)\]
- **CodeGemma: Open Code Models Based on Gemma**, \[[blog](https://huggingface.co/blog/codegemma)\]\[[report](https://storage.googleapis.com/deepmind-media/gemma/codegemma_report.pdf)\]
- AlphaCode: **Competition-Level Code Generation with AlphaCode**, _Li et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2203.07814)\]\[[dataset](https://github.com/google-deepmind/code_contests)\]\[[AlphaCode2_Tech_Report](https://storage.googleapis.com/deepmind-media/AlphaCode2/AlphaCode2_Tech_Report.pdf)\]
- **CodeGeeX: A Pre-Trained Model for Code Generation with Multilingual Evaluations on HumanEval-X**, _Zheng et al._, KDD 2023. \[[paper](https://arxiv.org/abs/2303.17568)\]\[[code](https://github.com/THUDM/CodeGeeX)\]\[[CodeGeeX2](https://github.com/THUDM/CodeGeeX2)\]\[[CodeGeeX4](https://github.com/THUDM/CodeGeeX4)\]
- **CodeGen: An Open Large Language Model for Code with Multi-Turn Program Synthesis**, _Nijkamp et al._, ICLR 2022. \[[paper](https://arxiv.org/abs/2203.13474)\]\[[code](https://github.com/salesforce/CodeGen)\]
- **CodeGen2: Lessons for Training LLMs on Programming and Natural Languages**, _Nijkamp et al._, ICLR 2023. \[[paper](https://arxiv.org/abs/2305.02309)\]\[[code](https://github.com/salesforce/CodeGen2)\]
- **CodeChain: Towards Modular Code Generation Through Chain of Self-revisions with Representative Sub-modules**, _Le et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.08992)\]\[[code](https://github.com/SalesforceAIResearch/CodeChain)\]
- **StarCoder: may the source be with you**, _Li et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.06161)\]\[[code](https://github.com/bigcode-project/starcoder)\]\[[bigcode-project](https://github.com/bigcode-project)\]\[[model](https://huggingface.co/bigcode)\]
- **StarCoder 2 and The Stack v2: The Next Generation**, _Lozhkov et al._, 2024. \[[paper](https://arxiv.org/abs/2402.19173)\]\[[code](https://github.com/bigcode-project/starcoder2)\]\[[starcoder.cpp](https://github.com/bigcode-project/starcoder.cpp)\]
- **WizardCoder: Empowering Code Large Language Models with Evol-Instruct**, _Luo et al._, ICLR 2024. \[[paper](https://arxiv.org/abs/2306.08568)\]\[[code](https://github.com/nlpxucan/WizardLM/tree/main/WizardCoder)\]
- **Magicoder: Source Code Is All You Need**, _Wei et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.02120)\]\[[code](https://github.com/ise-uiuc/magicoder)\]
- **Code Generation with AlphaCodium: From Prompt Engineering to Flow Engineering**, _Ridnik et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.08500)\]\[[code](https://github.com/Codium-ai/AlphaCodium)\]\[[pr-agent](https://github.com/Codium-ai/pr-agent)\]\[[cover-agent](https://github.com/Codium-ai/cover-agent)\]
- **DeepSeek-Coder: When the Large Language Model Meets Programming -- The Rise of Code Intelligence**, _Guo et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.14196)\]\[[code](https://github.com/deepseek-ai/DeepSeek-Coder)\]
- **DeepSeek-Coder-V2: Breaking the Barrier of Closed-Source Models in Code Intelligence**, _Zhu et al._, CoRR 2024. \[[paper](https://arxiv.org/abs/2406.11931)\]\[[code](https://github.com/deepseek-ai/DeepSeek-Coder-V2)\]
- **If LLM Is the Wizard, Then Code Is the Wand: A Survey on How Code Empowers Large Language Models to Serve as Intelligent Agents**, _Yang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.00812)\]
- **Design2Code: How Far Are We From Automating Front-End Engineering?**, _Si et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.03163)\]\[[code](https://github.com/NoviScl/Design2Code)\]
- **AutoCoder: Enhancing Code Large Language Model with AIEV-Instruct**, _Lei et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2405.14906)\]\[[code](https://github.com/bin123apple/AutoCoder)\]
- **SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering**, _Yang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2405.15793)\]\[[code](https://github.com/princeton-nlp/SWE-agent)\]\[[swe-bench-technical-report](https://www.cognition-labs.com/post/swe-bench-technical-report)\]\[[CodeR](https://github.com/NL2Code/CodeR)\]
- **Agentless: Demystifying LLM-based Software Engineering Agents**, _Xia et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2407.01489)\]\[[code](https://github.com/OpenAutoCoder/Agentless)\]
- **BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions**, _Zhuo et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2406.15877)\]\[[code](https://github.com/bigcode-project/bigcodebench)\]
- **OpenDevin: An Open Platform for AI Software Developers as Generalist Agents**, _Wang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2407.16741)\]\[[code](https://github.com/OpenDevin/OpenDevin)\]

- \[[CodeQwen1.5](https://github.com/QwenLM/CodeQwen1.5)\]\[[aiXcoder-7B](https://github.com/aixcoder-plugin/aiXcoder-7B)\]\[[codealpaca](https://github.com/sahil280114/codealpaca)\]
- \[[OpenDevin](https://github.com/OpenDevin/OpenDevin)\]\[[devika](https://github.com/stitionai/devika)\]\[[auto-code-rover](https://github.com/nus-apr/auto-code-rover)\]\[[developer](https://github.com/smol-ai/developer)\]\[[aider](https://github.com/paul-gauthier/aider)\]\[[claude-engineer](https://github.com/Doriandarko/claude-engineer)\]\[[SuperCoder](https://github.com/TransformerOptimus/SuperCoder)\]
- \[[screenshot-to-code](https://github.com/abi/screenshot-to-code)\]\[[vanna](https://github.com/vanna-ai/vanna)\]\[[NL2SQL_Handbook](https://github.com/HKUSTDial/NL2SQL_Handbook)\]

##### 3.2.4 Financial Application

- **DocLLM: A layout-aware generative language model for multimodal document understanding**, _Wang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.00908)\]
- **DocGraphLM: Documental Graph Language Model for Information Extraction**, _Wang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2401.02823)\]
- **FinBERT: A Pretrained Language Model for Financial Communications**, _Yang et al._, arxiv 2020. \[[paper](https://arxiv.org/abs/2006.08097)\]\[[Wiley paper](https://onlinelibrary.wiley.com/doi/full/10.1111/1911-3846.12832)\]\[[code](https://github.com/yya518/FinBERT)\]\[[finBERT](https://github.com/ProsusAI/finBERT)\]\[[valuesimplex/FinBERT](https://github.com/valuesimplex/FinBERT)\]
- **FinGPT: Open-Source Financial Large Language Models**, _Yang et al._, IJCAI 2023. \[[paper](https://arxiv.org/abs/2306.06031)\]\[[code](https://github.com/AI4Finance-Foundation/FinGPT)\]
- **FinRobot: An Open-Source AI Agent Platform for Financial Applications using Large Language Models**, _Yang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2405.14767)\]\[[code](https://github.com/AI4Finance-Foundation/FinRobot)\]
- **FinGPT: Instruction Tuning Benchmark for Open-Source Large Language Models in Financial Datasets**, _Wang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.04793)\]\[[code](https://github.com/AI4Finance-Foundation/FinGPT)\]
- **Instruct-FinGPT: Financial Sentiment Analysis by Instruction Tuning of General-Purpose Large Language Models**, _Zhang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2306.12659)\]\[[code](https://github.com/AI4Finance-Foundation/FinGPT/tree/master/fingpt/FinGPT_RAG/instruct-FinGPT)\]
- **FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance**, _Liu et al._, arxiv 2020. \[[paper](https://arxiv.org/abs/2011.09607)\]\[[code](https://github.com/AI4Finance-Foundation/FinRL)\]
- **FinRL-Meta: Market Environments and Benchmarks for Data-Driven Financial Reinforcement Learning**, _Liu et al._, NeurIPS 2022. \[[paper](https://arxiv.org/abs/2211.03107)\]\[[code](https://github.com/AI4Finance-Foundation/FinRL-Meta)\]
- **DISC-FinLLM: A Chinese Financial Large Language Model based on Multiple Experts Fine-tuning**, _Chen et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.15205)\]\[[code](https://github.com/FudanDISC/DISC-FinLLM)\]
- **A Multimodal Foundation Agent for Financial Trading: Tool-Augmented, Diversified, and Generalist**, _Zhang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.18485)\]
- **XuanYuan 2.0: A Large Chinese Financial Chat Model with Hundreds of Billions Parameters**, _Zhang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.12002)\]\[[code](https://github.com/Duxiaoman-DI/XuanYuan)\]\[[PIXIU](https://github.com/The-FinAI/PIXIU)\]
- **StructGPT: A General Framework for Large Language Model to Reason over Structured Data**, _Jiang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.09645)\]\[[code](https://github.com/RUCAIBox/StructGPT)\]
- **Large Language Model for Table Processing: A Survey**, _Lu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.05121)\]\[[llm-table-survey](https://github.com/godaai/llm-table-survey)\]\[[table-transformer](https://github.com/microsoft/table-transformer)\]\[[Awesome-Tabular-LLMs](https://github.com/SpursGoZmy/Awesome-Tabular-LLMs)\]\[[Awesome-LLM-Tabular](https://github.com/johnnyhwu/Awesome-LLM-Tabular)\]\[[Table-LLaVA](https://github.com/SpursGoZmy/Table-LLaVA)\]
- **rLLM: Relational Table Learning with LLMs**, _Li et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2407.20157)\]\[[code](https://github.com/rllm-team/rllm)\]
- **Data-Copilot: Bridging Billions of Data and Humans with Autonomous Workflow**, _Zhang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2306.07209)\]\[[code](https://github.com/zwq2018/Data-Copilot)\]
- **Data Interpreter: An LLM Agent For Data Science**, _Hong et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.18679)\]\[[code](https://github.com/geekan/MetaGPT/tree/main/examples/di)\]
- **AlphaFin: Benchmarking Financial Analysis with Retrieval-Augmented Stock-Chain Framework**, _Li et al._, COLING 2024. \[[paper](https://arxiv.org/abs/2403.12582)\]\[[code](https://github.com/AlphaFin-proj/AlphaFin)\]
- **LLMFactor: Extracting Profitable Factors through Prompts for Explainable Stock Movement Prediction**, _Wang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2406.10811)\]
- **A Survey of Large Language Models in Finance (FinLLMs)**, _Lee et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.02315)\]\[[code](https://github.com/adlnlp/FinLLMs)\]\[[Revolutionizing Finance with LLMs: An Overview of Applications and Insights](https://arxiv.org/abs/2401.11641)\]
- **A Survey of Large Language Models for Financial Applications: Progress, Prospects and Challenges**, _Nie et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2406.11903)\]
- **PEER: Expertizing Domain-Specific Tasks with a Multi-Agent Framework and Tuning Methods**, _Wang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2407.06985)\]\[[code](https://github.com/alipay/agentUniverse)\]\[[Stockagent](https://github.com/MingyuJ666/Stockagent)\]
- **Benchmarking Large Language Models on CFLUE -- A Chinese Financial Language Understanding Evaluation Dataset**, _Zhu et al._, ACL 2024. \[[paper](https://arxiv.org/abs/2405.10542)\]\[[code](https://github.com/aliyun/cflue)\]

- \[[gpt-investor](https://github.com/mshumer/gpt-investor)\]\[[FinGLM](https://github.com/MetaGLM/FinGLM)\]\[[agentUniverse](https://github.com/alipay/agentUniverse)\]\[[gs-quant](https://github.com/goldmansachs/gs-quant)\]\[[stockbot-on-groq](https://github.com/bklieger-groq/stockbot-on-groq)\]\[[Real-Time-Stock-Market-Prediction-using-Ensemble-DL-and-Rainbow-DQN](https://github.com/THINK989/Real-Time-Stock-Market-Prediction-using-Ensemble-DL-and-Rainbow-DQN)\]

##### 3.2.5 Information Retrieval

- **ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT**, _Khattab et al._, SIGIR 2020. \[[paper](https://arxiv.org/abs/2004.12832)\]\[[simbert](https://github.com/ZhuiyiTechnology/simbert)\]\[[roformer-sim](https://github.com/ZhuiyiTechnology/roformer-sim)\]
- **ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction**, _Santhanam et al._, NAACL 2022. \[[paper](https://arxiv.org/abs/2112.01488)\]\[[code](https://github.com/stanford-futuredata/ColBERT)\]\[[RAGatouille](https://github.com/bclavie/RAGatouille)\]\[[A Reproducibility Study of PLAID](https://arxiv.org/abs/2404.14989)\]
- **ColBERT-XM: A Modular Multi-Vector Representation Model for Zero-Shot Multilingual Information Retrieval**, _Louis et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.15059)\]\[[code](https://github.com/ant-louis/xm-retrievers)\]\[[model](https://huggingface.co/antoinelouis/colbert-xm)\]
- HyDE: **Precise Zero-Shot Dense Retrieval without Relevance Labels**, _Gao et al._, ACL 2023. \[[paper](https://arxiv.org/abs/2212.10496)\]\[[code](https://github.com/texttron/hyde)\]
- **Query2doc: Query Expansion with Large Language Models**, _Wang et al._, EMNLP 2023. \[[paper](https://arxiv.org/abs/2303.07678)\]\[[Query Expansion by Prompting Large Language Models](https://arxiv.org/abs/2305.03653)\]
- RankGPT: **Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents**, _Sun et al._, EMNLP 2023 Outstanding Paper. \[[paper](https://arxiv.org/abs/2304.09542)\]\[[code](https://github.com/sunnweiwei/RankGPT)\]
- **Large Language Models for Information Retrieval: A Survey**, _Zhu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2308.07107)\]\[[code](https://github.com/RUC-NLPIR/LLM4IR-Survey)\]\[[YuLan-IR](https://github.com/RUC-GSAI/YuLan-IR)\]
- **Large Language Models for Generative Information Extraction: A Survey**, _Xu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.17617)\]\[[code](https://github.com/quqxui/Awesome-LLM4IE-Papers)\]\[[UIE](https://github.com/universal-ie/UIE)\]\[[NERRE](https://github.com/LBNLP/NERRE)\]\[[uie_pytorch](https://github.com/HUSTAI/uie_pytorch)\]
- LLaRA: **Making Large Language Models A Better Foundation For Dense Retrieval**, _Li et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.15503)\]\[[code](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_reranker)\]
- **UniGen: A Unified Generative Framework for Retrieval and Question Answering with Large Language Models**, _Li et al._, AAAI 2024. \[[paper](https://arxiv.org/abs/2312.11036)\]
- **INTERS: Unlocking the Power of Large Language Models in Search with Instruction Tuning**, _Zhu et al._, ACL 2024. \[[paper](https://arxiv.org/abs/2401.06532)\]\[[code](https://github.com/DaoD/INTERS)\]
- GenIR: **From Matching to Generation: A Survey on Generative Information Retrieval**, _Li et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2404.14851)\]\[[code](https://github.com/RUC-NLPIR/GenIR-Survey)\]
- **D2LLM: Decomposed and Distilled Large Language Models for Semantic Search**, _Liao et al._, ACL 2024. \[[paper](https://arxiv.org/abs/2406.17262)\]\[[code](https://github.com/codefuse-ai/D2LLM)\]
- **BM25S: Orders of magnitude faster lexical search via eager sparse scoring**, _Xing Han Lù_, arxiv 2024. \[[paper](https://arxiv.org/abs/2407.03618)\]\[[code](https://github.com/xhluca/bm25s)\]\[[rank_bm25](https://github.com/dorianbrown/rank_bm25)\]
- **MindSearch: Mimicking Human Minds Elicits Deep AI Searcher**, _Chen et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2407.20183)\]\[[code](https://github.com/InternLM/MindSearch)\]

- **SIGIR-AP 2023 Tutorial: Recent Advances in Generative Information Retrieval** \[[link](https://sigir-ap2023-generative-ir.github.io/)\]
- **SIGIR 2024 Tutorial: Large Language Model Powered Agents for Information Retrieval** \[[link](https://llmagenttutorial.github.io/sigir2024)\]
- \[[search_with_lepton](https://github.com/leptonai/search_with_lepton)\]\[[LLocalSearch](https://github.com/nilsherzig/LLocalSearch)\]\[[FreeAskInternet](https://github.com/nashsu/FreeAskInternet)\]\[[storm](https://github.com/stanford-oval/storm)\]\[[searxng](https://github.com/searxng/searxng)\]\[[Perplexica](https://github.com/ItzCrazyKns/Perplexica)\]\[[rag-search](https://github.com/thinkany-ai/rag-search)\]
- \[[similarities](https://github.com/shibing624/similarities)\]\[[text2vec](https://github.com/shibing624/text2vec)\]


##### 3.2.6 Math

- **ToRA: A Tool-Integrated Reasoning Agent for Mathematical Problem Solving**, _Gou et al._, ICLR 2024. \[[paper](https://arxiv.org/abs/2309.17452)\]\[[code](https://github.com/microsoft/ToRA)\]
- **MathVista: Evaluating Math Reasoning in Visual Contexts with GPT-4V, Bard, and Other Large Multimodal Models**, _Lu et al._, ICLR 2024 Oral. \[[paper](https://arxiv.org/abs/2310.02255)\]\[[code](https://github.com/lupantech/MathVista)\]\[[MathBench](https://github.com/open-compass/MathBench)\]\[[OlympiadBench](https://github.com/OpenBMB/OlympiadBench)\]
- **InternLM-Math: Open Math Large Language Models Toward Verifiable Reasoning**, _Ying et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.06332)\]\[[code](https://github.com/InternLM/InternLM-Math)\]
- **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models**, _Shao et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.03300)\]\[[code](https://github.com/deepseek-ai/DeepSeek-Math)\]\[[Qwen2-Math](https://github.com/QwenLM/Qwen2-Math)\]
- **Common 7B Language Models Already Possess Strong Math Capabilities**, _Li et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.04706)\]\[[code](https://github.com/Xwin-LM/Xwin-LM/tree/main/Xwin-Math)\]
- **ChatGLM-Math: Improving Math Problem-Solving in Large Language Models with a Self-Critique Pipeline**, _Xu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2404.02893)\]\[[code](https://github.com/THUDM/ChatGLM-Math)\]
- **AlphaMath Almost Zero: process Supervision without process**, _Chen et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2405.03553)\]\[[code](https://github.com/MARIO-Math-Reasoning/Super_MARIO)\]
- **JiuZhang3.0: Efficiently Improving Mathematical Reasoning by Training Small Data Synthesis Models**, _Zhou et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2405.14365)\]\[[code](https://github.com/RUCAIBox/JiuZhang3.0)\]
- **Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B**, _Zhang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2406.07394)\]\[[code](https://github.com/trotsky1997/MathBlackBox)\]
- **Math-LLaVA: Bootstrapping Mathematical Reasoning for Multimodal Large Language Models**, _Shi et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2406.17294)\]\[[code](https://github.com/HZQ950419/Math-LLaVA)\]
- **We-Math: Does Your Large Multimodal Model Achieve Human-like Mathematical Reasoning?**, _Qiao et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2407.01284)\]\[[code](https://github.com/We-Math/We-Math)\]
- **MAVIS: Mathematical Visual Instruction Tuning**, _Zhang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2407.08739)\]\[[code](https://github.com/ZrrSkywalker/MAVIS)\]

- **AI Mathematical Olympiad - Progress Prize 1**, Kaggle Competition 2024. \[[Numina 1st Place Solution](https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/discussion/519303)\]\[[project-numina/aimo-progress-prize](https://github.com/project-numina/aimo-progress-prize)\]\[[How NuminaMath Won the 1st AIMO Progress Prize](https://huggingface.co/blog/winning-aimo-progress-prize)\]\[[NuminaMath-7B-TIR](https://huggingface.co/AI-MO/NuminaMath-7B-TIR)\]\[[AI achieves silver-medal standard solving International Mathematical Olympiad problems](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/)\]

##### 3.2.7 Medicine and Law

- **A Survey of Large Language Models in Medicine: Progress, Application, and Challenge**, _Zhou et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.05112)\]\[[code](https://github.com/AI-in-Health/MedLLMsPracticalGuide)\]\[[LLM-for-Healthcare](https://github.com/KaiHe-better/LLM-for-Healthcare)\]\[[GMAI-MMBench](https://github.com/uni-medical/GMAI-MMBench)\]
- **A Survey on Large Language Models for Critical Societal Domains: Finance, Healthcare, and Law**, _Chen et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2405.01769)\]\[[code](https://github.com/czyssrs/LLM_X_papers)\]
- **HuatuoGPT, towards Taming Language Model to Be a Doctor**, _Zhang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.15075)\]\[[code](https://github.com/FreedomIntelligence/HuatuoGPT)\]\[[Medical_NLP](https://github.com/FreedomIntelligence/Medical_NLP)\]\[[Zhongjing](https://github.com/SupritYoung/Zhongjing)\]\[[MedicalGPT](https://github.com/shibing624/MedicalGPT)\]\[[huatuogpt-vision](https://github.com/freedomintelligence/huatuogpt-vision)\]\[[Chain-of-Diagnosis](https://github.com/FreedomIntelligence/Chain-of-Diagnosis)\]
- **Chatlaw: A Multi-Agent Collaborative Legal Assistant with Knowledge Graph Enhanced Mixture-of-Experts Large Language Model**, _Cui et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2306.16092)\]\[[code](https://github.com/PKU-YuanGroup/ChatLaw)\]
- **DISC-LawLLM: Fine-tuning Large Language Models for Intelligent Legal Services**, _Yue et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.11325)\]\[[code](https://github.com/FudanDISC/DISC-LawLLM)\]
- **DISC-MedLLM: Bridging General Large Language Models and Real-World Medical Consultation**, _Bao et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2308.14346)\]\[[code](https://github.com/FudanDISC/DISC-MedLLM)\]
- **MedAgents: Large Language Models as Collaborators for Zero-shot Medical Reasoning**, _Tang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.10537)\]\[[code](https://github.com/gersteinlab/MedAgents)\]
- **MEDITRON-70B: Scaling Medical Pretraining for Large Language Models**, _Chen et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.16079)\]\[[meditron](https://github.com/epfLLM/meditron)\]
- Med-PaLM: **Large language models encode clinical knowledge**, _Singhal et al._, Nature 2023. \[[paper](https://www.nature.com/articles/s41586-023-06291-2)\]\[[Unofficial Implementation](https://github.com/kyegomez/Med-PaLM)\]
- **Capabilities of Gemini Models in Medicine**, _Saab et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2404.18416)\]
- AMIE: **Towards Conversational Diagnostic AI**, _Tu et al._, arxiv 2024.  \[[paper](https://arxiv.org/abs/2401.05654)\]\[[AMIE-pytorch](https://github.com/lucidrains/AMIE-pytorch)\]
- **Apollo: Lightweight Multilingual Medical LLMs towards Democratizing Medical AI to 6B People**, _Wang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.03640)\]\[[code](https://github.com/FreedomIntelligence/Apollo)\]\[[Medical_NLP](https://github.com/FreedomIntelligence/Medical_NLP)\]
- **Agent Hospital: A Simulacrum of Hospital with Evolvable Medical Agents**, _Li et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2405.02957)\]

- \[[openfold](https://github.com/aqlaboratory/openfold)\]\[[alphafold3-pytorch](https://github.com/lucidrains/alphafold3-pytorch)\]\[[AlphaFold3](https://github.com/kyegomez/AlphaFold3)\]\[[LucaOne](https://github.com/LucaOne/LucaOne)\]\[[esm](https://github.com/evolutionaryscale/esm)\]\[[AlphaPPImd](https://github.com/AspirinCode/AlphaPPImd)\]\[[visual-med-alpaca](https://github.com/cambridgeltl/visual-med-alpaca)\]

##### 3.2.8 Recommend System

- DIN: **Deep Interest Network for Click-Through Rate Prediction**, _Zhou et al._, KDD 2018. \[[paper](https://arxiv.org/abs/1706.06978)\]\[[code](https://github.com/zhougr1993/DeepInterestNetwork)\]\[[DIEN](https://github.com/mouna99/dien)\]\[[x-deeplearning](https://github.com/alibaba/x-deeplearning)\]
- MMoE: **Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts**, _Ma et al._, KDD 2018. \[[paper](https://dl.acm.org/doi/pdf/10.1145/3219819.3220007)\]\[[DeepCTR-Torch](https://github.com/shenweichen/DeepCTR-Torch)\]\[[pytorch-mmoe](https://github.com/ZhichenZhao/pytorch-mmoe)\]
- **Recommender Systems with Generative Retrieval**, _Rajput et al._, NeurIPS 2023. \[[paper](https://arxiv.org/abs/2305.05065)\]
- **Unifying Large Language Models and Knowledge Graphs: A Roadmap**, _Pan et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2306.08302)\]
- YuLan-Rec: **User Behavior Simulation with Large Language Model based Agents**, _Wang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2306.02552)\]\[[code](https://github.com/RUC-GSAI/YuLan-Rec)\]
- **SSLRec: A Self-Supervised Learning Framework for Recommendation**, _Ren et al._, WSDM 2024 Oral. \[[paper](https://arxiv.org/abs/2308.05697)\]\[[code](https://github.com/HKUDS/SSLRec)\]\[[Awesome-SSLRec-Papers](https://github.com/HKUDS/Awesome-SSLRec-Papers)\]
- RLMRec: **Representation Learning with Large Language Models for Recommendation**, _Ren et al._, WWW 2024. \[[paper](https://arxiv.org/abs/2310.15950)\]\[[code](https://github.com/HKUDS/RLMRec)\]
- **LLMRec: Large Language Models with Graph Augmentation for Recommendation**, _Wei et al._, WSDM 2024 Oral. \[[paper](https://arxiv.org/abs/2311.00423)\]\[[code](https://github.com/HKUDS/LLMRec)\]
- **XRec: Large Language Models for Explainable Recommendation**, _Ma et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2406.02377)\]\[[code](https://github.com/HKUDS/XRec)\]\[[SelfGNN](https://github.com/HKUDS/SelfGNN)\]
- **Agent4Rec_On Generative Agents in Recommendation**, _Zhang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.10108)\]\[[code](https://github.com/LehengTHU/Agent4Rec)\]
- LLM-KERec: **Breaking the Barrier: Utilizing Large Language Models for Industrial Recommendation Systems through an Inferential Knowledge Graph**, _Zhao et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.13750)\]
- **Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations**, _Zhai et al._, ICML 2024. \[[paper](https://arxiv.org/abs/2402.17152)\]\[[code](https://github.com/facebookresearch/generative-recommenders)\]
- **Wukong: Towards a Scaling Law for Large-Scale Recommendation**, _Zhang et al._, ICML 2024. \[[paper](https://arxiv.org/abs/2403.02545)\]\[[unofficial code](https://github.com/clabrugere/wukong-recommendation)\]
- **RecAI: Leveraging Large Language Models for Next-Generation Recommender Systems**, _Lian et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.06465)\]\[[code](https://github.com/microsoft/RecAI)\]
- **Knowledge Adaptation from Large Language Model to Recommendation for Practical Industrial Application**, _Jia et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2405.03988)\]

- \[[recommenders](https://github.com/recommenders-team/recommenders)\]\[[Source code for Twitter's Recommendation Algorithm](https://github.com/twitter/the-algorithm)\]\[[Awesome-RSPapers](https://github.com/RUCAIBox/Awesome-RSPapers)\]\[[RecBole](https://github.com/RUCAIBox/RecBole)\]\[[RecSysDatasets](https://github.com/RUCAIBox/RecSysDatasets)\]\[[LLM4Rec-Awesome-Papers](https://github.com/WLiK/LLM4Rec-Awesome-Papers)\]\[[Awesome-LLM-for-RecSys](https://github.com/CHIANGEL/Awesome-LLM-for-RecSys)\]\[[Awesome-LLM4RS-Papers](https://github.com/nancheng58/Awesome-LLM4RS-Papers)\]\[[ReChorus](https://github.com/THUwangcy/ReChorus)\]

##### 3.2.9 Tool Learning

- **Tool Learning with Foundation Models**, _Qin et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2304.08354)\]\[[code](https://github.com/OpenBMB/BMTools)\]
- **Tool Learning with Large Language Models: A Survey**, _Qu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2405.17935)\]\[[code](https://github.com/quchangle1/LLM-Tool-Survey)\]
- **Toolformer: Language Models Can Teach Themselves to Use Tools**, _Schick et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2302.04761)\]\[[toolformer-pytorch](https://github.com/lucidrains/toolformer-pytorch)\]\[[conceptofmind/toolformer](https://github.com/conceptofmind/toolformer)\]\[[xrsrke/toolformer](https://github.com/xrsrke/toolformer)\]\[[Graph_Toolformer](https://github.com/jwzhanggy/Graph_Toolformer)\]
- **ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs**, _Qin et al._, ICLR 2024 Spotlight. \[[paper](https://arxiv.org/abs/2307.16789)\]\[[code](https://github.com/OpenBMB/ToolBench)\]\[[StableToolBench](https://github.com/THUNLP-MT/StableToolBench)\]
- **Gorilla: Large Language Model Connected with Massive APIs**, _Patil et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.15334)\]\[[code](https://github.com/ShishirPatil/gorilla)\]
- **GPT4Tools: Teaching Large Language Model to Use Tools via Self-instruction**, _Yang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.18752)\]\[[code](https://github.com/AILab-CVC/GPT4Tools)\]
- **RestGPT: Connecting Large Language Models with Real-World RESTful APIs**, _Song et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2306.06624)\]\[[code](https://github.com/Yifan-Song793/RestGPT)\]
- LLMCompiler: **An LLM Compiler for Parallel Function Calling**, _Kim et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.04511)\]\[[code](https://github.com/SqueezeAILab/LLMCompiler)\]
- **Large Language Models as Tool Makers**, _Cai et al_, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.17126)\]\[[code](https://github.com/ctlllll/LLM-ToolMaker)\]
- **ToolAlpaca: Generalized Tool Learning for Language Models with 3000 Simulated Cases** _Tang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2306.05301)\]\[[code](https://github.com/tangqiaoyu/ToolAlpaca)\]\[[ToolQA](https://github.com/night-chen/ToolQA)\]\[[toolbench](https://github.com/sambanova/toolbench)\]
- **ToolChain\*: Efficient Action Space Navigation in Large Language Models with A\* Search**, _Zhuang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.13227)\]\[[code]\]
- **Chameleon: Plug-and-Play Compositional Reasoning with Large Language Models**, _Lu et al._, NeurIPS 2023. \[[paper](https://arxiv.org/abs/2304.09842)\]\[[code](https://github.com/lupantech/chameleon-llm)\]
- **ToolEyes: Fine-Grained Evaluation for Tool Learning Capabilities of Large Language Models in Real-world Scenarios**, _Ye et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.00741)\]\[[code](https://github.com/Junjie-Ye/ToolEyes)\]
- **AnyTool: Self-Reflective, Hierarchical Agents for Large-Scale API Calls**, _Du et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.04253)\]\[[code](https://github.com/dyabel/AnyTool)\]
- **LLMs in the Imaginarium: Tool Learning through Simulated Trial and Error**, _Wang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.04746)\]\[[code](https://github.com/microsoft/simulated-trial-and-error)\]
- **What Are Tools Anyway? A Survey from the Language Model Perspective**, _Wang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.15452)\]
- **ToolSandbox: A Stateful, Conversational, Interactive Evaluation Benchmark for LLM Tool Use Capabilities**, _Lu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2408.04682)\]\[[code](https://github.com/apple/ToolSandbox)\]
- **Re-Invoke: Tool Invocation Rewriting for Zero-Shot Tool Retrieval**, _Chen et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2408.01875)\]

- \[[functionary](https://github.com/MeetKai/functionary)\]\[[ToolLearningPapers](https://github.com/thunlp/ToolLearningPapers)\]\[[awesome-tool-llm](https://github.com/zorazrw/awesome-tool-llm)\]

#### 3.3 LLM Technique

- **How to Train Really Large Models on Many GPUs**, _Lilian Weng_, 2021. \[[blog](https://lilianweng.github.io/posts/2021-09-25-train-large/)\]
- **Training great LLMs entirely from ground zero in the wilderness as a startup**, _Yi Tay_, 2024. \[[blog](https://www.yitay.net/blog/training-great-llms-entirely-from-ground-zero-in-the-wilderness)\]\[[What happened to BERT & T5? On Transformer Encoders, PrefixLM and Denoising Objectives](https://www.yitay.net/blog/model-architecture-blogpost-encoders-prefixlm-denoising)\]
- \[[Awesome-LLM-System-Papers](https://github.com/AmadeusChan/Awesome-LLM-System-Papers)\]\[[awesome-production-llm](https://github.com/jihoo-kim/awesome-production-llm)\]

- **Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism**, _Shoeybi et al._, arxiv 2019. \[[paper](https://arxiv.org/abs/1909.08053)\]\[[code](https://github.com/NVIDIA/Megatron-LM)\]
- **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models**, _Rajbhandari et al._, arxiv 2019. \[[paper](https://arxiv.org/abs/1910.02054)\]\[[DeepSpeed](https://github.com/microsoft/DeepSpeed)\]
- **Colossal-AI: A Unified Deep Learning System For Large-Scale Parallel Training**, _Li et al._, ICPP 2023. \[[paper](https://arxiv.org/abs/2110.14883)\]\[[code](https://github.com/hpcaitech/ColossalAI)\]
- **MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs**, _Jiang et al._, NSDI 2024. \[[paper](https://arxiv.org/abs/2402.15627)\]\[[veScale](https://github.com/volcengine/veScale)\]\[[blog](https://www.semianalysis.com/p/100000-h100-clusters-power-network)\]\[[Parameter Server OSDI 2014](https://www.usenix.org/system/files/conference/osdi14/osdi14-paper-li_mu.pdf)\]\[[ps-lite](https://github.com/dmlc/ps-lite)\]\[[ByteCheckpoint](https://arxiv.org/abs/2407.20143)\]
- **A Theory on Adam Instability in Large-Scale Machine Learning**, _Molybog et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2304.09871)\]
- **Loss Spike in Training Neural Networks**, _Zhang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.12133)\]
- **Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling**, _Biderman et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2304.01373)\]\[[code](https://github.com/EleutherAI/pythia)\]
- **Continual Pre-Training of Large Language Models: How to (re)warm your model**, _Gupta et al._, \[[paper](https://arxiv.org/abs/2308.04014)\]
- **FLM-101B: An Open LLM and How to Train It with $100K Budget**, _Li et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.03852)\]\[[model](https://huggingface.co/CofeAI/FLM-101B)\]\[[Tele-FLM](https://huggingface.co/CofeAI/Tele-FLM)\]
- **Instruction Tuning with GPT-4**, _Peng et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2304.03277)\]\[[code](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)\]
- **DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines**, _Khattab et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.03714)\]\[[code](https://github.com/stanfordnlp/dspy)\]\[[textgrad](https://github.com/zou-group/textgrad)\]\[[appl](https://github.com/appl-team/appl)\]
- **Alphazero-like Tree-Search can Guide Large Language Model Decoding and Training**, _Feng et al._, ICML 2024. \[[paper](https://arxiv.org/abs/2309.17179)\]\[[code](https://github.com/waterhorse1/LLM_Tree_Search)\]
- **OpenFedLLM: Training Large Language Models on Decentralized Private Data via Federated Learning**, _Ye et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.06954)\]\[[code](https://github.com/rui-ye/OpenFedLLM)\]
- **Arcee's MergeKit: A Toolkit for Merging Large Language Models**, _Goddard et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.13257)\]\[[code](https://github.com/arcee-ai/MergeKit)\]\[[DistillKit](https://github.com/arcee-ai/DistillKit)\]\[[A Survey on Collaborative Strategies in the Era of Large Language Models](https://arxiv.org/abs/2407.06089)\]
- **A Survey on Self-Evolution of Large Language Models**, _Tao et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2404.14387)\]\[[code](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/Awesome-Self-Evolution-of-LLM)\]
- **Adam-mini: Use Fewer Learning Rates To Gain More**, _Zhang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2406.16793)\]\[[code](https://github.com/zyushun/Adam-mini)\]
- **RouteLLM: Learning to Route LLMs with Preference Data**, _Ong et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2406.18665)\]\[[code](https://github.com/lm-sys/routellm)\]
- **OpenDiLoCo: An Open-Source Framework for Globally Distributed Low-Communication Training**, _Jaghouar et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2407.07852)\]\[[code](https://github.com/PrimeIntellect-ai/OpenDiLoCo)\]\[[DiLoCo](https://arxiv.org/abs/2311.08105)\]
- **JailbreakZoo: Survey, Landscapes, and Horizons in Jailbreaking Large Language and Vision-Language Models**, _Jin et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2407.01599)\]\[[code](https://github.com/Allen-piexl/JailbreakZoo)\]\[[jailbreak_llms](https://github.com/verazuo/jailbreak_llms)\]

- \[[wandb](https://github.com/wandb/wandb)\]\[[aim](https://github.com/aimhubio/aim)\]\[[tensorboardX](https://github.com/lanpa/tensorboardX)\]

##### 3.3.1 Alignment

- **AI Alignment: A Comprehensive Survey**, _Ji et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.19852)\]\[[PKU-Alignment](https://github.com/PKU-Alignment)\]
- **Large Language Model Alignment: A Survey**, _Shen et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.15025)\]
- **Aligning Large Language Models with Human: A Survey**, _Wang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2307.12966)\]\[[code](https://github.com/GaryYufei/AlignLLMHumanSurvey)\]
- **A Comprehensive Survey of LLM Alignment Techniques: RLHF, RLAIF, PPO, DPO and More**, _Wang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2407.16216)\]
- \[[alignment-handbook](https://github.com/huggingface/alignment-handbook)\]

- **Self-Instruct: Aligning Language Models with Self-Generated Instructions**, _Wang et al._, ACL 2023. \[[paper](https://arxiv.org/abs/2212.10560)\]\[[code](https://github.com/yizhongw/self-instruct)\]\[[open-instruct](https://github.com/allenai/open-instruct)\]\[[Multi-modal-Self-instruct](https://github.com/zwq2018/Multi-modal-Self-instruct)\]\[[evol-instruct](https://github.com/nlpxucan/evol-instruct)\]\[[Automatic Instruction Evolving for Large Language Models](https://arxiv.org/abs/2406.00770)\]
- **Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing**, _Xu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2406.08464)\]\[[code](https://github.com/magpie-align/magpie)\]
- RLHF: \[[hf blog](https://huggingface.co/blog/rlhf)\]\[[OpenAI blog](https://openai.com/research/learning-from-human-preferences)\]\[[alignment blog](https://openai.com/blog/our-approach-to-alignment-research)\]\[[awesome-RLHF](https://github.com/opendilab/awesome-RLHF)\]
- **Secrets of RLHF in Large Language Models** \[[MOSS-RLHF](https://github.com/OpenLMLab/MOSS-RLHF)\]\[[Part I](https://arxiv.org/abs/2307.04964)\]\[[Part II](https://arxiv.org/abs/2401.06080)\]
- **Safe RLHF: Safe Reinforcement Learning from Human Feedback**, _Dai et al._, ICLR 2024 Spotlight. \[[paper](https://arxiv.org/abs/2310.12773)\]\[[code](https://github.com/PKU-Alignment/safe-rlhf)\]\[[align-anything](https://github.com/PKU-Alignment/align-anything)\]
- **The N+ Implementation Details of RLHF with PPO: A Case Study on TL;DR Summarization**, _Huang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.17031)\]\[[code](https://github.com/vwxyzjn/summarize_from_feedback_details)\]\[[blog](https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo)\]\[[trl](https://github.com/huggingface/trl)\]
- **RLHF Workflow: From Reward Modeling to Online RLHF**, _Dong et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2405.07863)\]\[[code](https://github.com/RLHFlow/RLHF-Reward-Modeling)\]
- **OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework**, _Hu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2405.11143)\]\[[code](https://github.com/OpenLLMAI/OpenRLHF)\]
- **LIMA: Less Is More for Alignment**, _Zhou et al._, NeurIPS 2023. \[[paper](https://arxiv.org/abs/2305.11206)\]
- DPO: **Direct Preference Optimization: Your Language Model is Secretly a Reward Model**, _Rafailov et al._, NeurIPS 2023 Runner-up Award. \[[paper](https://arxiv.org/abs/2305.18290)\]\[[Unofficial Implementation](https://github.com/eric-mitchell/direct-preference-optimization)\]\[[trl](https://github.com/huggingface/trl)\]\[[dpo_trainer](https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py)\]
- BPO: **Black-Box Prompt Optimization: Aligning Large Language Models without Model Training**, _Cheng et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.04155)\]\[[code](https://github.com/thu-coai/BPO)\]
- **KTO: Model Alignment as Prospect Theoretic Optimization**, _Ethayarajh et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.01306)\]\[[code](https://github.com/ContextualAI/HALOs)\]
- **ORPO: Monolithic Preference Optimization without Reference Model**, _Hong et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.07691)\]\[[code](https://github.com/xfactlab/orpo)\]
- TDPO: **Token-level Direct Preference Optimization**, _Zeng et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2404.11999)\]\[[code](https://github.com/Vance0124/Token-level-Direct-Preference-Optimization)\]\[[Step-DPO](https://github.com/dvlab-research/Step-DPO)\]\[[FineGrainedRLHF](https://github.com/allenai/FineGrainedRLHF)\]
- **SimPO: Simple Preference Optimization with a Reference-Free Reward**, _Meng et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2405.14734)\]\[[code](https://github.com/princeton-nlp/SimPO)\]
- **Constitutional AI: Harmlessness from AI Feedback**, _Bai et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2212.08073)\]\[[code](https://github.com/anthropics/ConstitutionalHarmlessnessPaper)\]
- **RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback**, _Lee et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.00267)\]\[[code]\]\[[awesome-RLAIF](https://github.com/mengdi-li/awesome-RLAIF)\]
- **Direct Language Model Alignment from Online AI Feedback**, _Guo et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.04792)\]
- **ReMax: A Simple, Effective, and Efficient Reinforcement Learning Method for Aligning Large Language Models**, _Li et al._, ICML 2024. \[[paper](https://arxiv.org/abs/2310.10505)\]\[[code](https://github.com/liziniu/ReMax)\]\[[policy_optimization](https://github.com/liziniu/policy_optimization)\]
- **Zephyr: Direct Distillation of LM Alignment**, _Tunstall et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.16944)\]\[[code](https://github.com/huggingface/alignment-handbook)\]

- **Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision**, _Burns et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.09390)\]\[[code](https://github.com/openai/weak-to-strong)\]\[[weak-to-strong-deception](https://github.com/keven980716/weak-to-strong-deception)\]
- SPIN: **Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models**, _Chen et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.01335)\]\[[code](https://github.com/uclaml/SPIN)\]\[[unofficial implementation](https://github.com/thomasgauthier/LLM-self-play)\]
- SPPO: **Self-Play Preference Optimization for Language Model Alignment**, _Wu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2405.00675)\]\[[code](https://github.com/uclaml/SPPO)\]
- CALM: **LLM Augmented LLMs: Expanding Capabilities through Composition**, _Bansal et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.02412)\]\[[CALM-pytorch](https://github.com/lucidrains/CALM-pytorch)\]
- **Self-Rewarding Language Models**, _Yuan et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.10020)\]\[[unofficial implementation](https://github.com/lucidrains/self-rewarding-lm-pytorch)\]\[[Meta-Rewarding Language Models](https://arxiv.org/abs/2407.19594)\]
- Anthropic: **Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training**, _Hubinger et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.05566)\]
- **LongAlign: A Recipe for Long Context Alignment of Large Language Models**, _Bai et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.18058)\]\[[code](https://github.com/THUDM/LongAlign)\]
- **Aligner: Achieving Efficient Alignment through Weak-to-Strong Correction**, _Ji et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.02416)\]\[[code](https://github.com/Aligner2024/aligner)\]
- **A Survey on Knowledge Distillation of Large Language Models**, _Xu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.13116)\]\[[code](https://github.com/Tebmer/Awesome-Knowledge-Distillation-of-LLMs)\]
- **NeMo-Aligner: Scalable Toolkit for Efficient Model Alignment**, _Shen et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2405.01481)\]\[[code](https://github.com/NVIDIA/NeMo-Aligner)\]\[[Nemotron-4 340B Technical Report](https://d1qx31qr3h6wln.cloudfront.net/publications/Nemotron_4_340B_8T.pdf)\]\[[Mistral NeMo](https://mistral.ai/news/mistral-nemo/)\]
- **Xwin-LM: Strong and Scalable Alignment Practice for LLMs** _Ni et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2405.20335)\]\[[code](https://github.com/Xwin-LM/Xwin-LM)\]
- **Towards Scalable Automated Alignment of LLMs: A Survey**, _Cao et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2406.01252)\]\[[code](https://github.com/cascip/awesome-auto-alignment)\]
- **Putting RL back in RLHF**, _Huang and Ahmadian_, 2024. \[[blog](https://huggingface.co/blog/putting_rl_back_in_rlhf_with_rloo)\]
- **Prover-Verifier Games improve legibility of language model outputs**, _Kirchner et al._, 2024. \[[blog](https://openai.com/index/prover-verifier-games-improve-legibility/)\]\[[paper](https://cdn.openai.com/prover-verifier-games-improve-legibility-of-llm-outputs/legibility.pdf)\]
- **Rule Based Rewards for Language Model Safety**, _Mu et al._, OpenAI 2024. \[[blog](https://openai.com/index/improving-model-safety-behavior-with-rule-based-rewards/)\]\[[paper](https://cdn.openai.com/rule-based-rewards-for-language-model-safety.pdf)\]\[[code](https://github.com/openai/safety-rbr-code-and-data)\]
- **SELF-GUIDE: Better Task-Specific Instruction Following via Self-Synthetic Finetuning**, _Zhao et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2407.12874)\]\[[code](https://github.com/zhaochenyang20/Prompt2Model-Self-Guide)\]\[[prompt2model](https://github.com/neulab/prompt2model)\]

##### 3.3.2 Context Length

- ALiBi: **Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation**, _Press et al._, ICLR 2022. \[[paper](https://arxiv.org/abs/2108.12409)\]\[[code](https://github.com/ofirpress/attention_with_linear_biases)\]
- Positional Interpolation: **Extending Context Window of Large Language Models via Positional Interpolation**, _Chen et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2306.15595)\]
- **Scaling Transformer to 1M tokens and beyond with RMT**, _Bulatov et al._, AAAI 2024. \[[paper](https://arxiv.org/abs/2304.11062)\]\[[code](https://github.com/booydar/recurrent-memory-transformer/tree/aaai24)\]\[[LM-RMT](https://github.com/booydar/LM-RMT)\]
- **RecurrentGPT: Interactive Generation of (Arbitrarily) Long Text**, _Zhou et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.13304)\]\[[code](https://github.com/aiwaves-cn/RecurrentGPT)\]
- **LongNet: Scaling Transformers to 1,000,000,000 Tokens**, _Ding et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2307.02486)\]\[[code](https://github.com/microsoft/torchscale/blob/main/torchscale/model/LongNet.py)\]\[[unofficial code](https://github.com/kyegomez/LongNet)\]
- **Focused Transformer: Contrastive Training for Context Scaling**, _Tworkowski et al._, NeurIPS 2023. \[[paper](https://arxiv.org/abs/2307.03170)\]\[[code](https://github.com/CStanKonrad/long_llama)\]
- **LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models**, _Chen et al._, ICLR 2024 Oral. \[[paper](https://arxiv.org/abs/2309.12307)\]\[[code](https://github.com/dvlab-research/LongLoRA)\]
- StreamingLLM: **Efficient Streaming Language Models with Attention Sinks**, _Xiao et al._, ICLR 2024. \[[paper](https://arxiv.org/abs/2309.17453)\]\[[code](https://github.com/mit-han-lab/streaming-llm)\]\[[SwiftInfer](https://github.com/hpcaitech/SwiftInfer)\]\[[SwiftInfer blog](https://hpc-ai.com/blog/colossal-ai-swiftinfer)\]
- **YaRN: Efficient Context Window Extension of Large Language Models**, _Peng et al._, ICLR 2024. \[[paper](https://arxiv.org/abs/2309.00071)\]\[[code](https://github.com/jquesnelle/yarn)\]
- **Ring Attention with Blockwise Transformers for Near-Infinite Context**, _Liu et al._, ICLR 2024. \[[paper](https://arxiv.org/abs/2310.01889)\]\[[code](https://github.com/lhao499/ringattention)\]\[[ring-attention-pytorch](https://github.com/lucidrains/ring-attention-pytorch)\]\[[local-attention](https://github.com/lucidrains/local-attention)\]\[[tree_attention](https://github.com/Zyphra/tree_attention)\]
- **LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression**, _Jiang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.06839)\]\[[code](https://github.com/microsoft/LLMLingua)\]
- **LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens**, _Ding et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.13753)\]\[[code](https://github.com/jshuadvd/LongRoPE)\]
- **LLM Maybe LongLM: Self-Extend LLM Context Window Without Tuning**, _Jin et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.01325)\]\[[code](https://github.com/datamllab/LongLM)\]
- **The What, Why, and How of Context Length Extension Techniques in Large Language Models -- A Detailed Survey**, _Pawar et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.07872)\]\[[Awesome-LLM-Long-Context-Modeling](https://github.com/Xnhyacinth/Awesome-LLM-Long-Context-Modeling)\]
- **Data Engineering for Scaling Language Models to 128K Context**, _Fu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.10171)\]\[[code](https://github.com/FranxYao/Long-Context-Data-Engineering)\]
- CEPE: **Long-Context Language Modeling with Parallel Context Encoding**, _Yen et al._, ACL 2024. \[[paper](https://arxiv.org/abs/2402.16617)\]\[[code](https://github.com/princeton-nlp/CEPE)\]
- **InfLLM: Training-Free Long-Context Extrapolation for LLMs with an Efficient Context Memory**, _Xiao et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.04617)\]\[[code](https://github.com/thunlp/InfLLM)\]
- **Counting-Stars: A Simple, Efficient, and Reasonable Strategy for Evaluating Long-Context Large Language Models**, _Song et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.11802)\]\[[code](https://github.com/nick7nlp/Counting-Stars)\]\[[LLMTest_NeedleInAHaystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)\]\[[LooGLE](https://github.com/bigai-nlco/LooGLE)\]\[[LongBench](https://github.com/THUDM/LongBench)\]\[[google-deepmind/loft](https://github.com/google-deepmind/loft)\]
- Infini-Transformer: **Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention**, _Munkhdalai et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2404.07143)\]\[[infini-transformer-pytorch](https://github.com/lucidrains/infini-transformer-pytorch)\]\[[InfiniTransformer](https://github.com/Beomi/InfiniTransformer)\]\[[infini-mini-transformer](https://github.com/jiahe7ay/infini-mini-transformer)\]\[[megalodon](https://github.com/XuezheMax/megalodon)\]
- **Extending Llama-3's Context Ten-Fold Overnight**, _Zhang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2404.19553)\]\[[code](https://github.com/FlagOpen/FlagEmbedding/tree/master/Long_LLM/longllm_qlora)\]\[[activation_beacon](https://github.com/FlagOpen/FlagEmbedding/tree/master/Long_LLM/activation_beacon)\]
- **Make Your LLM Fully Utilize the Context**, _An et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2404.16811)\]\[[code](https://github.com/microsoft/FILM)\]
- CoPE: **Contextual Position Encoding: Learning to Count What's Important**, _Golovneva et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2405.18719)\]\[[rope_cope](https://github.com/chunhuizhang/personal_chatgpt/blob/main/tutorials/position_encoding/rope_cope.ipynb)\]
- **Scaling Granite Code Models to 128K Context**, _Stallone et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2407.13739)\]\[[code](https://github.com/ibm-granite/granite-code-models)\]

##### 3.3.3 Corpus

- \[[datatrove](https://github.com/huggingface/datatrove)\]\[[datasets](https://github.com/huggingface/datasets)\]\[[doccano](https://github.com/doccano/doccano)\]\[[label-studio](https://github.com/HumanSignal/label-studio)\]\[[autolabel](https://github.com/refuel-ai/autolabel)\]
- C4: **Documenting Large Webtext Corpora: A Case Study on the Colossal Clean Crawled Corpus**, _Dodge et al._, arxiv 2021. \[[paper](https://arxiv.org/abs/2104.08758)\]\[[dataset](https://huggingface.co/datasets/allenai/c4)\]
- **The BigScience ROOTS Corpus: A 1.6TB Composite Multilingual Dataset**, _Laurençon et al._, NeurIPS 2023. \[[paper](https://arxiv.org/abs/2303.03915)\]\[[code](https://github.com/bigscience-workshop/data-preparation)\]\[[dataset](https://huggingface.co/bigscience-data)\]
- **The RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora with Web Data, and Web Data Only**, _Penedo et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2306.01116)\]\[[dataset](https://huggingface.co/datasets/tiiuae/falcon-refinedweb)\]
- **Data-Juicer: A One-Stop Data Processing System for Large Language Models**, _Chen et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.02033)\]\[[code](https://github.com/modelscope/data-juicer)\]
- UltraChat: **Enhancing Chat Language Models by Scaling High-quality Instructional Conversations**, _Ding et al._, EMNLP 2023. \[[paper](https://arxiv.org/abs/2305.14233)\]\[[code](https://github.com/thunlp/UltraChat)\]\[[ultrachat](https://huggingface.co/datasets/stingning/ultrachat)\]
- **UltraFeedback: Boosting Language Models with High-quality Feedback**, _Cui et al._, ICML 2024. \[[paper](https://arxiv.org/abs/2310.01377)\]\[[code](https://github.com/OpenBMB/UltraFeedback)\]\[[UltraInteract_sft](https://huggingface.co/datasets/openbmb/UltraInteract_sft)\]
- **What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning**, _Liu et al._, ICLR 2024. \[[paper](https://arxiv.org/abs/2312.15685)\]\[[code](https://github.com/hkust-nlp/deita)\]
- **WanJuan-CC: A Safe and High-Quality Open-sourced English Webtext Dataset**, _Qiu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.19282)\]\[[dataset](https://opendatalab.com/OpenDataLab/WanJuanCC)\]\[[LabelLLM](https://github.com/opendatalab/LabelLLM)\]\[[labelU](https://github.com/opendatalab/labelU)\]\[[MinerU](https://github.com/opendatalab/MinerU)\]\[[PDF-Extract-Kit](https://github.com/opendatalab/PDF-Extract-Kit)\]
- **Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research**, _Soldaini et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.00159)\]\[[code](https://github.com/allenai/dolma)\]\[[OLMo](https://github.com/allenai/OLMo)\]
- **Datasets for Large Language Models: A Comprehensive Survey**, _Liu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.18041)\]\[[Awesome-LLMs-Datasets](https://github.com/lmmlzn/Awesome-LLMs-Datasets)\]
- **DataDreamer: A Tool for Synthetic Data Generation and Reproducible LLM Workflows**, _Patel et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.10379)\]\[[code](https://github.com/datadreamer-dev/datadreamer)\]
- **Large Language Models for Data Annotation: A Survey**, _Tan et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.13446)\]\[[code](https://github.com/Zhen-Tan-dmml/LLM4Annotation)\]
- **Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance**, _Ye et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.16952)\]\[[code](https://github.com/yegcjs/mixinglaws)\]
- **COIG-CQIA: Quality is All You Need for Chinese Instruction Fine-tuning**, _Bai et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.18058)\]\[[dataset](https://huggingface.co/datasets/m-a-p/COIG-CQIA)\]
- **Best Practices and Lessons Learned on Synthetic Data for Language Models**, _Liu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2404.07503)\]
- **The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale**, _HuggingFace_, 2024. \[[paper](https://arxiv.org/abs/2406.17557)\]\[[blogpost](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)\]\[[fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)\]\[[fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)\]
- **DataComp-LM: In search of the next generation of training sets for language models**, _Li et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2406.11794)\]\[[code](https://github.com/mlfoundations/dclm)\]\[[apple/DCLM-7B-8k](https://huggingface.co/apple/DCLM-7B-8k)\]
- **Scaling Synthetic Data Creation with 1,000,000,000 Personas**, _Chan et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2406.20094)\]\[[code](https://github.com/tencent-ailab/persona-hub)\]

- \[[RedPajama-Data](https://github.com/togethercomputer/RedPajama-Data)\]\[[xland-minigrid-datasets](https://github.com/dunno-lab/xland-minigrid-datasets)\]\[[OmniCorpus](https://github.com/OpenGVLab/OmniCorpus)\]\[[dclm](https://github.com/mlfoundations/dclm)\]\[[Infinity-Instruct](https://github.com/FlagOpen/Infinity-Instruct)\]\[[MNBVC](https://github.com/esbatmop/MNBVC)\]
- \[[llm-datasets](https://github.com/mlabonne/llm-datasets)\]

##### 3.3.4 Evaluation

- \[[Awesome-LLM-Eval](https://github.com/onejune2018/Awesome-LLM-Eval)\]\[[LLM-eval-survey](https://github.com/MLGroupJLU/LLM-eval-survey)\]\[[llm_benchmarks](https://github.com/leobeeson/llm_benchmarks)\]
- MMLU: **Measuring Massive Multitask Language Understanding**, _Hendrycks et al._, ICLR 2021.  \[[paper](https://arxiv.org/abs/2009.03300)\]\[[code](https://github.com/hendrycks/test)\]
- HELM: **Holistic Evaluation of Language Models**, _Liang et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2211.09110)\]\[[code](https://github.com/stanford-crfm/helm)\]
- **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena**, _Zheng et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2306.05685)\]\[[code](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge)\]
- **SuperCLUE: A Comprehensive Chinese Large Language Model Benchmark**, _Xu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2307.15020)\]\[[code](https://github.com/CLUEbenchmark/SuperCLUE)\]\[[SuperCLUE-RAG](https://github.com/CLUEbenchmark/SuperCLUE-RAG)\]
- **C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for Foundation Models**, _Huang et al._, NeurIPS 2023. \[[paper](https://arxiv.org/abs/2305.08322)\]\[[code](https://github.com/hkust-nlp/ceval)\]\[[chinese-llm-benchmark](https://github.com/jeinlee1991/chinese-llm-benchmark)\]
- **CMMLU: Measuring massive multitask language understanding in Chinese**, _Li et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2306.09212)\]\[[code](https://github.com/haonan-li/CMMLU)\]
- **CMMMU: A Chinese Massive Multi-discipline Multimodal Understanding Benchmark**, _Zhang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.11944)\]\[[code](https://github.com/CMMMU-Benchmark/CMMMU)\]
- **Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference**, _Chiang et al._, ICML 2024. \[[paper](https://arxiv.org/abs/2403.04132)\]\[[demo](https://chat.lmsys.org/)\]
- **Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models**, _Kim et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2405.01535)\]\[[code](https://github.com/prometheus-eval/prometheus-eval)\]
- **LMMs-Eval: Reality Check on the Evaluation of Large Multimodal Models**, _Zhang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2407.12772)\]\[[code](https://github.com/EvolvingLMMs-Lab/lmms-eval)\]

- \[[Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)\]
- \[[AlpacaEval Leaderboard](https://tatsu-lab.github.io/alpaca_eval/)\]\[[alpaca_eval](https://github.com/tatsu-lab/alpaca_eval)\]
- \[[Chatbot-Arena-Leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)\]\[[blog](https://lmsys.org/blog/2023-05-03-arena/)\]\[[FastChat](https://github.com/lm-sys/FastChat)\]\[[arena-hard](https://github.com/lm-sys/arena-hard)\]
- \[[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)\]\[[OpenAI Evals](https://github.com/openai/evals)\]\[[simple-evals](https://github.com/openai/simple-evals)\]
- \[[OpenCompass](https://github.com/open-compass/opencompass)\]\[[GAOKAO-Eval](https://github.com/open-compass/GAOKAO-Eval)\]\[[VLMEvalKit](https://github.com/open-compass/VLMEvalKit)\]
- \[[llm-colosseum](https://github.com/OpenGenerativeAI/llm-colosseum)\]

##### 3.3.5 Hallucination

- **Extrinsic Hallucinations in LLMs**, _Lilian Weng_, 2024. \[[blog](https://lilianweng.github.io/posts/2024-07-07-hallucination/)\]
- **Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models**, _Zhang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.01219)\]\[[code](https://github.com/HillZhang1999/llm-hallucination-survey)\]
- **A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions**, _Huang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.05232)\]\[[code](https://github.com/LuckyyySTA/Awesome-LLM-hallucination)\]\[[Awesome-MLLM-Hallucination](https://github.com/showlab/Awesome-MLLM-Hallucination)\]
- **The Dawn After the Dark: An Empirical Study on Factuality Hallucination in Large Language Models**, _Li et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.03205)\]\[[code](https://github.com/RUCAIBox/HaluEval-2.0)\]
- **FacTool: Factuality Detection in Generative AI -- A Tool Augmented Framework for Multi-Task and Multi-Domain Scenarios**, _Chem et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2307.13528)\]\[[code](https://github.com/GAIR-NLP/factool)\]\[[OlympicArena](https://github.com/GAIR-NLP/OlympicArena)\]
- **Chain-of-Verification Reduces Hallucination in Large Language Models**, _Dhuliawala et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.11495)\]\[[code](https://github.com/lastmile-ai/aiconfig/tree/main/cookbooks/Chain-of-Verification)\]
- **HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models**, _Guan et al._, CVPR 2024. \[[paper](https://arxiv.org/abs/2310.14566)\]\[[code](https://github.com/tianyi-lab/HallusionBench)\]
- **Woodpecker: Hallucination Correction for Multimodal Large Language Models**, _Yin et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.16045)\]\[[code](https://github.com/BradyFU/Woodpecker)\]
- **OPERA: Alleviating Hallucination in Multi-Modal Large Language Models via Over-Trust Penalty and Retrospection-Allocation**, _Huang et al._, CVPR 2024 Highlight. \[[paper](https://arxiv.org/abs/2311.17911)\]\[[code](https://github.com/shikiw/OPERA)\]
- **TrustLLM: Trustworthiness in Large Language Models**, _Sun et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.05561)\]\[[code](https://github.com/HowieHwong/TrustLLM)\]
- SAFE: **Long-form factuality in large language models**, _Wei et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.18802)\]\[[code](https://github.com/google-deepmind/long-form-factuality)\]
- **Detecting hallucinations in large language models using semantic entropy**, _Farquhar et al._, Nature 2024. \[[paper](https://www.nature.com/articles/s41586-024-07421-0)\]\[[semantic_uncertainty](https://github.com/jlko/semantic_uncertainty)\]\[[long_hallucinations](https://github.com/jlko/long_hallucinations)\]\[[Lynx-hallucination-detection](https://github.com/patronus-ai/Lynx-hallucination-detection)\]

##### 3.3.6 Inference

- **How to make LLMs go fast**, 2023. \[[blog](https://vgel.me/posts/faster-inference/)\]
- **A Visual Guide to Quantization**, 2024. \[[blog](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization)\]
- **Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems**, _Miao et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.15234)\]\[[Awesome-Quantization-Papers](https://github.com/Zhen-Dong/Awesome-Quantization-Papers)\]\[[awesome-model-quantization](https://github.com/htqin/awesome-model-quantization)\]\[[qllm-eval](https://github.com/thu-nics/qllm-eval)\]
- **Full Stack Optimization of Transformer Inference: a Survey**, _Kim et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2302.14017)\]
- **A Survey on Efficient Inference for Large Language Models**, _Zhou et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2404.14294)\]

- **LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale**, _Dettmers et al._, NeurIPS 2022. \[[paper](https://arxiv.org/abs/2208.07339)\]\[[code](https://github.com/TimDettmers/bitsandbytes)\]
- **LLM-FP4: 4-Bit Floating-Point Quantized Transformers**, _Liu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.16836)\]\[[code](https://github.com/nbasyl/LLM-FP4)\]
- **OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models**, _Shao et al._, ICLR 2024 Spotlight. \[[paper](https://arxiv.org/abs/2308.13137)\]\[[code](https://github.com/OpenGVLab/OmniQuant)\]
- **BitNet: Scaling 1-bit Transformers for Large Language Models**, _Wang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.11453)\]\[[code](https://github.com/microsoft/unilm/tree/master/bitnet)\]\[[unofficial implementation](https://github.com/kyegomez/BitNet)\]\[[T-MAC](https://github.com/microsoft/T-MAC)\]\[[BiLLM](https://github.com/Aaronhuang-778/BiLLM)\]
- **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers**, _Frantar et al._, ICLR 2023. \[[paper](https://arxiv.org/abs/2210.17323)\]\[[code](https://github.com/IST-DASLab/gptq)\]\[[AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)\]
- **QMoE: Practical Sub-1-Bit Compression of Trillion-Parameter Models**, _Frantar et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.16795)\]\[[code](https://github.com/IST-DASLab/qmoe)\]
- **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration**, _Lin et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2306.00978)\]\[[code](https://github.com/mit-han-lab/llm-awq)\]\[[AutoAWQ](https://github.com/casper-hansen/AutoAWQ)\]\[[qserve](https://github.com/mit-han-lab/qserve)\]
- **LLM in a flash: Efficient Large Language Model Inference with Limited Memory**, _Alizadeh et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.11514)\]\[[air_llm](https://github.com/lyogavin/Anima/tree/main/air_llm)\]
- **LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models**, _Jiang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.05736)\]\[[code](https://github.com/microsoft/LLMLingua)\]
- **FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU**, _Sheng et al._, ICML 2023. \[[paper](https://arxiv.org/abs/2303.06865)\]\[[code](https://github.com/FMInference/FlexGen)\]
- **PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU**, _Song et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.12456)\]\[[code](https://github.com/SJTU-IPADS/PowerInfer)\]\[[llama.cpp](https://github.com/ggerganov/llama.cpp)\]\[[Anima](https://github.com/lyogavin/Anima)\]\[[PowerInfer-2](https://arxiv.org/abs/2406.06282)\]
- **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**, _Dao et al._, NeurIPS 2022. \[[paper](https://arxiv.org/abs/2205.14135)\]\[[code](https://github.com/Dao-AILab/flash-attention)\]
- **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning**, _Tri Dao_, ICLR 2024. \[[paper](https://arxiv.org/abs/2307.08691)\]\[[code](https://github.com/Dao-AILab/flash-attention)\]
- **FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision**, _Shah et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2407.08608)\]\[[code](https://github.com/Dao-AILab/flash-attention)\]
- vllm: **Efficient Memory Management for Large Language Model Serving with PagedAttention**, _Kwon et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.06180)\]\[[code](https://github.com/vllm-project/vllm)\]\[[FastChat](https://github.com/lm-sys/FastChat)\]
- SGLang: **Fast and Expressive LLM Inference with RadixAttention and SGLang**, _Zheng et al._, Stanford blog 2024. \[[blog](https://lmsys.org/blog/2024-01-17-sglang/)\]\[[paper](https://arxiv.org/abs/2312.07104)\]\[[code](https://github.com/sgl-project/sglang/)\]
- **Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads**, _Cai et al._, ICML 2024. \[[paper](https://arxiv.org/abs/2401.10774)\]\[[code](https://github.com/FasterDecoding/Medusa)\]
- **EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty**, _Li et al._, ICML 2024. \[[paper](https://arxiv.org/abs/2401.15077)\]\[[code](https://github.com/SafeAILab/EAGLE)\]\[[LLMSpeculativeSampling](https://github.com/feifeibear/LLMSpeculativeSampling)\]\[[Spec-Bench](https://github.com/hemingkx/Spec-Bench)\]
- **APAR: LLMs Can Do Auto-Parallel Auto-Regressive Decoding**, _Liu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.06761)\]\[[code]\]\[[Ouroboros](https://github.com/thunlp/Ouroboros)\]
- **CLLMs: Consistency Large Language Models**, _Kou et al._, ICML 2024. \[[paper](https://arxiv.org/abs/2403.00835)\]\[[code](https://github.com/hao-ai-lab/Consistency_LLM)\]\[[LookaheadDecoding](https://github.com/hao-ai-lab/LookaheadDecoding)\]
- **MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention**, _Jiang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2407.02490)\]\[[code](https://github.com/microsoft/MInference)\]
- Sarathi-Serve: **Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve**, _Agrawal et al._, OSDI 2024. \[[paper](https://arxiv.org/abs/2403.02310)\]\[[code](https://github.com/microsoft/sarathi-serve)\]\[[Orca OSDI 2022](https://www.usenix.org/system/files/osdi22-yu.pdf)\]
- **DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving**, _Zhong et al._, OSDI 2024. \[[paper](https://arxiv.org/abs/2401.09670)\]\[[code](https://github.com/LLMServe/DistServe)\]
- **Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving**, _Qin et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2407.00079)\]\[[code](https://github.com/kvcache-ai/Mooncake)\]\[[ktransformers](https://github.com/kvcache-ai/ktransformers)\]

- \[[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)\]\[[FasterTransformer](https://github.com/NVIDIA/FasterTransformer)\]\[[TritonServer](https://github.com/triton-inference-server/server)\]\[[GenerativeAIExamples](https://github.com/NVIDIA/GenerativeAIExamples)\]\[[TensorRT-Model-Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer)\]\[[TensorRT](https://github.com/NVIDIA/TensorRT)\]
- \[[DeepSpeed-MII](https://github.com/microsoft/DeepSpeed-MII)\]\[[DeepSpeed-FastGen](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen)\]\[[ONNX Runtime](https://github.com/microsoft/onnxruntime)\]\[[onnx](https://github.com/onnx/onnx)\]
- \[[text-generation-inference](https://github.com/huggingface/text-generation-inference)\]\[[quantization](https://huggingface.co/docs/transformers/main/en/quantization)\]\[[optimum-quanto](https://github.com/huggingface/optimum-quanto)\]
- \[[OpenLLM](https://github.com/bentoml/OpenLLM)\]\[[mlc-llm](https://github.com/mlc-ai/mlc-llm)\]\[[ollama](https://github.com/jmorganca/ollama)\]\[[open-webui](https://github.com/open-webui/open-webui)\]\[[torchchat](https://github.com/pytorch/torchchat)\]
- \[[LMDeploy](https://github.com/InternLM/lmdeploy)\]\[[Mooncake](https://github.com/kvcache-ai/Mooncake)\]
- \[[ggml](https://github.com/ggerganov/ggml)\]\[[exllamav2](https://github.com/turboderp/exllamav2)\]\[[llama.cpp](https://github.com/ggerganov/llama.cpp)\]\[[gpt-fast](https://github.com/pytorch-labs/gpt-fast)\]\[[lightllm](https://github.com/ModelTC/lightllm)\]\[[fastllm](https://github.com/ztxz16/fastllm)\]\[[CTranslate2](https://github.com/OpenNMT/CTranslate2)\]\[[ipex-llm](https://github.com/intel-analytics/ipex-llm)\]\[[rtp-llm](https://github.com/alibaba/rtp-llm)\]\[[KsanaLLM](https://github.com/pcg-mlp/KsanaLLM)\]
- \[[ChuanhuChatGPT](https://github.com/GaiZhenbiao/ChuanhuChatGPT)\]\[[ChatGPT-Next-Web](https://github.com/ChatGPTNextWeb/ChatGPT-Next-Web)\]\[[OpenLLM](https://github.com/bentoml/OpenLLM)\]

##### 3.3.7 MoE

- **Mixture of Experts Explained**, _Sanseviero et al._, Hugging Face Blog 2023. \[[blog](https://huggingface.co/blog/moe)\]
- **Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer**, _Shazeer et al._, arxiv 2017. \[[paper](https://arxiv.org/abs/1701.06538)\]\[[Re-Implementation](https://github.com/davidmrau/mixture-of-experts)\]
- **GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding**, _Lepikhin et al._, arxiv 2020. \[[paper](https://arxiv.org/abs/2006.16668)\]\[[mixture-of-experts](https://github.com/lucidrains/mixture-of-experts)\]
- **MegaBlocks: Efficient Sparse Training with Mixture-of-Experts**, _Gale et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2211.15841)\]\[[code](https://github.com/stanford-futuredata/megablocks)\]
- **Mixture-of-Experts Meets Instruction Tuning:A Winning Combination for Large Language Models**, _Shen et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.14705)\]\[[code]\]
- **Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity**, _Fedus et al._, arxiv 2021. \[[paper](https://arxiv.org/abs/2101.03961)\]\[[code](https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/moe.py)\]
- **Fast Inference of Mixture-of-Experts Language Models with Offloading**, _Eliseev and Mazur_, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.17238)\]\[[code](https://github.com/dvmazur/mixtral-offloading)\]
- Mixtral-8×7B: **Mixtral of Experts**, _Jiang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2401.04088)\]\[[code](https://github.com/mistralai/mistral-inference)\]\[[megablocks-public](https://github.com/mistralai/megablocks-public)\]\[[model](https://huggingface.co/mistralai)\]\[[blog](https://mistral.ai/news/mixtral-of-experts/)\]\[[Chinese-Mixtral-8x7B](https://github.com/HIT-SCIR/Chinese-Mixtral-8x7B)\]\[[Chinese-Mixtral](https://github.com/ymcui/Chinese-Mixtral)\]
- **DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models**, _Dai et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.06066)\]\[[code](https://github.com/deepseek-ai/DeepSeek-MoE)\]
- **DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model**, _DeepSeek-AI_, arxiv 2024. \[[paper](https://arxiv.org/abs/2405.04434)\]\[[code](https://github.com/deepseek-ai/DeepSeek-V2)\]
- **Let the Expert Stick to His Last: Expert-Specialized Fine-Tuning for Sparse Architectural Large Language Models**, _Wang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2407.01906)\]
- **Evolutionary Optimization of Model Merging Recipes**, _Akiba et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.13187)\]\[[code](https://github.com/SakanaAI/evolutionary-model-merge)\]
- **A Closer Look into Mixture-of-Experts in Large Language Models**, _Lo et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2406.18219)\]\[[code](https://github.com/kamanphoebe/Look-into-MoEs)\]
- **A Survey on Mixture of Experts**, _Cai et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2407.06204)\]\[[code](https://github.com/withinmiaov/A-Survey-on-Mixture-of-Experts)\]

- \[[llama-moe](https://github.com/pjlab-sys4nlp/llama-moe)\]\[[Aurora](https://github.com/WangRongsheng/Aurora)\]\[[OpenMoE](https://github.com/XueFuzhao/OpenMoE)\]\[[makeMoE](https://github.com/AviSoori1x/makeMoE)\]\[[PEER-pytorch](https://github.com/lucidrains/PEER-pytorch)\]

##### 3.3.8 PEFT (Parameter-efficient Fine-tuning)

- \[[DeepSpeed](https://github.com/microsoft/DeepSpeed)\]\[[DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples)\]\[[blog](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)\]
- \[[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)\]\[[NeMo](https://github.com/NVIDIA/NeMo)\]\[[Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed)\]\[[Megatron-DeepSpeed](https://github.com/bigscience-workshop/Megatron-DeepSpeed)\]
- \[[torchtune](https://github.com/pytorch/torchtune)\]\[[torchtitan](https://github.com/pytorch/torchtitan)\]
- \[[PEFT](https://github.com/huggingface/peft)\]\[[trl](https://github.com/huggingface/trl)\]\[[accelerate](https://github.com/huggingface/accelerate)\]\[[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)\]\[[LMFlow](https://github.com/OptimalScale/LMFlow)\]\[[xtuner](https://github.com/InternLM/xtuner)\]\[[MFTCoder](https://github.com/codefuse-ai/MFTCoder)\]\[[llm-foundry](https://github.com/mosaicml/llm-foundry)\]\[[swift](https://github.com/modelscope/swift)\]
- \[[mergekit](https://github.com/arcee-ai/mergekit)\]\[[merge-models](https://huggingface.co/blog/mlabonne/merge-models)\]\[[Model Merging](https://huggingface.co/collections/osanseviero/model-merging-65097893623330a3a51ead66)\]\[[OpenChatKit](https://github.com/togethercomputer/OpenChatKit)\]

- **LoRA: Low-Rank Adaptation of Large Language Models**, _Hu et al._, ICLR 2022. \[[paper](https://arxiv.org/abs/2106.09685)\]\[[code](https://github.com/microsoft/LoRA)\]\[[LoRA From Scratch](https://lightning.ai/lightning-ai/studios/code-lora-from-scratch)\]\[[lora](https://github.com/cloneofsimo/lora)\]\[[dora](https://github.com/catid/dora)\]\[[MoRA](https://github.com/kongds/MoRA)\]\[[ziplora-pytorch](https://github.com/mkshing/ziplora-pytorch)\]\[[alpaca-lora](https://github.com/tloen/alpaca-lora)\]
- **QLoRA: Efficient Finetuning of Quantized LLMs**, _Dettmers et al._, NeurIPS 2023 Oral. \[[paper](https://arxiv.org/abs/2305.14314)\]\[[code](https://github.com/artidoro/qlora)\]\[[bitsandbytes](https://github.com/TimDettmers/bitsandbytes)\]\[[unsloth](https://github.com/unslothai/unsloth)\]\[[ir-qlora](https://github.com/htqin/ir-qlora)\]\[[fsdp_qlora](https://github.com/AnswerDotAI/fsdp_qlora)\]
- **S-LoRA: Serving Thousands of Concurrent LoRA Adapters**, _Sheng et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.03285)\]\[[code](https://github.com/S-LoRA/S-LoRA)\]\[[AdaLoRA](https://github.com/QingruZhang/AdaLoRA)\]\[[LoRAMoE](https://github.com/Ablustrund/LoRAMoE)\]\[[lorahub](https://github.com/sail-sg/lorahub)\]\[[O-LoRA](https://github.com/cmnfriend/O-LoRA)\]\[[qa-lora](https://github.com/yuhuixu1993/qa-lora)\]
- **LoRA-GA: Low-Rank Adaptation with Gradient Approximation**, _Wang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2407.05000)\]\[[code](https://github.com/Outsider565/LoRA-GA)\]\[[LoRA-Pro blog](https://kexue.fm/archives/10266)\]\[[dora](https://github.com/catid/dora)\]
- **GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection**, _Zhao et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.03507)\]\[[code](https://github.com/jiaweizzhao/galore)\]\[[Q-GaLore](https://github.com/VITA-Group/Q-GaLore)\]\[[WeLore](https://github.com/VITA-Group/WeLore)\]
- **Prefix-Tuning: Optimizing Continuous Prompts for Generation**, _Li et al._, ACL 2021. \[[paper](https://arxiv.org/abs/2101.00190)\]\[[code](https://github.com/XiangLi1999/PrefixTuning)\]
- Adapter: **Parameter-Efficient Transfer Learning for NLP**, _Houlsby et al._, ICML 2019. \[[paper](https://arxiv.org/abs/1902.00751)\]\[[code](https://github.com/google-research/adapter-bert)\]\[[unify-parameter-efficient-tuning](https://github.com/jxhe/unify-parameter-efficient-tuning)\]
- **Adapters: A Unified Library for Parameter-Efficient and Modular Transfer Learning**, _Poth et al._, EMNLP 2023. \[[paper](https://arxiv.org/abs/2311.11077)\]\[[code](https://github.com/adapter-hub/adapters)\]\[[A Survey on LoRA of Large Language Models](https://arxiv.org/abs/2407.11046)\]
- **LLM-Adapters: An Adapter Family for Parameter-Efficient Fine-Tuning of Large Language Models**, _Hu et al._, EMNLP 2023. \[[paper](https://arxiv.org/abs/2304.01933)\]\[[code](https://github.com/AGI-Edgerunners/LLM-Adapters)\]
- **LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention**, _Zhang et al._, ICLR 2024. \[[paper](https://arxiv.org/abs/2303.16199)\]\[[code](https://github.com/OpenGVLab/LLaMA-Adapter)\]
- **LLaMA Pro: Progressive LLaMA with Block Expansion**, _Wu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.02415)\]\[[code](https://github.com/TencentARC/LLaMA-Pro)\]
- P-Tuning: **GPT Understands, Too**, _Liu et al._, arxiv 2021. \[[paper](https://arxiv.org/abs/2103.10385)\]\[[code](https://github.com/THUDM/P-tuning)\]
- **P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks**, _Liu et al._, ACL 2022. \[[paper](https://arxiv.org/abs/2110.07602)\]\[[code](https://github.com/THUDM/P-tuning-v2)\]\[[pet](https://github.com/timoschick/pet)\]\[[PrefixTuning](https://github.com/XiangLi1999/PrefixTuning)\]
- **Towards a Unified View of Parameter-Efficient Transfer Learning**, _He et al._, ICLR 2022. \[[paper](https://arxiv.org/abs/2110.04366)\]\[[code](https://github.com/jxhe/unify-parameter-efficient-tuning)\]
- **Parameter-efficient fine-tuning of large-scale pre-trained language models**, _Ding et al._, Nature Machine Intelligence 2023. \[[paper](https://www.nature.com/articles/s42256-023-00626-4)\]\[[code](https://github.com/thunlp/OpenDelta)\]
- **Mixed Precision Training**, _Micikevicius et al._, ICLR 2018. \[[paper](https://arxiv.org/abs/1710.03740)\]
- **8-bit Optimizers via Block-wise Quantization** _Dettmers et al._, ICLR 2022. \[[paper](https://arxiv.org/abs/2110.02861)\]\[[code](https://github.com/timdettmers/bitsandbytes)\]
- **FP8-LM: Training FP8 Large Language Models** _Peng et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.18313)\]\[[code](https://github.com/Azure/MS-AMP)\]
- **Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive Survey**, _Han et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.14608)\]
- **LMFlow: An Extensible Toolkit for Finetuning and Inference of Large Foundation Models**, _Diao et al._, NAACL 2024. \[[paper](https://arxiv.org/abs/2306.12420)\]\[[code](https://github.com/OptimalScale/LMFlow)\]
- **LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models**, _Zheng et al._, ACL 2024. \[[paper](https://arxiv.org/abs/2403.13372)\]\[[code](https://github.com/hiyouga/LLaMA-Factory)\]
- **ReFT: Representation Finetuning for Language Models**, _Wu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2404.03592)\]\[[code](https://github.com/stanfordnlp/pyreft)\]

##### 3.3.9 Prompt Learning

- **OpenPrompt: An Open-source Framework for Prompt-learning**, _Ding et al._, arxiv 2021. \[[paper](https://arxiv.org/abs/2111.01998)\]\[[code](https://github.com/thunlp/OpenPrompt)\]
- **Learning to Generate Prompts for Dialogue Generation through Reinforcement Learning**, _Su et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2206.03931)\]
- **Large Language Models Are Human-Level Prompt Engineers**, _Zhou et al._, ICLR 2023. \[[paper](https://arxiv.org/abs/2211.01910)\]\[[code](https://github.com/keirp/automatic_prompt_engineer)\]
- **Large Language Models as Optimizers**, _Yang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.03409)\]\[[code](https://github.com/google-deepmind/opro)\]
- **Principled Instructions Are All You Need for Questioning LLaMA-1/2, GPT-3.5/4**, _Bsharat et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.16171)\]\[[code](https://github.com/VILA-Lab/ATLAS)\]
- **Meta-Prompting: Enhancing Language Models with Task-Agnostic Scaffolding**, _Suzgun and Kalai_, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.12954)\]\[[code](https://github.com/suzgunmirac/meta-prompting)\]
- AutoPrompt: **Intent-based Prompt Calibration: Enhancing prompt optimization with synthetic boundary cases**, _Levi et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.03099)\]\[[code](https://github.com/Eladlev/AutoPrompt)\]\[[automatic_prompt_engineer](https://github.com/keirp/automatic_prompt_engineer)\]\[[appl](https://github.com/appl-team/appl)\]\[[sammo](https://github.com/microsoft/sammo)\]
- **LangGPT: Rethinking Structured Reusable Prompt Design Framework for LLMs from the Programming Language**, _Wang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.16929)\]\[[code](https://github.com/yzfly/langgpt)\]
- **The Prompt Report: A Systematic Survey of Prompting Techniques**, _Schulhoff et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2406.06608)\]\[[code](https://github.com/trigaten/Prompt_Systematic_Review)\]\[[A Survey of Prompt Engineering Methods in Large Language Models for Different NLP Tasks](https://arxiv.org/abs/2407.12994)\]
- \[[PromptPapers](https://github.com/thunlp/PromptPapers)\]\[[OpenAI Docs](https://platform.openai.com/docs/guides/prompt-engineering)\]\[[ChatGPT Prompt Engineering for Developers](https://prompt-engineering.xiniushu.com/)\]\[[Prompt Engineering Guide](https://www.promptingguide.ai/zh)\]\[[k12promptguide](https://www.k12promptguide.com/)\]\[[gpt-prompt-engineer](https://github.com/mshumer/gpt-prompt-engineer)\]\[[awesome-chatgpt-prompts](https://github.com/f/awesome-chatgpt-prompts)\]\[[awesome-chatgpt-prompts-zh](https://github.com/PlexPt/awesome-chatgpt-prompts-zh)\]

- **The Power of Scale for Parameter-Efficient Prompt Tuning**, _Lester et al._, EMNLP 2021. \[[paper](https://arxiv.org/abs/2104.08691)\]\[[code](https://github.com/google-research/prompt-tuning)\]\[[soft-prompt-tuning](https://github.com/kipgparker/soft-prompt-tuning)\]\[[Prompt-Tuning](https://github.com/mkshing/Prompt-Tuning)\]
- **A Survey on In-context Learning**, _Dong et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2301.00234)\]\[[code](https://github.com/dqxiu/icl_paperlist)\]
- **Rethinking the Role of Demonstrations: What Makes In-Context Learning Work**, _Min et al._, EMNLP 2022. \[[paper](https://arxiv.org/abs/2202.12837)\]\[[code](https://github.com/Alrope123/rethinking-demonstrations)\]
- **Larger language models do in-context learning differently**, _Wei et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2303.03846)\]
- **PAL: Program-aided Language Models**, _Gao et al._, ICML 2023. \[[paper](https://arxiv.org/abs/2211.10435)\]\[[code](https://github.com/reasoning-machines/pal)\]
- **A Comprehensive Survey on Instruction Following**, _Lou et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2303.10475)\]\[[code](https://github.com/RenzeLou/awesome-instruction-learning)\]
- RLHF: **Deep reinforcement learning from human preferences**, _Christiano et al._, NIPS 2017. \[[paper](https://arxiv.org/abs/1706.03741)\]
- RLHF: **Fine-Tuning Language Models from Human Preferences**, _Ziegler et al._, arxiv 2019. \[[paper](https://arxiv.org/abs/1909.08593)\]\[[code](https://github.com/openai/lm-human-preferences)\]
- RLHF: **Learning to summarize from human feedback**, _Stiennon et al._, NeurIPS 2020. \[[paper](https://arxiv.org/abs/2009.01325)\]\[[code](https://github.com/openai/summarize-from-feedback)\]
- RLHF: **Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback**, _Bai et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2204.05862)\]\[[code](https://github.com/anthropics/hh-rlhf)\]
- **Finetuned Language Models Are Zero-Shot Learners**, _Wei et al._, ICLR 2022. \[[paper](https://arxiv.org/abs/2109.01652)\]
- **Instruction Tuning for Large Language Models: A Survey**, _Zhang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2308.10792)\]\[[code](https://github.com/xiaoya-li/Instruction-Tuning-Survey)\]
- **What learning algorithm is in-context learning? Investigations with linear models**, _Akyürek et al._, ICLR 2023. \[[paper](https://arxiv.org/abs/2211.15661)\]
- **Why Can GPT Learn In-Context? Language Models Implicitly Perform Gradient Descent as Meta-Optimizers**, _Dai et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2212.10559)\]\[[code](https://github.com/microsoft/LMOps/tree/main/understand_icl)\]

##### 3.3.10 RAG (Retrieval Augmented Generation)

- **Retrieval-Augmented Generation for Large Language Models: A Survey**, _Gao et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.10997)\]\[[code](https://github.com/Tongji-KGLLM/RAG-Survey)\]\[[Modular RAG](https://arxiv.org/abs/2407.21059)\]
- **Retrieval-Augmented Generation for AI-Generated Content: A Survey**, _Zhao et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.19473)\]\[[code](https://github.com/hymie122/RAG-Survey)\]
- **A Survey on Retrieval-Augmented Text Generation for Large Language Models**, _Huang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2404.10981)\]\[[Retrieval-Augmented Generation for Natural Language Processing: A Survey](https://arxiv.org/abs/2407.13193)\]
- **RAG and RAU: A Survey on Retrieval-Augmented Language Model in Natural Language Processing**, _Hu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2404.19543)\]\[[code](https://github.com/2471023025/RALM_Survey)\]

- **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**, _Lewis et al._, NeurIPS 2020. \[[paper](https://arxiv.org/abs/2005.11401)\]\[[code](https://github.com/huggingface/transformers/tree/main/examples/research_projects/rag)\]\[[model](https://huggingface.co/facebook/rag-token-nq)\]\[[docs](https://huggingface.co/docs/transformers/main/model_doc/rag)\]\[[FAISS](https://github.com/facebookresearch/faiss)\]
- **Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection**, _Asai et al._, ICLR 2024 Oral. \[[paper](https://arxiv.org/abs/2310.11511)\]\[[code](https://github.com/AkariAsai/self-rag)\]\[[CRAG](https://github.com/HuskyInSalt/CRAG)\]\[[Golden-Retriever](https://arxiv.org/abs/2408.00798)\]
- **Dense Passage Retrieval for Open-Domain Question Answering**, _Karpukhin et al._, EMNLP 2020. \[[paper](https://arxiv.org/abs/2004.04906)\]\[[code](https://github.com/facebookresearch/DPR)\]
- **Internet-Augmented Dialogue Generation** _Komeili et al._, arxiv 2021. \[[paper](https://arxiv.org/abs/2107.07566)\]
- RETRO: **Improving language models by retrieving from trillions of tokens**, _Borgeaud et al._, arxiv 2021. \[[paper](https://arxiv.org/abs/2112.04426)\]\[[RETRO-pytorch](https://github.com/lucidrains/RETRO-pytorch)\]
- FLARE: **Active Retrieval Augmented Generation**, _Jiang et al._, EMNLP 2023. \[[paper](https://arxiv.org/abs/2305.06983)\]\[[code](https://github.com/jzbjyb/FLARE)\]
- **FreshLLMs: Refreshing Large Language Models with Search Engine Augmentation**, _Vu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.03214)\]\[[code](https://github.com/freshllms/freshqa)\]
- **Chain-of-Note: Enhancing Robustness in Retrieval-Augmented Language Models**, _Yu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.09210)\]
- **Learning to Filter Context for Retrieval-Augmented Generation**, _Wang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.08377)\]\[[code](https://github.com/zorazrw/filco)\]
- **RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval**, _Sarthi et al._, ICLR 2024. \[[paper](https://arxiv.org/abs/2401.18059)\]\[[code](https://github.com/parthsarthi03/raptor)\]\[[tree2retriever](https://github.com/yanqiangmiffy/tree2retriever)\]\[[GoMate](https://github.com/gomate-community/GoMate)\]
- **When Large Language Models Meet Vector Databases: A Survey**, _Jing et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.01763)\]
- **RAFT: Adapting Language Model to Domain Specific RAG**, _Zhang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.10131)\]\[[code](https://github.com/ShishirPatil/gorilla/tree/main/raft)\]
- **RA-ISF: Learning to Answer and Understand from Retrieval Augmentation via Iterative Self-Feedback**, _Liu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.06840)\]\[[code](https://github.com/OceannTwT/ra-isf)\]
- **RQ-RAG: Learning to Refine Queries for Retrieval Augmented Generation**, _Chan et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2404.00610)\]\[[code](https://github.com/chanchimin/RQ-RAG)\]\[[Adaptive-RAG](https://github.com/starsuzi/Adaptive-RAG)\]\[[Advanced RAG 11: Query Classification and Refinement](https://ai.gopubby.com/advanced-rag-11-query-classification-and-refinement-2aec79f4140b)\]
- **Blended RAG: Improving RAG (Retriever-Augmented Generation) Accuracy with Semantic Search and Hybrid Query-Based Retrievers**, _Sawarkar et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2404.07220)\]\[[code](https://github.com/ibm-ecosystem-engineering/Blended-RAG)\]\[[infinity](https://github.com/infiniflow/infinity)\]
- **FlashRAG: A Modular Toolkit for Efficient Retrieval-Augmented Generation Research**, _Jin et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2405.13576)\]\[[code](https://github.com/ruc-nlpir/flashrag)\]
- **HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models**, _Gutiérrez et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2405.14831)\]\[[code](https://github.com/OSU-NLP-Group/HippoRAG)\]
- **From Local to Global: A Graph RAG Approach to Query-Focused Summarization**, _Edge et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2404.16130)\]\[[code](https://github.com/microsoft/graphrag)\]\[[GraphRAG-Local-UI](https://github.com/severian42/GraphRAG-Local-UI)\]\[[graph-rag](https://github.com/sarthakrastogi/graph-rag)\]\[[llm-graph-builder](https://github.com/neo4j-labs/llm-graph-builder)\]\[[Triplex](https://huggingface.co/SciPhi/Triplex)\]\[[knowledge_graph_maker](https://github.com/rahulnyk/knowledge_graph_maker)\]
- **Searching for Best Practices in Retrieval-Augmented Generation**, _Wang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2407.01219)\]\[[code](https://github.com/FudanDNN-NLP/RAG)\]\[[Seven Failure Points When Engineering a Retrieval Augmented Generation System](https://arxiv.org/abs/2401.05856)\]
- **Improving Retrieval Augmented Language Model with Self-Reasoning**, _Xia et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2407.19813)\]
- **RAG Foundry: A Framework for Enhancing LLMs for Retrieval Augmented Generation**, _Fleischer et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2408.02545)\]\[[code](https://github.com/IntelLabs/RAGFoundry)\]\[[fastRAG](https://github.com/IntelLabs/fastRAG)\]
- **RAGEval: Scenario Specific RAG Evaluation Dataset Generation Framework**, _Zhu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2408.01262)\]\[[ragas](https://github.com/explodinggradients/ragas)\]
- **A Hybrid RAG System with Comprehensive Enhancement on Complex Reasoning**, _Yuan et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2408.05141)\]\[[code](https://gitlab.aicrowd.com/shizueyy/crag-new)\]\[[ind_kdd_2024/](https://www.biendata.net/competition/ind_kdd_2024/)\]\[[KDD2024-WhoIsWho-Top3](https://github.com/yanqiangmiffy/KDD2024-WhoIsWho-Top3)\]

- **ACL 2023 Tutorial: Retrieval-based Language Models and Applications**, _Asai et al._, ACL 2023. \[[link](https://acl2023-retrieval-lm.github.io/)\]
- \[[Advanced RAG Techniques: an Illustrated Overview](https://pub.towardsai.net/advanced-rag-techniques-an-illustrated-overview-04d193d8fec6)\]\[[Chinese Version](https://zhuanlan.zhihu.com/p/674755232)\]
- \[[LangChain](https://github.com/langchain-ai/langchain)\]\[[blog](https://blog.langchain.dev/deconstructing-rag/)\]\[[LangChain Hub](https://smith.langchain.com/hub)\]\[[langgraph](https://github.com/langchain-ai/langgraph)\]
- \[[LlamaIndex](https://github.com/run-llama/llama_index)\]\[[A Cheat Sheet and Some Recipes For Building Advanced RAG](https://blog.llamaindex.ai/a-cheat-sheet-and-some-recipes-for-building-advanced-rag-803a9d94c41b)\]\[[llama-agents](https://github.com/run-llama/llama-agents)\]
- \[[chatgpt-retrieval-plugin](https://github.com/openai/chatgpt-retrieval-plugin)\]
- \[[haystack](https://github.com/deepset-ai/haystack)\]\[[Langchain-Chatchat](https://github.com/chatchat-space/Langchain-Chatchat)\]
- \[[ragas](https://github.com/explodinggradients/ragas)\]
- **Browse the web with GPT-4V and Vimium** \[[vimGPT](https://github.com/ishan0102/vimGPT)\]
- \[[QAnything](https://github.com/netease-youdao/QAnything)\]\[[ragflow](https://github.com/infiniflow/ragflow)\]\[[fastRAG](https://github.com/IntelLabs/fastRAG)\]\[[anything-llm](https://github.com/Mintplex-Labs/anything-llm)\]\[[FastGPT](https://github.com/labring/FastGPT)\]\[[mem0](https://github.com/mem0ai/mem0)\]\[[Memary](https://github.com/kingjulio8238/Memary)\]
- \[[trt-llm-rag-windows](https://github.com/NVIDIA/trt-llm-rag-windows)\]\[[history_rag](https://github.com/wxywb/history_rag)\]\[[gpt-crawler](https://github.com/BuilderIO/gpt-crawler)\]\[[R2R](https://github.com/SciPhi-AI/R2R)\]\[[rag-notebook-to-microservices](https://github.com/wenqiglantz/rag-notebook-to-microservices)\]\[[MaxKB](https://github.com/1Panel-dev/MaxKB)\]\[[Verba](https://github.com/weaviate/Verba)\]\[[cognita](https://github.com/truefoundry/cognita)\]
- \[[RAG-Retrieval](https://github.com/NLPJCL/RAG-Retrieval)\]\[[FlashRank](https://github.com/PrithivirajDamodaran/FlashRank)\]\[[rank_bm25](https://github.com/dorianbrown/rank_bm25)\]\[[PGRAG](https://github.com/IAAR-Shanghai/PGRAG)\]\[[CRUD_RAG](https://github.com/IAAR-Shanghai/CRUD_RAG)\]\[[PlanRAG](https://github.com/myeon9h/PlanRAG)\]\[[DPA-RAG](https://github.com/dongguanting/DPA-RAG)\]
- \[[PDF-Extract-Kit](https://github.com/opendatalab/PDF-Extract-Kit)\]\[[colpali](https://github.com/illuin-tech/colpali)\]

###### Text Embedding

- **BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models**, _Thakur et al._, NeurIPS 2021. \[[paper](https://arxiv.org/abs/2104.08663)\]\[[code](https://github.com/beir-cellar/beir)\]
- **MTEB: Massive Text Embedding Benchmark**, _Muennighoff et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2210.07316)\]\[[code](https://github.com/embeddings-benchmark/mteb)\]\[[leaderboard](https://huggingface.co/spaces/mteb/leaderboard)\]
- **Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks**, _Reimers et al._, EMNLP 2019. \[[paper](https://arxiv.org/abs/1908.10084)\]\[[code](https://github.com/UKPLab/sentence-transformers)\]\[[model](https://huggingface.co/sentence-transformers)\]\[[vec2text](https://github.com/jxmorris12/vec2text)\]
- **SimCSE: Simple Contrastive Learning of Sentence Embeddings**, _Gao et al._, EMNLP 2021. \[[paper](https://arxiv.org/abs/2104.08821)\]\[[code](https://github.com/princeton-nlp/SimCSE)\]\[[AnglE ACL 2024](https://github.com/SeanLee97/AnglE)\]
- OpenAI: **Text and Code Embeddings by Contrastive Pre-Training**, _Neelakantan et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2201.10005)\]\[[blog](https://openai.com/blog/introducing-text-and-code-embeddings)\]
- MRL: **Matryoshka Representation Learning**, _Kusupati et al._, NeurIPS 2022. \[[paper](https://arxiv.org/abs/2205.13147)\]\[[code](https://github.com/RAIVNLab/MRL)\]
- BGE: **C-Pack: Packaged Resources To Advance General Chinese Embedding**, _Xiao et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.07597)\]\[[code](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/baai_general_embedding)\]\[[bge reranker](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/reranker)\]\[[FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)\]
- LLM-Embedder: **Retrieve Anything To Augment Large Language Models**, _Zhang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.07554)\]\[[code](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_embedder)\]\[[llm_reranker](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_reranker)\]\[[FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)\]
- **BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation**, _Chen et al._, arxiv 2024. \[[paper](https://export.arxiv.org/abs/2402.03216)\]\[[code](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/BGE_M3)\]\[[FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)\]
- \[[m3e-base](https://huggingface.co/moka-ai/m3e-base)\]\[[acge_text_embedding](https://huggingface.co/aspire/acge_text_embedding)\]\[[xiaobu-embedding-v2](https://huggingface.co/lier007/xiaobu-embedding-v2)\]\[[stella_en_1.5B_v5](https://huggingface.co/dunzhang/stella_en_1.5B_v5)\]
- **Jina Embeddings 2: 8192-Token General-Purpose Text Embeddings for Long Documents**, _Günther et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.19923)\]\[[jina-embeddings-v2](https://huggingface.co/jinaai/jina-embeddings-v2-base-en)\]\[[jina-reranker-v2](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual)\]\[[pe_rank](https://github.com/liuqi6777/pe_rank)\]\[[Jina CLIP](https://arxiv.org/abs/2405.20204)\]
- GTE: **Towards General Text Embeddings with Multi-stage Contrastive Learning**, _Li et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2308.03281)\]\[[model](https://huggingface.co/thenlper/gte-large-zh)\]\[[gte-Qwen2-7B-instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct)\]\[[gte-large-en-v1.5](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5)\]
- \[[BCEmbedding](https://github.com/netease-youdao/BCEmbedding)\]\[[bce-embedding-base_v1](https://huggingface.co/maidalun1020/bce-embedding-base_v1)\]\[[bce-reranker-base_v1](https://huggingface.co/maidalun1020/bce-reranker-base_v1)\]
- \[[CohereV3](https://huggingface.co/Cohere/Cohere-embed-multilingual-v3.0)\]
- **One Embedder, Any Task: Instruction-Finetuned Text Embeddings**, _Su et al._, ACL 2023. \[[paper](https://arxiv.org/abs/2212.09741)\]\[[code](https://github.com/xlang-ai/instructor-embedding)\]
- E5: **Improving Text Embeddings with Large Language Models**, _Wang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.00368)\]\[[code](https://github.com/microsoft/unilm/tree/master/e5)\]\[[model](https://huggingface.co/intfloat/e5-mistral-7b-instruct)\]\[[llm2vec](https://github.com/McGill-NLP/llm2vec)\]
- **Nomic Embed: Training a Reproducible Long Context Text Embedder**, _Nussbaum et al._, Nomic AI 2024. \[[paper](https://arxiv.org/abs/2402.01613)\]\[[code](https://github.com/nomic-ai/contrastors)\]
- GritLM: **Generative Representational Instruction Tuning**, _Muennighoff et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.09906)\]\[[code](https://github.com/ContextualAI/gritlm)\]
- **LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders**, _BehnamGhader et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2404.05961)\]\[[code](https://github.com/McGill-NLP/llm2vec)\]
- **NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models**, _Lee et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2405.17428)\]\[[model](https://huggingface.co/nvidia/NV-Embed-v1)\]
- PE-Rank: **Leveraging Passage Embeddings for Efficient Listwise Reranking with Large Language Models**, _Liu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2406.14848)\]\[[code](https://github.com/liuqi6777/pe_rank)\]

- \[[JamAIBase](https://github.com/EmbeddedLLM/JamAIBase)\]

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
- **Large Language Models Are Reasoning Teachers**, _Ho et al._, ACL 2023. \[[paper](https://arxiv.org/abs/2212.10071)\]\[[code](https://github.com/itsnamgyu/reasoning-teacher)\]
- **Least-to-Most Prompting Enables Complex Reasoning in Large Language Models**, _Zhou et al._, ICLR 2023. \[[paper](https://arxiv.org/abs/2205.10625)\]
- DEPS: **Describe, Explain, Plan and Select: Interactive Planning with Large Language Models Enables Open-World Multi-Task Agents**, _Wang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2302.01560)\]\[[code](https://github.com/CraftJarvis/MC-Planner)\]
- RAP: **Reasoning with Language Model is Planning with World Model**, _Hao et al._, EMNLP 2023. \[[paper](https://arxiv.org/abs/2305.14992)\]\[[code](https://github.com/maitrix-org/llm-reasoners)\]\[[LLM Reasoners COLM 2024](https://arxiv.org/abs/2404.05221)\]
- LEMA: **Learning From Mistakes Makes LLM Better Reasoner**, _An et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.20689)\]\[[code](https://github.com/microsoft/LEMA)\]
- **Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks**, _Chen et al._, TMLR 2023. \[[paper](https://arxiv.org/abs/2211.12588)\]\[[code](https://github.com/wenhuchen/Program-of-Thoughts)\]
- **Chain of Code: Reasoning with a Language Model-Augmented Code Emulator**, _Li et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.04474)\]\[[code]\]
- **The Impact of Reasoning Step Length on Large Language Models**, _Jin et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.04925)\]\[[code](https://github.com/jmyissb/The-Impact-of-Reasoning-Step-Length-on-Large-Language-Models)\]
- **Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models**, _Wang et al._, ACL 2023. \[[paper](https://arxiv.org/abs/2305.04091)\]\[[code](https://github.com/AGI-Edgerunners/Plan-and-Solve-Prompting)\]\[[maestro](https://github.com/Doriandarko/maestro)\]
- **Improving Factuality and Reasoning in Language Models through Multiagent Debate**, _Du et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.14325)\]\[[code](https://github.com/composable-models/llm_multiagent_debate)\]\[[Multi-Agents-Debate](https://github.com/Skytliang/Multi-Agents-Debate)\]
- **Self-Refine: Iterative Refinement with Self-Feedback**, _Madaan et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2303.17651)\]\[[code](https://github.com/madaan/self-refine)\]\[[MCT Self-Refine](https://github.com/trotsky1997/MathBlackBox)\]
- **Reflexion: Language Agents with Verbal Reinforcement Learning**, _Shinn et al._, NeurIPS 2023. \[[paper](https://arxiv.org/abs/2303.11366)\]\[[code](https://github.com/noahshinn/reflexion)\]
- **CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing**, _Gou et al._, ICLR 2024. \[[paper](https://arxiv.org/abs/2305.11738)\]\[[code](https://github.com/microsoft/ProphetNet/tree/master/CRITIC)\]
- LATS: **Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models**, _Zhou et al._, ICML 2024. \[[paper](https://arxiv.org/abs/2310.04406)\]\[[code](https://github.com/lapisrocks/LanguageAgentTreeSearch)\]
- **Self-Discover: Large Language Models Self-Compose Reasoning Structures**, _Zhou et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.03620)\]\[[unofficial implementation](https://github.com/catid/self-discover)\]\[[SELF-DISCOVER](https://github.com/kailashsp/SELF-DISCOVER)\]
- **RAT: Retrieval Augmented Thoughts Elicit Context-Aware Reasoning in Long-Horizon Generation**, _Wang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.05313)\]\[[code](https://github.com/CraftJarvis/RAT)\]
- **KnowAgent: Knowledge-Augmented Planning for LLM-Based Agents**, _Zhu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.03101)\]\[[code](https://github.com/zjunlp/KnowAgent)\]\[[KnowLM](https://github.com/zjunlp/KnowLM)\]
- **Advancing LLM Reasoning Generalists with Preference Trees**, _Yuan et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2404.02078)\]\[[code](https://github.com/OpenBMB/Eurus)\]
- **Buffer of Thoughts: Thought-Augmented Reasoning with Large Language Models**, _Yang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2406.04271)\]\[[code](https://github.com/YangLing0818/buffer-of-thought-llm)\]\[[SymbCoT](https://github.com/Aiden0526/SymbCoT)\]

- ReST-EM: **Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models**, _Singh et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.06585)\]\[[unofficial code](https://github.com/lucidrains/ReST-EM-pytorch)\]
- **ReST meets ReAct: Self-Improvement for Multi-Step Reasoning LLM Agent**, _Aksitov et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.10003)\]\[[code]\]
- **Orca 2: Teaching Small Language Models How to Reason**, _Mitra et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.11045)\]\[[code]\]
- Searchformer: **Beyond A*: Better Planning with Transformers via Search Dynamics Bootstrapping**, _Lehnert et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.14083)\]
- **How Far Are We from Intelligent Visual Deductive Reasoning?**, _Zhang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.04732)\]\[[code](https://github.com/apple/ml-rpm-bench)\]
- **Husky: A Unified, Open-Source Language Agent for Multi-Step Reasoning**, _Kim et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2406.06469)\]\[[code](https://github.com/agent-husky/Husky-v1)\]
- **Sibyl: Simple yet Effective Agent Framework for Complex Real-world Reasoning**, _Wang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2407.10718)\]\[[code](https://github.com/Ag2S1/Sibyl-System)\]
- **QueryAgent: A Reliable and Efficient Reasoning Framework with Environmental Feedback-based Self-Correction**, _Huang et al._, ACL 2024. \[[paper](https://arxiv.org/abs/2403.11886)\]\[[code](https://github.com/cdhx/QueryAgent)\]
- **Internal Consistency and Self-Feedback in Large Language Models: A Survey**, _Liang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2407.14507)\]\[[code](https://github.com/IAAR-Shanghai/ICSFSurvey)\]
- **Prover-Verifier Games improve legibility of language model outputs**, _Kirchner et al._, 2024. \[[blog](https://openai.com/index/prover-verifier-games-improve-legibility/)\]\[[paper](https://cdn.openai.com/prover-verifier-games-improve-legibility-of-llm-outputs/legibility.pdf)\]
- **Self-Training with Direct Preference Optimization Improves Chain-of-Thought Reasoning**, _Wang et al._, ACL 2024. \[[paper](https://arxiv.org/abs/2407.18248)\]\[[code](https://github.com/tianduowang/dpo-st)\]

- \[[llm-reasoners](https://github.com/maitrix-org/llm-reasoners)\]

###### Survey
- \[[Prompt4ReasoningPapers](https://github.com/zjunlp/Prompt4ReasoningPapers)\]


#### 3.4 LLM Theory

- **Scaling Laws for Neural Language Models**, _Kaplan et al._, arxiv 2020. \[[paper](https://arxiv.org/abs/2001.08361)\]\[[unofficial code](https://github.com/shehper/scaling_laws)\]
- **Emergent Abilities of Large Language Models**, _Wei et al._, TMRL 2022. \[[paper](https://arxiv.org/abs/2206.07682)\]
- Chinchilla: **Training Compute-Optimal Large Language Models**, _Hoffmann et al._, NeurIPS 2022. \[[paper](https://arxiv.org/abs/2203.15556)\]
- **Scaling Laws for Autoregressive Generative Modeling**, _Henighan et al._, arxiv 2020. \[[paper](https://arxiv.org/abs/2010.14701)\]
- **Are Emergent Abilities of Large Language Models a Mirage**, _Schaeffer et al._, NeurIPS 2023 Outstanding Paper. \[[paper](https://arxiv.org/abs/2304.15004)\]
- **Understanding Emergent Abilities of Language Models from the Loss Perspective**, _Du et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.15796)\]
- S2A: **System 2 Attention (is something you might need too)**, _Weston et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.11829)\]\[[Distilling System 2 into System 1](https://arxiv.org/abs/2407.06023)\]\[[system-2-research](https://github.com/open-thought/system-2-research)\]
- **Memory3: Language Modeling with Explicit Memory**, _Yang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2407.01178)\]
- **Scaling Laws for Downstream Task Performance of Large Language Models**, _Isik et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.04177)\]
- **Scalable Pre-training of Large Autoregressive Image Models**, _El-Nouby et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.08541)\]\[[code](https://github.com/apple/ml-aim)\]
- **When Scaling Meets LLM Finetuning: The Effect of Data, Model and Finetuning Method**, _Zhang et al._, ICLR 2024. \[[paper](https://arxiv.org/abs/2402.17193)\]
- **Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws**, _Allen-Zhu et al_, arxiv 2024. \[[paper](https://arxiv.org/abs/2404.05405)\]
- **Physics of Language Models: Part 2.1, Grade-School Math and the Hidden Reasoning Process**, _Ye et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2407.20311)\]\[[project page](https://physics.allen-zhu.com/part-2-grade-school-math/part-2-1)\]
- **Language Modeling Is Compression**, _Delétang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.10668)\]
- **Language Models Represent Space and Time**, _Gurnee and Tegmark_, ICLR 2024. \[[paper](https://arxiv.org/abs/2310.02207)\]\[[code](https://github.com/wesg52/world-models)\]
- **The Platonic Representation Hypothesis**, _Huh et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2405.07987)\]\[[code](https://github.com/minyoungg/platonic-rep)\]
- **Observational Scaling Laws and the Predictability of Language Model Performance**, _Ruan et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2405.10938)\]\[[code](https://github.com/ryoungj/ObsScaling)\]

- **Language models can explain neurons in language models**, _OpenAI_, 2023. \[[blog](https://openai.com/research/language-models-can-explain-neurons-in-language-models)\]\[[code](https://github.com/openai/automated-interpretability)\]\[[transformer-debugger](https://github.com/openai/transformer-debugger)\]
- **Scaling and evaluating sparse autoencoders**, _Gao et al._, arxiv 2024. \[[OpenAI Blog](https://openai.com/index/extracting-concepts-from-gpt-4/)\]\[[paper](https://arxiv.org/abs/2406.04093)\]\[[code](https://github.com/openai/sparse_autoencoder)\]\[[sae-auto-interp](https://github.com/EleutherAI/sae-auto-interp)\]
- **Towards Monosemanticity: Decomposing Language Models With Dictionary Learning**, _Anthropic_, 2023. \[[blog](https://www.anthropic.com/news/towards-monosemanticity-decomposing-language-models-with-dictionary-learning)\]
- **Mapping the Mind of a Large Language Model**, _Anthropic_, 2024. \[[blog](https://www.anthropic.com/research/mapping-mind-language-model)\]
- **Usable XAI: 10 Strategies Towards Exploiting Explainability in the LLM Era**, _Wu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.08946)\]\[[code](https://github.com/JacksonWuxs/UsableXAI_LLM)\]
- **LM Transparency Tool: Interactive Tool for Analyzing Transformer Language Models**, _Tufanov et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2404.07004)\]\[[code](https://github.com/facebookresearch/llm-transparency-tool)\]
- **Transformer Explainer: Interactive Learning of Text-Generative Models**, _Cho et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2408.04619)\]\[[code](https://github.com/poloclub/transformer-explainer)\]\[[demo](https://poloclub.github.io/transformer-explainer)\]
- **What needs to go right for an induction head? A mechanistic study of in-context learning circuits and their formation**, _Singh et al._, ICML 2024 Spotlight. \[[paper](https://arxiv.org/abs/2404.07129)\]\[[code](https://github.com/aadityasingh/icl-dynamics)\]
- \[[Transformer Circuits Thread](https://transformer-circuits.pub/)\]\[[colah's blog](http://colah.github.io/)\]\[[Transformer Interpretability](https://arena3-chapter1-transformer-interp.streamlit.app)\]\[[Awesome-Interpretability-in-Large-Language-Models](https://github.com/ruizheliUOA/Awesome-Interpretability-in-Large-Language-Models)\]\[[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)\]\[[inseq](https://github.com/inseq-team/inseq)\]

- ROME: **Locating and Editing Factual Associations in GPT**, _Meng et al._, NeurIPS 2022. \[[paper](https://arxiv.org/abs/2202.05262)\]\[[code](https://github.com/kmeng01/rome)\]\[[FastEdit](https://github.com/hiyouga/FastEdit)\]
- **Editing Large Language Models: Problems, Methods, and Opportunities**, _Yao et al._, EMNLP 2023. \[[paper](https://arxiv.org/abs/2305.13172)\]\[[code](https://github.com/zjunlp/EasyEdit)\]\[[Knowledge Mechanisms in Large Language Models: A Survey and Perspective](https://arxiv.org/abs/2407.15017)\]
- **A Comprehensive Study of Knowledge Editing for Large Language Models**, _Zhang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.01286)\]\[[code](https://github.com/zjunlp/EasyEdit)\]

#### 3.5 Chinese Model

- \[[Awesome-Chinese-LLM](https://github.com/HqWu-HITCS/Awesome-Chinese-LLM)\]\[[awesome-LLMs-In-China](https://github.com/wgwang/awesome-LLMs-In-China)\]\[[awesome-LLM-resourses](https://github.com/WangRongsheng/awesome-LLM-resourses)\]
- **GLM: General Language Model Pretraining with Autoregressive Blank Infilling**, _Du et al._, ACL 2022. \[[paper](https://arxiv.org/abs/2103.10360)\]\[[code](https://github.com/THUDM/GLM)\]
- **GLM-130B: An Open Bilingual Pre-trained Model**, _Zeng et al._, ICLR 2023. \[[paper](https://arxiv.org/abs/2210.02414v2)\]\[[code](https://github.com/THUDM/GLM-130B/)\]
- **ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools**, _Zeng et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2406.12793)\]\[[ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)\]\[[ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B)\]\[[ChatGLM3](https://github.com/THUDM/ChatGLM3)\]\[[GLM-4](https://github.com/THUDM/GLM-4)\]\[[AgentTuning](https://github.com/THUDM/AgentTuning)\]\[[AlignBench](https://github.com/THUDM/AlignBench)\]
- **Baichuan 2: Open Large-scale Language Models**, _Yang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.10305)\]\[[code](https://github.com/baichuan-inc/Baichuan2)\]
- **Qwen Technical Report**, _Bai et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.16609)\]\[[code](https://github.com/QwenLM/Qwen)\]
- **Qwen2 Technical Report**, _Yang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2407.10671)\]\[[code](https://github.com/QwenLM/Qwen2)\]\[[Qwen-Agent](https://github.com/QwenLM/Qwen-Agent)\]\[[AutoIF](https://github.com/QwenLM/AutoIF)\]
- **Yi: Open Foundation Models by 01.AI**, _Young et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.04652)\]\[[code](https://github.com/01-ai/Yi)\]\[[Yi-1.5](https://github.com/01-ai/Yi-1.5)\]
- **InternLM2 Technical Report**, _Cai et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.17297)\]\[[code](https://github.com/InternLM/InternLM)\]\[[HuixiangDou](https://github.com/InternLM/HuixiangDou)\]
- **DeepSeek LLM: Scaling Open-Source Language Models with Longtermism**, _Bi et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.02954)\]\[[DeepSeek-LLM](https://github.com/deepseek-ai/DeepSeek-LLM)\]\[[DeepSeek-V2](https://github.com/deepseek-ai/DeepSeek-V2)\]\[[DeepSeek-Coder)](https://github.com/deepseek-ai/DeepSeek-Coder)\]
- **TeleChat Technical Report**, _Wang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.03804)\]\[[code](https://github.com/Tele-AI/Telechat)\]\[[Tele-FLM Technical Report](https://arxiv.org/abs/2404.16645)\]\[[Tele-FLM](https://huggingface.co/CofeAI/Tele-FLM)\]\[[Tele-FLM-1T](https://huggingface.co/CofeAI/Tele-FLM-1T)\]
- **Efficient and Effective Text Encoding for Chinese LLaMA and Alpaca**, Cui et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2304.08177)\]\[[code](https://github.com/ymcui/Chinese-LLaMA-Alpaca)\]\[[Chinese-LLaMA-Alpaca-2](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2)\]\[[Chinese-LLaMA-Alpaca-3](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3)\]\[[baby-llama2-chinese](https://github.com/DLLXW/baby-llama2-chinese)\]
- **Rethinking Optimization and Architecture for Tiny Language Models**, _Tang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.02791)\]\[[code](https://github.com/YuchuanTian/RethinkTinyLM)\]
- **Towards Effective and Efficient Continual Pre-training of Large Language Models**, _Chen et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2407.18743)\]\[[code](https://github.com/RUC-GSAI/Llama-3-SynE)\]
- \[[MOSS](https://github.com/OpenLMLab/MOSS)\]\[[MOSS-RLHF](https://github.com/OpenLMLab/MOSS-RLHF)\]
- \[[MiniCPM](https://github.com/OpenBMB/MiniCPM)\]\[[Skywork](https://github.com/SkyworkAI/Skywork)\]\[[Skywork-MoE](https://github.com/SkyworkAI/Skywork-MoE)\]\[[Orion](https://github.com/OrionStarAI/Orion)\]\[[BELLE](https://github.com/LianjiaTech/BELLE)\]\[[Yuan-2.0](https://github.com/IEIT-Yuan/Yuan-2.0)\]\[[Yuan2.0-M32](https://github.com/IEIT-Yuan/Yuan2.0-M32)\]\[[Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM)\]\[[Index-1.9B](https://github.com/bilibili/Index-1.9B)\]
- \[[LlamaFamily/Llama-Chinese](https://github.com/LlamaFamily/Llama-Chinese)\]\[[LinkSoul-AI/Chinese-Llama-2-7b](https://github.com/LinkSoul-AI/Chinese-Llama-2-7b)\]\[[llama3-Chinese-chat](https://github.com/CrazyBoyM/llama3-Chinese-chat)\]\[[phi3-Chinese](https://github.com/CrazyBoyM/phi3-Chinese)\]\[[LLM-Chinese](https://github.com/CrazyBoyM/LLM-Chinese)\]\[[Llama3-Chinese-Chat](https://github.com/Shenzhi-Wang/Llama3-Chinese-Chat)\]\[[llama3-chinese](https://github.com/seanzhang-zhichen/llama3-chinese)\]
- \[[Firefly](https://github.com/yangjianxin1/Firefly)\]\[[GPT2-chitchat](https://github.com/yangjianxin1/GPT2-chitchat)\]
- Alpaca-CoT: **An Empirical Study of Instruction-tuning Large Language Models in Chinese**, _Si et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.07328)\]\[[code](https://github.com/PhoebusSi/Alpaca-CoT)\]

---

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
- ConvNeXt: **A ConvNet for the 2020s**, _Liu et al._, CVPR 2022. \[[paper](https://arxiv.org/abs/2201.03545)\]\[[code](https://github.com/facebookresearch/ConvNeXt)\]

### 2. Contrastive Learning

- MoCo: **Momentum Contrast for Unsupervised Visual Representation Learning**, _He et al._, CVPR 2020. \[[paper](https://arxiv.org/abs/1911.05722)\]\[[code](https://github.com/facebookresearch/moco)\]
- SimCLR: **A Simple Framework for Contrastive Learning of Visual Representations**, _Chen et al._, PMLR 2020. \[[paper](https://arxiv.org/abs/2002.05709)\]\[[code](https://github.com/google-research/simclr)\]
- **DINOv2: Learning Robust Visual Features without Supervision**, _Oquab et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2304.07193)\]\[[code](https://github.com/facebookresearch/dinov2)\]
- **FeatUp: A Model-Agnostic Framework for Features at Any Resolution**, _Fu et al._, ICLR 2024. \[[paper](https://arxiv.org/abs/2403.10516v1)\]\[[code](https://github.com/mhamilton723/FeatUp)\]

- InfoNCE Loss: **Representation Learning with Contrastive Predictive Coding**, _Oord et al._, arxiv 2018. \[[paper](https://arxiv.org/abs/1807.03748)\]\[[unofficial code](https://github.com/jefflai108/Contrastive-Predictive-Coding-PyTorch)\]

### 3. CV Application

- **NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis**, _Mildenhall et al._, ECCV 2020. \[[paper](https://arxiv.org/abs/2003.08934)\]\[[code](https://github.com/bmild/nerf)\]\[[nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch)\]\[[NeRF-Factory](https://github.com/kakaobrain/NeRF-Factory)\]\[[LERF](https://github.com/kerrj/lerf)\]\[[LangSplat](https://github.com/minghanqin/LangSplat)\]
- GFP-GAN: **Towards Real-World Blind Face Restoration with Generative Facial Prior**, _Wang et al._, CVPR 2021. \[[paper](https://arxiv.org/abs/2101.04061)\]\[[code](https://github.com/TencentARC/GFPGAN)\]
- CodeFormer: **Towards Robust Blind Face Restoration with Codebook Lookup Transformer**, _Zhou et al._, NeurIPS 2022. \[[paper](https://arxiv.org/abs/2206.11253)\]\[[code](https://github.com/sczhou/CodeFormer)\]\[[APISR](https://github.com/Kiteretsu77/APISR)\]\[[EvTexture](https://github.com/DachunKai/EvTexture)\]\[[video2x](https://github.com/k4yt3x/video2x)\]
- **BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers**, _Li et al._, ECCV 2022. \[[paper](https://arxiv.org/abs/2203.17270)\]\[[code](https://github.com/fundamentalvision/BEVFormer)\]\[[occupancy_networks](https://github.com/autonomousvision/occupancy_networks)\]\[[VoxFormer](https://github.com/NVlabs/VoxFormer)\]\[[TPVFormer](https://github.com/wzzheng/TPVFormer)\]\[[GeMap](https://github.com/cnzzx/GeMap)\]
- UniAD: **Planning-oriented Autonomous Driving**, _Hu et al._, CVPR 2023 Best Paper. \[[paper](https://arxiv.org/abs/2212.10156)\]\[[code](https://github.com/OpenDriveLab/UniAD)\]
- \[[apollo](https://github.com/ApolloAuto/apollo)\]\[[dagr](https://github.com/uzh-rpg/dagr)\]

- **Nougat: Neural Optical Understanding for Academic Documents**, _Blecher et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2308.13418)\]\[[code](https://github.com/facebookresearch/nougat)\]\[[marker](https://github.com/VikParuchuri/marker)\]\[[kosmos-2.5](https://github.com/microsoft/unilm/tree/master/kosmos-2.5)\]\[[gptpdf](https://github.com/CosmosShadow/gptpdf)\]\[[omniparse](https://github.com/adithya-s-k/omniparse)\]\[[llama_parse](https://github.com/run-llama/llama_parse)\]\[[PDF-Extract-Kit](https://github.com/opendatalab/PDF-Extract-Kit)\]
- **FaceChain: A Playground for Identity-Preserving Portrait Generation**, _Liu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2308.14256)\]\[[code](https://github.com/modelscope/facechain)\]
- MGIE: **Guiding Instruction-based Image Editing via Multimodal Large Language Models**, _Fu et al._, ICLR 2024 Spotlight. \[[paper](https://arxiv.org/abs/2309.17102)\]\[[code](https://github.com/apple/ml-mgie)\]
- **PhotoMaker: Customizing Realistic Human Photos via Stacked ID Embedding**, _Li et al._, CVPR 2024. \[[paper](https://arxiv.org/abs/2312.04461)\]\[[code](https://github.com/TencentARC/PhotoMaker)\]\[[AnyDoor](https://github.com/ali-vilab/AnyDoor)\]
- **InstantID: Zero-shot Identity-Preserving Generation in Seconds**, _Wang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.07519)\]\[[code](https://github.com/InstantID/InstantID)\]\[[InstantStyle](https://github.com/InstantStyle/InstantStyle)\]\[[ID-Animator](https://github.com/ID-Animator/ID-Animator)\]\[[ConsistentID](https://github.com/JackAILab/ConsistentID)\]\[[ComfyUI-InstantID](https://github.com/ZHO-ZHO-ZHO/ComfyUI-InstantID)\]
- **ReplaceAnything as you want: Ultra-high quality content replacement**, \[[link](https://aigcdesigngroup.github.io/replace-anything/)\]\[[OutfitAnyone](https://humanaigc.github.io/outfit-anyone/)\]\[[IDM-VTON](https://github.com/yisol/IDM-VTON)\]\[[IMAGDressing](https://github.com/muzishen/IMAGDressing)\]\[[CatVTON](https://github.com/Zheng-Chong/CatVTON)\]
- LayerDiffusion: **Transparent Image Layer Diffusion using Latent Transparency**, _Zhang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.17113)\]\[[code](https://github.com/layerdiffusion/LayerDiffusion)\]\[[sd-forge-layerdiffusion](https://github.com/layerdiffusion/sd-forge-layerdiffusion)\]\[[LayerDiffuse_DiffusersCLI](https://github.com/lllyasviel/LayerDiffuse_DiffusersCLI)\]\[[IC-Light](https://github.com/lllyasviel/IC-Light)\]\[[Paints-UNDO](https://github.com/lllyasviel/Paints-UNDO)\]
- **Unique3D: High-Quality and Efficient 3D Mesh Generation from a Single Image**, _Wu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2405.20343)\]\[[code](https://github.com/AiuniAI/Unique3D)\]\[[MeshAnything](https://github.com/buaacyw/MeshAnything)\]\[[MeshAnythingV2](https://github.com/buaacyw/MeshAnythingV2)\]\[[InstantMesh](https://github.com/TencentARC/InstantMesh)\]\[[prolificdreamer](https://github.com/thu-ml/prolificdreamer)\]
- **SF3D: Stable Fast 3D Mesh Reconstruction with UV-unwrapping and Illumination Disentanglement**, _Boss et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2408.00653)\]\[[code](https://github.com/Stability-AI/stable-fast-3d)\]

- \[[deepfakes/faceswap](https://github.com/deepfakes/faceswap)\]\[[DeepFaceLab](https://github.com/iperov/DeepFaceLab)\]\[[DeepFaceLive](https://github.com/iperov/DeepFaceLive)\]\[[deepface](https://github.com/serengil/deepface)\]\[[Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam)\]
- \[[IOPaint](https://github.com/Sanster/IOPaint)\]\[[SPADE](https://github.com/NVlabs/SPADE)\]\[[EasyOCR](https://github.com/JaidedAI/EasyOCR)\]\[[PowerPaint](https://github.com/open-mmlab/PowerPaint)\]
- \[[MuseV](https://github.com/TMElyralab/MuseV)\]\[[ToonCrafter](https://github.com/ToonCrafter/ToonCrafter)\]
- \[[supervision](https://github.com/roboflow/supervision)\]

### 4. Foundation Model

- ViT: **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**, _Dosovitskiy et al._, ICLR 2021. \[[paper](https://arxiv.org/abs/2010.11929)\]\[[code](https://github.com/google-research/vision_transformer)\]\[[vit-pytorch](https://github.com/lucidrains/vit-pytorch)\]\[[efficientvit](https://github.com/mit-han-lab/efficientvit)\]\[[EfficientFormer](https://github.com/snap-research/EfficientFormer)\]\[[ViT-Adapter](https://github.com/czczup/ViT-Adapter)\]
- ViT-Adapter: **Vision Transformer Adapter for Dense Predictions**, _Chen et al._, ICLR 2023 Spotlight. \[[paper](https://arxiv.org/abs/2205.08534)\]\[[code](https://github.com/czczup/ViT-Adapter)\]
- **Vision Transformers Need Registers**, _Darcet et al._, ICLR 2024 Outstanding Paper. \[[paper](https://arxiv.org/abs/2309.16588)\]
- DeiT: **Training data-efficient image transformers & distillation through attention**, _Touvron et al._, ICML 2021. \[[paper](https://arxiv.org/abs/2012.12877)\]\[[code](https://github.com/facebookresearch/deit)\]
- **ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision**, _Kim et al._, ICML 2021. \[[paper](https://arxiv.org/abs/2102.03334)\]\[[code](https://github.com/dandelin/vilt)\]
- **Swin Transformer: Hierarchical Vision Transformer using Shifted Windows**, _Liu et al._, ICCV 2021. \[[paper](https://arxiv.org/abs/2103.14030)\]\[[code](https://github.com/microsoft/Swin-Transformer)\]
- MAE: **Masked Autoencoders Are Scalable Vision Learners**, _He et al._, CVPR 2022. \[[paper](https://arxiv.org/abs/2111.06377)\]\[[code](https://github.com/facebookresearch/mae)\]
- **Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks**, _Xiao et al._, CVPR 2024 Oral. \[[paper](https://arxiv.org/abs/2311.06242)\]\[[model](https://huggingface.co/collections/microsoft/florence-6669f44df0d87d9c3bfb76de)\]\[[Inference code](https://huggingface.co/microsoft/Florence-2-large/blob/main/sample_inference.ipynb)\]
- LVM: **Sequential Modeling Enables Scalable Learning for Large Vision Models**, _Bai et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.00785)\]\[[code](https://github.com/ytongbai/LVM)\]
- GLEE: **General Object Foundation Model for Images and Videos at Scale**, _Wu wt al._, CVPR 2024 Highlight. \[[paper](https://arxiv.org/abs/2312.09158)\]\[[code](https://github.com/FoundationVision/GLEE)\]
- **Tokenize Anything via Prompting**, _Pan et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.09128)\]\[[code](https://github.com/baaivision/tokenize-anything)\]
- **Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model** _Zhu et al._, ICML 2024. \[[paper](https://arxiv.org/abs/2401.09417)\]\[[code](https://github.com/hustvl/Vim)\]\[[VMamba](https://github.com/MzeroMiko/VMamba)\]\[[mambaout](https://github.com/yuweihao/mambaout)\]
- **MambaVision: A Hybrid Mamba-Transformer Vision Backbone**, _Hatamizadeh and Kautz_, arxiv 2024. \[[paper](https://arxiv.org/abs/2407.08083)\]\[[code](https://github.com/NVlabs/MambaVision)\]
- **Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data**, _Yang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.10891)\]\[[code](https://github.com/LiheYoung/Depth-Anything)\]\[[Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)\]
- **Vision Superalignment: Weak-to-Strong Generalization for Vision Foundation Models**, _Guo et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.03749)\]\[[code](https://github.com/ggjy/vision_weak_to_strong)\]
- TiTok: **An Image is Worth 32 Tokens for Reconstruction and Generation**, _Yu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2406.07550)\]\[[titok-pytorch](https://github.com/lucidrains/titok-pytorch)\]
- **Theia: Distilling Diverse Vision Foundation Models for Robot Learning**, _Shang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2407.20179)\]\[[code](https://github.com/bdaiinstitute/theia)\]

- \[[pytorch-image-models](https://github.com/huggingface/pytorch-image-models)\]\[[Pointcept](https://github.com/Pointcept/Pointcept)\]

### 5. Generative Model (GAN and VAE)

- GAN: **Generative Adversarial Networks**, _Goodfellow et al._, arxiv 2014. \[[paper](https://arxiv.org/abs/1406.2661)\]\[[code](https://github.com/goodfeli/adversarial)\]\[[Pytorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN)\]
- StyleGAN3: **Alias-Free Generative Adversarial Networks**, _Karras etal._, NeurIPS 2021. \[[paper](https://arxiv.org/abs/2106.12423)\]\[[code](https://github.com/NVlabs/stylegan3)\]\[[StyleFeatureEditor](https://github.com/AIRI-Institute/StyleFeatureEditor)\]
- GigaGAN: **Scaling up GANs for Text-to-Image Synthesis**, _Kang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2303.05511)\]\[[code](https://github.com/lucidrains/gigagan-pytorch)\]
- \[[pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)\]\[[img2img-turbo](https://github.com/GaParmar/img2img-turbo)\]
- VAE: **Auto-Encoding Variational Bayes**, _Kingma et al._, arxiv 2013. \[[paper](https://arxiv.org/abs/1312.6114)\]\[[code](https://github.com/jaanli/variational-autoencoder)\]\[[Pytorch-VAE](https://github.com/AntixK/PyTorch-VAE)\]
- VQ-VAE: **Neural Discrete Representation Learning**, _Oord et al._, NIPS 2017. \[[paper](https://arxiv.org/abs/1711.00937)\]\[[code](https://github.com/AntixK/PyTorch-VAE/blob/master/models/vq_vae.py)\]\[[vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch)\]
- VQ-VAE-2: **Generating Diverse High-Fidelity Images with VQ-VAE-2**, _Razavi et al._, arxiv 2019. \[[paper](https://arxiv.org/abs/1906.00446)\]\[[code](https://github.com/rosinality/vq-vae-2-pytorch)\]
- VQGAN: **Taming Transformers for High-Resolution Image Synthesis**, _Esser et al._, CVPR 2021. \[[paper](https://arxiv.org/abs/2012.09841)\]\[[code](https://github.com/CompVis/taming-transformers)\]
- **Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction**, _Tian et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2404.02905)\]\[[code](https://github.com/FoundationVision/VAR)\]\[[LlamaGen](https://github.com/FoundationVision/LlamaGen)\]
- **Autoregressive Image Generation without Vector Quantization**, _Li et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2406.11838)\]\[[code](https://github.com/LTH14/mar)\]\[[autoregressive-diffusion-pytorch](https://github.com/lucidrains/autoregressive-diffusion-pytorch)\]

### 6. Image Editing

- **InstructPix2Pix: Learning to Follow Image Editing Instructions**, _Brooks et al._, CVPR 2023 Highlight. \[[paper](https://arxiv.org/abs/2211.09800)\]\[[code](https://github.com/timothybrooks/instruct-pix2pix)\]
- **Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold**, _Pan et al._, SIGGRAPH 2023. \[[paper](https://arxiv.org/abs/2305.10973)\]\[[code](https://github.com/XingangPan/DragGAN)\]
- **DragDiffusion: Harnessing Diffusion Models for Interactive Point-based Image Editing**, _Shi et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2306.14435)\]\[[code](https://github.com/Yujun-Shi/DragDiffusion)\]
- **DragonDiffusion: Enabling Drag-style Manipulation on Diffusion Models**, _Mou et al._, ICLR 2024 Spolight. \[[paper](https://arxiv.org/abs/2307.02421)\]\[[code](https://github.com/MC-E/DragonDiffusion)\]
- **LEDITS++: Limitless Image Editing using Text-to-Image Models**, _Brack et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.16711)\]\[[code](https://huggingface.co/spaces/editing-images/leditsplusplus/tree/main)\]\[[demo](https://huggingface.co/spaces/editing-images/leditsplusplus)\]
- **Diffusion Model-Based Image Editing: A Survey**, _Huang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.17525)\]\[[code](https://github.com/SiatMMLab/Awesome-Diffusion-Model-Based-Image-Editing-Methods)\]
- MimicBrush: **Zero-shot Image Editing with Reference Imitation**, _Chen et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2406.07547)\]\[[code](https://github.com/ali-vilab/MimicBrush)\]\[[EchoMimic](https://github.com/BadToBest/EchoMimic)\]
- **A Survey of Multimodal-Guided Image Editing with Text-to-Image Diffusion Models**, _Shuai et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2406.14555)\]\[[code](https://github.com/xinchengshuai/Awesome-Image-Editing)\]

- \[[ComfyUI-UltraEdit-ZHO](https://github.com/ZHO-ZHO-ZHO/ComfyUI-UltraEdit-ZHO)\]

### 7. Object Detection

- DETR: **End-to-End Object Detection with Transformers**, _Carion et al._, arxiv 2020. \[[paper](https://arxiv.org/abs/2005.12872)\]\[[code](https://github.com/facebookresearch/detr)\]
- Focus-DETR: **Less is More_Focus Attention for Efficient DETR**, _Zheng et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2307.12612)\]\[[code](https://github.com/huawei-noah/noah-research)\]
- **U2-Net_Going Deeper with Nested U-Structure for Salient Object Detection**, _Qin et al._, arxiv 2020. \[[paper](https://arxiv.org/abs/2005.09007)\]\[[code](https://github.com/xuebinqin/U-2-Net)\]
- YOLO: **You Only Look Once: Unified, Real-Time Object Detection** _Redmon et al._, arxiv 2015. \[[paper](https://arxiv.org/abs/1506.02640)\]
- **YOLOX: Exceeding YOLO Series in 2021**, _Ge et al._, arxiv 2021. \[[paper](https://arxiv.org/abs/2107.08430)\]\[[code](https://github.com/Megvii-BaseDetection/YOLOX)\]
- **Gold-YOLO: Efficient Object Detector via Gather-and-Distribute Mechanism**, _Wang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.11331)\]\[[code](https://github.com/huawei-noah/Efficient-Computing/tree/master/Detection/Gold-YOLO)\]
- **Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection**, _Liu et al._, ECCV 2024. \[[paper](https://arxiv.org/abs/2303.05499)\]\[[code](https://github.com/IDEA-Research/GroundingDINO)\]\[[OV-DINO](https://github.com/wanghao9610/OV-DINO)\]
- **YOLO-World: Real-Time Open-Vocabulary Object Detection**, _Cheng et al._, CVPR 2024. \[[paper](https://arxiv.org/abs/2401.17270)\]\[[code](https://github.com/ailab-cvc/yolo-world)\]
- **YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information**, _Wang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.13616)\]\[[code](https://github.com/WongKinYiu/yolov9)\]
- **T-Rex2: Towards Generic Object Detection via Text-Visual Prompt Synergy**, _Jiang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.14610)\]\[[code](https://github.com/IDEA-Research/T-Rex)\]
- **YOLOv10: Real-Time End-to-End Object Detection**, _Wang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2405.14458)\]\[[code](https://github.com/THU-MIG/yolov10)\]

- \[[detectron2](https://github.com/facebookresearch/detectron2)\]\[[yolov5](https://github.com/ultralytics/yolov5)\]\[[mmdetection](https://github.com/open-mmlab/mmdetection)\]\[[detrex](https://github.com/IDEA-Research/detrex)\]\[[ultralytics](https://github.com/ultralytics/ultralytics)\]

### 8. Semantic Segmentation

- **U-Net: Convolutional Networks for Biomedical Image Segmentation**, _Ronneberger et al._, MICCAI 2015. \[[paper](https://arxiv.org/abs/1505.04597)\]\[[Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)\]\[[xLSTM-UNet-Pytorch](https://github.com/tianrun-chen/xLSTM-UNet-Pytorch)\]
- **Segment Anything**, _Kirillov et al._, ICCV 2023. \[[paper](https://arxiv.org/abs/2304.02643)\]\[[code](https://github.com/facebookresearch/segment-anything)\]\[[SAM-Adapter-PyTorch](https://github.com/tianrun-chen/SAM-Adapter-PyTorch)\]
- **SAM 2: Segment Anything in Images and Videos**, _Ravi et al._, SIGGRAPH 2024. \[[blog](https://ai.meta.com/blog/segment-anything-2/)\]\[[paper](https://arxiv.org/abs/2408.00714)\]\[[code](https://github.com/facebookresearch/segment-anything-2)\]
- **EfficientSAM: Leveraged Masked Image Pretraining for Efficient Segment Anything**, _Xiong et al._, CVPR 2024. \[[paper](https://arxiv.org/abs/2312.00863)\]\[[code](https://github.com/yformer/EfficientSAM)\]\[[RobustSAM](https://github.com/robustsam/RobustSAM)\]\[[MobileSAM](https://github.com/ChaoningZhang/MobileSAM)\]
- **Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks**, _Ren et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.14159)\]\[[code](https://github.com/IDEA-Research/Grounded-Segment-Anything)\]\[[Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2)\]
- **LISA: Reasoning Segmentation via Large Language Model**, _Lai et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2308.00692)\]\[[code](https://github.com/dvlab-research/LISA)\]
- **OMG-LLaVA: Bridging Image-level, Object-level, Pixel-level Reasoning and Understanding**, _Zhang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2406.19389)\]\[[code](https://github.com/lxtGH/OMG-Seg)\]

- \[[mmsegmentation](https://github.com/open-mmlab/mmsegmentation)\]\[[mmdeploy](https://github.com/open-mmlab/mmdeploy)\]\[[Painter](https://github.com/baaivision/Painter)\]

### 9. Video

- **VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training**, _Tong et al._, NeurIPS 2022 Spotlight. \[[paper](https://arxiv.org/abs/2203.12602)\]\[[code](https://github.com/MCG-NJU/VideoMAE)\]
- **MagicVideo-V2: Multi-Stage High-Aesthetic Video Generation**, _Wang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.04468)\]
- \[[V-JEPA](https://github.com/facebookresearch/jepa)\]\[[I-JEPA](https://github.com/facebookresearch/ijepa)\]
- **VideoMamba: State Space Model for Efficient Video Understanding**, _Li et al._, ECCV 2024. \[[paper](https://arxiv.org/abs/2403.06977)\]\[[code](https://github.com/OpenGVLab/VideoMamba)\]
- **VideoChat: Chat-Centric Video Understanding**, _Li et al._, CVPR 2024 Highlight. \[[paper](https://arxiv.org/abs/2305.06355)\]\[[code](https://github.com/OpenGVLab/Ask-Anything)\]
- **MVBench: A Comprehensive Multi-modal Video Understanding Benchmark**, _Li et al._, CVPR 2024 Highlight. \[[paper](https://arxiv.org/abs/2311.17005)\]\[[code](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2)\]
- **MiraData: A Large-Scale Video Dataset with Long Durations and Structured Captions**, _Ju et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2407.06358)\]\[[code](https://github.com/mira-space/MiraData)\]

### 10. Survey for CV

- **ConvNet vs Transformer, Supervised vs CLIP: Beyond ImageNet Accuracy**, _Vishniakov et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.09215)\]\[[code](https://github.com/kirill-vish/Beyond-INet)\]
- **Parameter-Efficient Fine-Tuning for Pre-Trained Vision Models: A Survey**, _Xin et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.02242)\]\[[code](https://github.com/synbol/Awesome-Parameter-Efficient-Transfer-Learning)\]

---

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
- **Amphion: An Open-Source Audio, Music and Speech Generation Toolkit**, _Zhang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.09911)\]\[[code](https://github.com/open-mmlab/Amphion)\]\[[FoleyCrafter](https://github.com/open-mmlab/FoleyCrafter)\]\[[vta-ldm](https://github.com/ariesssxu/vta-ldm)\]
- VITS: **Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech**, _Kim et al._, ICML 2021. \[[paper](https://arxiv.org/abs/2106.06103)\]\[[code](https://github.com/jaywalnut310/vits)\]\[[Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)\]\[[so-vits-svc-fork](https://github.com/voicepaw/so-vits-svc-fork)\]\[[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)\]\[[VITS-fast-fine-tuning](https://github.com/Plachtaa/VITS-fast-fine-tuning)\]
- **OpenVoice: Versatile Instant Voice Cloning**, _Qin et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.01479)\]\[[code](https://github.com/myshell-ai/OpenVoice)\]\[[MockingBird](https://github.com/babysor/MockingBird)\]\[[clone-voice](https://github.com/jianchang512/clone-voice)\]
- **NaturalSpeech 3: Zero-Shot Speech Synthesis with Factorized Codec and Diffusion Models**, _Ju et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.03100)\]\[[Seed-TTS](https://arxiv.org/abs/2406.02430)\]\[[e2-tts-pytorch](https://github.com/lucidrains/e2-tts-pytorch)\]
- **VoiceCraft: Zero-Shot Speech Editing and Text-to-Speech in the Wild**, _Peng et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.16973)\]\[[code](https://github.com/jasonppy/voicecraft)\]
- **WavLLM: Towards Robust and Adaptive Speech Large Language Model**, _Hu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2404.00656)\]\[[code](https://github.com/microsoft/SpeechT5/tree/main/WavLLM)\]
- **Hallo: Hierarchical Audio-Driven Visual Synthesis for Portrait Image Animation**, _Xu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2406.08801)\]\[[code](https://github.com/fudan-generative-vision/hallo)\]\[[champ](https://github.com/fudan-generative-vision/champ)\]
- **StreamSpeech: Simultaneous Speech-to-Speech Translation with Multi-task Learning**, _Zhang et al._, ACL 2024. \[[paper](https://arxiv.org/abs/2406.03049)\]\[[code](https://github.com/ictnlp/StreamSpeech)\]
- **FunAudioLLM: Voice Understanding and Generation Foundation Models for Natural Interaction Between Humans and LLMs**, _Tongyi Speech Team_, arxiv 2024. \[[paper](https://arxiv.org/abs/2407.04051)\]\[[code](https://github.com/FunAudioLLM)\]\[[CosyVoice](https://github.com/FunAudioLLM/CosyVoice)\]
- **Qwen2-Audio Technical Report**, _Chu et al._, arxiv 2024. \[[blog](https://qwenlm.github.io/blog/qwen2-audio/)\]\[[paper](https://arxiv.org/abs/2407.10759)\]\[[code](https://github.com/QwenLM/Qwen2-Audio)\]\[[Qwen-Audio](https://github.com/QwenLM/Qwen-Audio)\]
- **Language Model Can Listen While Speaking**, _Ma et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2408.02622)\]\[[demo](https://ziyang.tech/LSLM/)\]

- **Github Repositories**
- \[[coqui-ai/TTS](https://github.com/coqui-ai/TTS)\]\[[suno-ai/bark](https://github.com/suno-ai/bark)\]\[[ChatTTS](https://github.com/2noise/ChatTTS)\]\[[WhisperSpeech](https://github.com/collabora/WhisperSpeech)\]\[[MeloTTS](https://github.com/myshell-ai/MeloTTS)\]\[[parler-tts](https://github.com/huggingface/parler-tts)\]\[[fish-speech](https://github.com/fishaudio/fish-speech)\]\[[MARS5-TTS](https://github.com/Camb-ai/MARS5-TTS)\]\[[metavoice-src](https://github.com/metavoiceio/metavoice-src)\]
- \[[stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools)\]\[[Qwen-Audio](https://github.com/QwenLM/Qwen-Audio)\]\[[pyannote-audio](https://github.com/pyannote/pyannote-audio)\]\[[ims-toucan](https://github.com/digitalphonetics/ims-toucan)\]\[[AudioLCM](https://github.com/Text-to-Audio/AudioLCM)\]
- \[[FunASR](https://github.com/alibaba-damo-academy/FunASR)\]\[[FunClip](https://github.com/alibaba-damo-academy/FunClip)\]\[[FunAudioLLM](https://github.com/FunAudioLLM)\]\[[TeleSpeech-ASR](https://github.com/Tele-AI/TeleSpeech-ASR)\]\[[EmotiVoice](https://github.com/netease-youdao/EmotiVoice)\]
- \[[SadTalker](https://github.com/OpenTalker/SadTalker)\]\[[Wav2Lip](https://github.com/Rudrabha/Wav2Lip)\]\[[video-retalking](https://github.com/OpenTalker/video-retalking)\]\[[SadTalker-Video-Lip-Sync](https://github.com/Zz-ww/SadTalker-Video-Lip-Sync)\]\[[AniPortrait](https://github.com/Zejun-Yang/AniPortrait)\]\[[GeneFacePlusPlus](https://github.com/yerfor/GeneFacePlusPlus)\]\[[V-Express](https://github.com/tencent-ailab/V-Express)\]\[[MuseTalk](https://github.com/TMElyralab/MuseTalk)\]\[[EchoMimic](https://github.com/BadToBest/EchoMimic)\]
- \[[Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)\]\[[Awesome-ChatTTS](https://github.com/panyanyany/Awesome-ChatTTS)\]
- \[[speech-trident](https://github.com/ga642381/speech-trident)\]\[[AudioNotes](https://github.com/harry0703/AudioNotes)\]

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

- CLIP: **Learning Transferable Visual Models From Natural Language Supervision**, _Radford et al._, ICML 2021. \[[paper](https://arxiv.org/abs/2103.00020)\]\[[code](https://github.com/OpenAI/CLIP)\]\[[clip-as-service](https://github.com/jina-ai/clip-as-service)\]\[[open_clip](https://github.com/mlfoundations/open_clip)\]\[[EVA](https://github.com/baaivision/EVA)\]\[[DIVA](https://github.com/baaivision/DIVA)\]
- DALL-E2: **Hierarchical Text-Conditional Image Generation with CLIP Latents**, _Ramesh et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2204.06125)\]\[[code](https://github.com/lucidrains/DALLE2-pytorch)\]
- **HiCLIP: Contrastive Language-Image Pretraining with Hierarchy-aware Attention**, _Geng et al._, ICLR 2023. \[[paper](https://arxiv.org/abs/2303.02995)\]\[[code](https://github.com/jeykigung/HiCLIP)\]
- **Chinese CLIP: Contrastive Vision-Language Pretraining in Chinese**, _Yang et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2211.01335)\]\[[code](https://github.com/OFA-Sys/Chinese-CLIP)\]
- MetaCLIP: **Demystifying CLIP Data**, _Xu et al._, ICLR 2024 Spotlight. \[[paper](https://arxiv.org/abs/2309.16671)\]\[[code](https://github.com/facebookresearch/MetaCLIP)\]
- **Alpha-CLIP: A CLIP Model Focusing on Wherever You Want**, _Sun et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.03818)\]\[[code](https://github.com/SunzeY/AlphaCLIP)\]\[[Bootstrap3D](https://github.com/SunzeY/Bootstrap3D)\]
- MMVP: **Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs**, _Tong et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.06209)\]\[[code](https://github.com/tsb0601/MMVP)\]
- **MobileCLIP: Fast Image-Text Models through Multi-Modal Reinforced Training**, _Vasu et al._, CVPR 20224. \[[paper](https://arxiv.org/abs/2311.17049)\]\[[code](https://github.com/apple/ml-mobileclip)\]
- **Long-CLIP: Unlocking the Long-Text Capability of CLIP**, _Zhang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.15378)\]\[[code](https://github.com/beichenzbc/Long-CLIP)\]

### 4. Diffusion Model

- **Tutorial on Diffusion Models for Imaging and Vision**, _Stanley H. Chan_, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.18103)\]
- **Denoising Diffusion Probabilistic Models**, _Ho et al._, NeurIPS 2020. \[[paper](https://arxiv.org/abs/2006.11239)\]\[[code](https://github.com/hojonathanho/diffusion)\]\[[Pytorch Implementation](https://github.com/lucidrains/denoising-diffusion-pytorch)\]\[[RDDM](https://github.com/nachifur/RDDM)\]
- **Improved Denoising Diffusion Probabilistic Models**, _Nichol and Dhariwal_, ICML 2021. \[[paper](https://arxiv.org/abs/2102.09672)\]\[[code](https://github.com/openai/improved-diffusion)\]
- **Diffusion Models Beat GANs on Image Synthesis**, _Dhariwal and Nichol_, NeurIPS 2021. \[[paper](https://arxiv.org/abs/2105.05233)\]\[[code](https://github.com/openai/guided-diffusion)\]
- **Classifier-Free Diffusion Guidance**, _Ho and Salimans_, NeurIPS 2021. \[[paper](https://arxiv.org/abs/2207.12598)\]\[[code](https://github.com/lucidrains/classifier-free-guidance-pytorch)\]
- **GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models**, _Nichol et al._, arxiv 2021. \[[paper](https://arxiv.org/abs/2112.10741)\]\[[code](https://github.com/openai/glide-text2im)\]
- DALL-E2: **Hierarchical Text-Conditional Image Generation with CLIP Latents**, _Ramesh et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2204.06125)\]\[[code](https://github.com/lucidrains/DALLE2-pytorch)\]\[[dalle-mini](https://github.com/borisdayma/dalle-mini)\]
- Stable-Diffusion: **High-Resolution Image Synthesis with Latent Diffusion Models**, _Rombach et al._, CVPR 2022. \[[paper](https://arxiv.org/abs/2112.10752)\]\[[code](https://github.com/CompVis/latent-diffusion)\]\[[CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)\]\[[Stability-AI/stablediffusion](https://github.com/Stability-AI/stablediffusion)\]\[[ml-stable-diffusion](https://github.com/apple/ml-stable-diffusion)\]
- **SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis**, _Podell et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2307.01952)\]\[[code](https://github.com/Stability-AI/generative-models)\]\[[SDXL-Lightning](https://huggingface.co/ByteDance/SDXL-Lightning)\]
- **Introducing Stable Cascade**, _Stability AI_, 2024. \[[link](https://stability.ai/news/introducing-stable-cascade)\]\[[code](https://github.com/Stability-AI/StableCascade)\]\[[model](https://huggingface.co/stabilityai/stable-cascade)\]
- **SDXL-Turbo: Adversarial Diffusion Distillation**, _Sauer et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.17042)\]\[[code](https://github.com/Stability-AI/generative-models)\]
- LCM: **Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference**, _Luo et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.04378)\]\[[code](https://github.com/luosiallen/latent-consistency-model)\]\[[Hyper-SD](https://huggingface.co/ByteDance/Hyper-SD)\]\[[DMD2](https://github.com/tianweiy/DMD2)\]\[[ddim](https://github.com/ermongroup/ddim)\]
- **LCM-LoRA: A Universal Stable-Diffusion Acceleration Module**, _Luo et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.05556)\]\[[code](https://github.com/luosiallen/latent-consistency-model)\]\[[diffusion-forcing](https://github.com/buoyancy99/diffusion-forcing)\]
- Stable Diffusion 3: **Scaling Rectified Flow Transformers for High-Resolution Image Synthesis**, _Esser et al._, ICML 2024 Best Paper. \[[paper](https://arxiv.org/abs/2403.03206)\]\[[model](https://huggingface.co/stabilityai/stable-diffusion-3-medium)\]\[[mmdit](https://github.com/lucidrains/mmdit)\]
- SD3-Turbo: **Fast High-Resolution Image Synthesis with Latent Adversarial Diffusion Distillation**, _Sauer et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.12015)\]
- **StreamDiffusion: A Pipeline-level Solution for Real-time Interactive Generation**, _Kodaira et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.12491)\]\[[code](https://github.com/cumulo-autumn/StreamDiffusion)\]
- **DiffuseKronA: A Parameter Efficient Fine-tuning Method for Personalized Diffusion Models**, _Marjit et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.17412)\]\[[code](https://github.com/IBM/DiffuseKronA)\]
- **Video Diffusion Models**, _Ho et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2204.03458)\]\[[code](https://github.com/lucidrains/video-diffusion-pytorch)\]
- **Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets**, _Blattmann et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.15127)\]\[[code](https://github.com/Stability-AI/generative-models)\]\[[Stable Video 4D](https://huggingface.co/stabilityai/sv4d)\]\[[VideoCrafter](https://github.com/AILab-CVC/VideoCrafter)\]\[[Video-Infinity](https://github.com/Yuanshi9815/Video-Infinity)\]
- **Consistency Models**, _Song et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2303.01469)\]\[[code](https://github.com/openai/consistency_models)\]\[[Consistency Decoder](https://github.com/openai/consistencydecoder)\]
- **A Survey on Video Diffusion Models**, _Xing et al._, srxiv 2023. \[[paper](https://arxiv.org/abs/2310.10647)\]\[[code](https://github.com/ChenHsing/Awesome-Video-Diffusion-Models)\]
- **Diffusion Models: A Comprehensive Survey of Methods and Applications**, _Yang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2209.00796)\]\[[code](https://github.com/YangLing0818/Diffusion-Models-Papers-Survey-Taxonomy)\]
- **Language Model Beats Diffusion -- Tokenizer is Key to Visual Generation**, _Yu et al._, ICLR 2024. \[[paper](https://arxiv.org/abs/2310.05737)\]\[[magvit2-pytorch](https://github.com/lucidrains/magvit2-pytorch)\]\[[LlamaGen](https://github.com/FoundationVision/LlamaGen)\]
- **The Chosen One: Consistent Characters in Text-to-Image Diffusion Models**, _Avrahami et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.10093)\]\[[code](https://github.com/ZichengDuan/TheChosenOne)\]
- U-ViT: **All are Worth Words: A ViT Backbone for Diffusion Models**, _Bao et al._, CVPR 2023. \[[paper](https://arxiv.org/abs/2209.12152)\]\[[code](https://github.com/baofff/U-ViT)\]
- **UniDiffuser: One Transformer Fits All Distributions in Multi-Modal Diffusion**, _Bao et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2303.06555)\]\[[code](https://github.com/thu-ml/unidiffuser)\]
- **Matryoshka Diffusion Models**, _Gu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.15111)\]\[[code](https://github.com/apple/ml-mdm)\]
- SEDD: **Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution**, _Lou et al._, ICML 2024 Best Paper. \[[paper](https://arxiv.org/abs/2310.16834)\]\[[code](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion)\]
- l-DAE: **Deconstructing Denoising Diffusion Models for Self-Supervised Learning**, _Chen et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.14404)\]
- DiT: **Scalable Diffusion Models with Transformers**, _Peebles et al._, ICCV 2023 Oral. \[[paper](https://arxiv.org/abs/2212.09748)\]\[[code](https://github.com/facebookresearch/DiT)\]\[[OpenDiT](https://github.com/NUS-HPC-AI-Lab/OpenDiT)\]\[[MDT](https://github.com/sail-sg/MDT)\]\[[PipeFusion](https://github.com/PipeFusion/PipeFusion)\]
- **SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers**, _Ma et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.08740)\]\[[code](https://github.com/willisma/SiT)\]
- **Hyper-SD: Trajectory Segmented Consistency Model for Efficient Image Synthesis**, _Ren et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2404.13686)\]\[[model](https://huggingface.co/ByteDance/Hyper-SD)\]
- **Inf-DiT: Upsampling Any-Resolution Image with Memory-Efficient Diffusion Transformer**, _Yang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2405.04312)\]\[[code](https://github.com/THUDM/Inf-DiT)\]
- **Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion**, _Chen et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2407.01392)\]\[[code](https://github.com/buoyancy99/diffusion-forcing)\]
- **Stretching Each Dollar: Diffusion Training from Scratch on a Micro-Budget**, _Sehwag et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2407.15811)\]\[[code](https://github.com/SonyResearch/micro_diffusion)\]

- **Github Repositories**
- \[[Awesome-Diffusion-Models](https://github.com/diff-usion/Awesome-Diffusion-Models)\]\[[Awesome-Video-Diffusion](https://github.com/showlab/Awesome-Video-Diffusion)\]
- \[[stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)\]\[[stable-diffusion-webui-colab](https://github.com/camenduru/stable-diffusion-webui-colab)\]\[[sd-webui-controlnet](https://github.com/Mikubill/sd-webui-controlnet)\]\[[stable-diffusion-webui-forge](https://github.com/lllyasviel/stable-diffusion-webui-forge)\]\[[automatic](https://github.com/vladmandic/automatic)\]
- \[[Fooocus](https://github.com/lllyasviel/Fooocus)\]\[[Omost](https://github.com/lllyasviel/Omost)\]
- \[[ComfyUI](https://github.com/comfyanonymous/ComfyUI)\]\[[streamlit](https://github.com/streamlit/streamlit)\]\[[gradio](https://github.com/gradio-app/gradio)\]\[[ComfyUI-Workflows-ZHO](https://github.com/ZHO-ZHO-ZHO/ComfyUI-Workflows-ZHO)\]
- \[[diffusers](https://github.com/huggingface/diffusers)\]\[[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)\]

### 5. Multimodal LLM

- LLaVA: **Visual Instruction Tuning**, _Liu et al._, NeurIPS 2023 Oral. \[[paper](https://arxiv.org/abs/2304.08485)\]\[[code](https://github.com/haotian-liu/LLaVA)\]\[[vip-llava](https://github.com/mu-cai/vip-llava)\]\[[LLaVA-pp](https://github.com/mbzuai-oryx/LLaVA-pp)\]\[[TinyLLaVA_Factory](https://github.com/TinyLLaVA/TinyLLaVA_Factory)\]\[[LLaVA-RLHF](https://github.com/llava-rlhf/LLaVA-RLHF)\]
- LLaVA-1.5: **Improved Baselines with Visual Instruction Tuning**, _Liu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.03744)\]\[[code](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md)\]
- **LLaVA-NeXT-Interleave: Tackling Multi-image, Video, and 3D in Large Multimodal Models**, _Li et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2407.07895)\]\[[code](https://github.com/LLaVA-VL/LLaVA-NeXT)\]\[[MG-LLaVA](https://github.com/PhoenixZ810/MG-LLaVA)\]\[[LongVA](https://github.com/EvolvingLMMs-Lab/LongVA)\]
- **LLaVA-OneVision: Easy Visual Task Transfer**, _Li et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2408.03326)\]\[[code](https://github.com/LLaVA-VL/LLaVA-NeXT/tree/main/scripts/train)\]
- **LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day**, _Li et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2306.00890)\]\[[code](https://github.com/microsoft/LLaVA-Med)\]
- **Video-LLaVA: Learning United Visual Representation by Alignment Before Projection**, _Lin et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.10122)\]\[[code](https://github.com/PKU-YuanGroup/Video-LLaVA)\]
- **MoE-LLaVA: Mixture of Experts for Large Vision-Language Models**, _Lin et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.15947)\]\[[code](https://github.com/PKU-YuanGroup/MoE-LLaVA)\]
- **MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models**, _Zhu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2304.10592)\]\[[code](https://github.com/Vision-CAIR/MiniGPT-4)\]\[[MiniGPT-4-ZH](https://github.com/RiseInRose/MiniGPT-4-ZH)\]
- **MiniGPT-v2: large language model as a unified interface for vision-language multi-task learning**, _Chen et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.09478)\]\[[code](https://github.com/Vision-CAIR/MiniGPT-4)\]
- **MiniGPT4-Video: Advancing Multimodal LLMs for Video Understanding with Interleaved Visual-Textual Tokens**, _Ataallah et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2404.03413)\]\[[code](https://github.com/Vision-CAIR/MiniGPT4-video)\]
- **MiniGPT-5: Interleaved Vision-and-Language Generation via Generative Vokens**, _Zheng et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.02239)\]\[[code](https://github.com/eric-ai-lab/MiniGPT-5)\]

- **Flamingo: a Visual Language Model for Few-Shot Learning**, _Alayrac et al._, NeurIPS 2022. \[[paper](https://arxiv.org/abs/2204.14198)\]\[[open-flamingo](https://github.com/mlfoundations/open_flamingo)\]\[[flamingo-pytorch](https://github.com/lucidrains/flamingo-pytorch)\]
- **Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding**, _Zhang et al._, EMNLP 2023. \[[paper](https://arxiv.org/abs/2306.02858)\]\[[code](https://github.com/DAMO-NLP-SG/Video-LLaMA)\]\[[VideoLLaMA2](https://github.com/DAMO-NLP-SG/VideoLLaMA2)\]\[[VideoLLM-online](https://github.com/showlab/VideoLLM-online)\]
- **BuboGPT: Enabling Visual Grounding in Multi-Modal LLMs**, _Zhao et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2307.08581)\]\[[code](https://github.com/magic-research/bubogpt)\]\[[AnyGPT](https://github.com/OpenMOSS/AnyGPT)\]
- **Emu: Generative Pretraining in Multimodality**, _Sun et al._, ICLR 2024. \[[paper](https://arxiv.org/abs/2307.05222)\]\[[code](https://github.com/baaivision/Emu)\]
- EVE: **Unveiling Encoder-Free Vision-Language Models**, _Diao et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2406.11832)\]\[[code](https://github.com/baaivision/EVE)\]
- **CogVLM: Visual Expert for Pretrained Language Models**, _Wang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.03079)\]\[[code](https://github.com/THUDM/CogVLM)\]\[[CogVLM2](https://github.com/THUDM/CogVLM2)\]\[[VisualGLM-6B](https://github.com/THUDM/VisualGLM-6B)\]\[[CogCoM](https://github.com/THUDM/CogCoM)\]
- **DreamLLM: Synergistic Multimodal Comprehension and Creation**, _Dong et al._, ICLR 2024 Spotlight. \[[paper](https://arxiv.org/abs/2309.11499)\]\[[code](https://github.com/RunpeiDong/DreamLLM)\]\[[dreambench_plus](https://github.com/yuangpeng/dreambench_plus)\]
- **Meta-Transformer: A Unified Framework for Multimodal Learning**, _Zhang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2307.10802)\]\[[code](https://github.com/invictus717/MetaTransformer)\]
- **NExT-GPT: Any-to-Any Multimodal LLM**, _Wu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.05519)\]\[[code](https://github.com/NExT-GPT/NExT-GPT)\]
- **Visual ChatGPT: Talking, Drawing and Editing with Visual Foundation Models**, _Wu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2303.04671)\]\[[code](https://github.com/moymix/TaskMatrix)\]
- SoM: **Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V**, _Yang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.11441)\]\[[code](https://github.com/microsoft/SoM)\]
- **Ferret: Refer and Ground Anything Anywhere at Any Granularity**, _You et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.07704)\]\[[code](https://github.com/apple/ml-ferret)\]\[[Ferret-UI](https://arxiv.org/abs/2404.05719)\]
- **4M-21: An Any-to-Any Vision Model for Tens of Tasks and Modalities**, _Bachmann et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2406.09406)\]\[[code](https://github.com/apple/ml-4m)\]
- **Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond**, _Bai et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2308.12966)\]\[[code](https://github.com/QwenLM/Qwen-VL)\]
- **InternLM-XComposer: A Vision-Language Large Model for Advanced Text-image Comprehension and Composition**, _Zhang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.15112)\]\[[code](https://github.com/InternLM/InternLM-XComposer)\]
- **InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks**, _Chen et al._, CVPR 2024 Oral. \[[paper](https://arxiv.org/abs/2312.14238)\]\[[code](https://github.com/OpenGVLab/InternVL)\]\[[InternVideo](https://github.com/OpenGVLab/InternVideo)\]\[[InternVid](https://github.com/OpenGVLab/InternVideo/tree/main/Data/InternVid)\]
- **DeepSeek-VL: Towards Real-World Vision-Language Understanding**, _Lu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.05525)\]\[[code](https://github.com/deepseek-ai/DeepSeek-VL)\]
- **ShareGPT4V: Improving Large Multi-Modal Models with Better Captions**, _Chen et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.12793)\]\[[code](https://github.com/ShareGPT4Omni/ShareGPT4V)\]
- **ShareGPT4Video: Improving Video Understanding and Generation with Better Captions**, _Chen et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2406.04325)\]\[[code](https://github.com/ShareGPT4Omni/ShareGPT4Video)\]
- **TinyGPT-V: Efficient Multimodal Large Language Model via Small Backbones**, _Yuan et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.16862)\]\[[code](https://github.com/DLYuanGod/TinyGPT-V)\]
- **Monkey: Image Resolution and Text Label Are Important Things for Large Multi-modal Models**, _Li et al._, CVPR 2024. \[[paper](https://arxiv.org/abs/2311.06607)\]\[[code](https://github.com/Yuliang-Liu/Monkey)\]
- **Vary: Scaling up the Vision Vocabulary for Large Vision-Language Models**, _Wei et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.06109)\]\[[code](https://github.com/Ucas-HaoranWei/Vary)\]
- Vary-toy: **Small Language Model Meets with Reinforced Vision Vocabulary**, _Wei et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.12503)\]\[[code](https://github.com/Ucas-HaoranWei/Vary-toy)\]
- **VILA: On Pre-training for Visual Language Models**, _Lin et al._, CVPR 2024. \[[paper](https://arxiv.org/abs/2312.07533)\]\[[code](https://github.com/NVlabs/VILA)\]
- LWM: **World Model on Million-Length Video And Language With RingAttention**, _Liu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.08268)\]\[[code](https://github.com/LargeWorldModel/LWM)\]
- **Chameleon: Mixed-Modal Early-Fusion Foundation Models**, _Chameleon Team_, arxiv 2024. \[[paper](https://arxiv.org/abs/2405.09818)\]\[[code](https://github.com/facebookresearch/chameleon)\]
- **Uni-MoE: Scaling Unified Multimodal LLMs with Mixture of Experts**, _Li et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2405.11273)\]\[[code](https://github.com/hitsz-tmg/umoe-scaling-unified-multimodal-llms)\]
- RL4VLM: **Fine-Tuning Large Vision-Language Models as Decision-Making Agents via Reinforcement Learning**, _Zhai et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2405.10292)\]\[[code](https://github.com/RL4VLM/RL4VLM)\]\[[RLHF-V](https://github.com/RLHF-V/RLHF-V)\]\[[RLAIF-V](https://github.com/RLHF-V/RLAIF-V)\]
- **OpenVLA: An Open-Source Vision-Language-Action Model**, _Kim et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2406.09246)\]\[[code](https://github.com/openvla/openvla)\]
- **Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis**, _Fu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2405.21075)\]\[[code](https://github.com/BradyFU/Video-MME)\]\[[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)\]\[[VLMEvalKit](https://github.com/open-compass/VLMEvalKit)\]\[[multimodal-needle-in-a-haystack](https://github.com/Wang-ML-Lab/multimodal-needle-in-a-haystack)\]\[[MM-NIAH](https://github.com/OpenGVLab/MM-NIAH)\]\[[VideoNIAH](https://github.com/joez17/VideoNIAH)\]\[[ChartMimic](https://github.com/ChartMimic/ChartMimic)\]\[[WildVision](https://arxiv.org/abs/2406.11069)\]
- **MM-Vet v2: A Challenging Benchmark to Evaluate Large Multimodal Models for Integrated Capabilities**, _Yu et al._, ICML 2024. \[[paper](https://arxiv.org/abs/2408.00765)\]\[[code](https://github.com/yuweihao/MM-Vet)\]\[[UniBench: Visual Reasoning Requires Rethinking Vision-Language Beyond Scaling](https://arxiv.org/abs/2408.04810)\]
- **Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs**, _Tong et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2406.16860)\]\[[code](https://github.com/cambrian-mllm/cambrian)\]
- **video-SALMONN: Speech-Enhanced Audio-Visual Large Language Models**, _Sun et al._, ICML 2024. \[[paper](https://arxiv.org/abs/2406.15704)\]\[[code](https://github.com/bytedance/SALMONN)\]
- **ANOLE: An Open, Autoregressive, Native Large Multimodal Models for Interleaved Image-Text Generation**, _Chern et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2407.06135)\]\[[code](https://github.com/GAIR-NLP/anole)\]
- **PaliGemma: A versatile 3B VLM for transfer**, _Beyer et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2407.07726)\]\[[code](https://github.com/google-research/big_vision/tree/main/big_vision/configs/proj/paligemma)\]\[[pytorch-paligemma](https://github.com/hkproj/pytorch-paligemma)\]
- **MiniCPM-V: A GPT-4V Level MLLM on Your Phone**, _Yao et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2408.01800)\]\[[code](https://github.com/OpenBMB/MiniCPM-V)\]\[[RLHF-V](https://github.com/RLHF-V/RLHF-V)\]\[[RLAIF-V](https://github.com/RLHF-V/RLAIF-V)\]
- **VITA: Towards Open-Source Interactive Omni Multimodal LLM*, _Fu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2408.05211)\]\[[code](https://github.com/VITA-MLLM)\]

- \[[MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V)\]\[[moondream](https://github.com/vikhyat/moondream)\]\[[MobileVLM](https://github.com/Meituan-AutoML/MobileVLM)\]\[[OmniFusion](https://github.com/AIRI-Institute/OmniFusion)\]\[[Bunny](https://github.com/BAAI-DCAI/Bunny)\]\[[MiCo](https://github.com/invictus717/MiCo)\]\[[Vitron](https://github.com/SkyworkAI/Vitron)\]\[[mPLUG-Owl](https://github.com/X-PLUG/mPLUG-Owl)\]
- \[[datacomp](https://github.com/mlfoundations/datacomp)\]\[[MMDU](https://github.com/Liuziyu77/MMDU)\]\[[MINT-1T](https://github.com/mlfoundations/MINT-1T)\]\[[OpenVid-1M](https://github.com/NJU-PCALab/OpenVid-1M)\]
- \[[mllm](https://github.com/UbiquitousLearning/mllm)\]\[[lmms-finetune](https://github.com/zjysteven/lmms-finetune)\]

### 6. Text2Image

- DALL-E: **Zero-Shot Text-to-Image Generation**, _Ramesh et al._, arxiv 2021. \[[paper](https://arxiv.org/abs/2102.12092)\]\[[code](https://github.com/openai/DALL-E)\]
- DALL-E3: **Improving Image Generation with Better Captions**, _Betker et al._, OpenAI 2023. \[[paper](https://cdn.openai.com/papers/dall-e-3.pdf)\]\[[code](https://github.com/openai/consistencydecoder)\]\[[blog](https://openai.com/dall-e-3)\]\[[Glyph-ByT5](https://github.com/AIGText/Glyph-ByT5)\]
- ControlNet: **Adding Conditional Control to Text-to-Image Diffusion Models**, _Zhang et al._, ICCV 2023 Marr Prize. \[[paper](https://arxiv.org/abs/2302.05543)\]\[[code](https://github.com/lllyasviel/ControlNet)\]\[[ControlNet_Plus_Plus](https://github.com/liming-ai/ControlNet_Plus_Plus)\]\[[ControlNeXt](https://github.com/dvlab-research/ControlNeXt)\]
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
- **IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models**, _Ye et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2308.06721)\]\[[code](https://github.com/tencent-ailab/IP-Adapter)\]\[[ID-Animator](https://github.com/ID-Animator/ID-Animator)\]\[[InstantID](https://github.com/InstantID/InstantID)\]
- **Controllable Generation with Text-to-Image Diffusion Models: A Survey**, _Cao et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.04279)\]\[[code](https://github.com/PRIV-Creation/Awesome-Controllable-T2I-Diffusion-Models)\]
- **StoryDiffusion: Consistent Self-Attention for Long-Range Image and Video Generation**, _Zhou et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2405.01434)\]\[[code](https://github.com/HVision-NKU/StoryDiffusion)\]\[[AutoStudio](https://github.com/donahowe/AutoStudio)\]
- **Hunyuan-DiT: A Powerful Multi-Resolution Diffusion Transformer with Fine-Grained Chinese Understanding**, _Li et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2405.08748)\]\[[code](https://github.com/Tencent/HunyuanDiT)\]\[[xDiT](https://github.com/xdit-project/xDiT)\]
- \[[Kolors](https://github.com/Kwai-Kolors/Kolors)\]\[[EVLM: An Efficient Vision-Language Model for Visual Understanding](https://arxiv.org/abs/2407.14177)\]

- \[[flux](https://github.com/black-forest-labs/flux)\]\[[x-flux](https://github.com/XLabs-AI/x-flux)\]

### 7. Text2Video

- **Animate Anyone: Consistent and Controllable Image-to-Video Synthesis for Character Animation**, _Hu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.17117)\]\[[code](https://github.com/HumanAIGC/AnimateAnyone)\]\[[Open-AnimateAnyone](https://github.com/guoqincode/Open-AnimateAnyone)\]\[[Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone)\]\[[AnimateAnyone](https://github.com/novitalabs/AnimateAnyone)\]\[[UniAnimate](https://github.com/ali-vilab/UniAnimate)\]
- **EMO: Emote Portrait Alive - Generating Expressive Portrait Videos with Audio2Video Diffusion Model under Weak Conditions**, _Tian et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.17485)\]\[[code](https://github.com/HumanAIGC/EMO)\]\[[V-Express](https://github.com/tencent-ailab/V-Express)\]
- **AniPortrait: Audio-Driven Synthesis of Photorealistic Portrait Animation**, _Wei wt al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.17694)\]\[[code](https://github.com/Zejun-Yang/AniPortrait)\]
- **DreaMoving: A Human Video Generation Framework based on Diffusion Models**, _Feng et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.05107)\]\[[code](https://github.com/dreamoving/dreamoving-project)\]
- **MagicAnimate:Temporally Consistent Human Image Animation using Diffusion Model**, _Xu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.16498)\]\[[code](https://github.com/magic-research/magic-animate)\]\[[champ](https://github.com/fudan-generative-vision/champ)\]\[[MegActor](https://github.com/megvii-research/MegActor)\]
- **DynamiCrafter: Animating Open-domain Images with Video Diffusion Priors**, _Xing et al._, ECCV 2024. \[[paper](https://arxiv.org/abs/2310.12190)\]\[[code](https://github.com/Doubiiu/DynamiCrafter)\]
- **LivePortrait: Efficient Portrait Animation with Stitching and Retargeting Control**, _Guo et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2407.03168)\]\[[code](https://github.com/KwaiVGI/LivePortrait)\]\[[FasterLivePortrait](https://github.com/warmshao/FasterLivePortrait)\]\[[FollowYourEmoji](https://github.com/mayuelala/FollowYourEmoji)\]
- **FlowVid: Taming Imperfect Optical Flows for Consistent Video-to-Video Synthesis**, _Liang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.17681)\]\[[code](https://github.com/Jeff-LiangF/FlowVid)\]

- \[[Awesome-Video-Diffusion](https://github.com/showlab/Awesome-Video-Diffusion)\]
- **Video Diffusion Models**, _Ho et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2204.03458)\]\[[video-diffusion-pytorch](https://github.com/lucidrains/video-diffusion-pytorch)\]
- **Make-A-Video: Text-to-Video Generation without Text-Video Data**, _Singer et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2209.14792)\]\[[make-a-video-pytorch](https://github.com/lucidrains/make-a-video-pytorch)\]
- **Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation**, _Wu et al._, ICCV 2023. \[[paper](https://arxiv.org/abs/2212.11565)\]\[[code](https://github.com/showlab/Tune-A-Video)\]
- **Text2Video-Zero: Text-to-Image Diffusion Models are Zero-Shot Video Generators**, _Khachatryan et al._, ICCV 2023 Oral. \[[paper](https://arxiv.org/abs/2303.13439)\]\[[code](https://github.com/Picsart-AI-Research/Text2Video-Zero)\]\[[StreamingT2V](https://github.com/Picsart-AI-Research/StreamingT2V)\]
- **CogVideo: Large-scale Pretraining for Text-to-Video Generation via Transformers**, _Hong et al._, ICLR 2023. \[[paper](https://arxiv.org/abs/2205.15868)\]\[[code](https://github.com/THUDM/CogVideo)\]
- **CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer**, _Yang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2408.06072)\]\[[code](https://github.com/THUDM/CogVideo)\]

- **Follow Your Pose: Pose-Guided Text-to-Video Generation using Pose-Free Videos**, _Ma et al._, AAAI 2024. \[[paper](https://arxiv.org/abs/2304.01186)\]\[[code](https://github.com/mayuelala/FollowYourPose)\]\[[Follow-Your-Pose v2](https://arxiv.org/abs/2406.03035)\]\[[Follow-Your-Emoji](https://arxiv.org/abs/2406.01900)\]
- **Follow-Your-Click: Open-domain Regional Image Animation via Short Prompts**, _Ma et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.08268)\]\[[code](https://github.com/mayuelala/FollowYourClick)\]
- **AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning**, _Guo et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2307.04725)\]\[[code](https://github.com/guoyww/AnimateDiff)\]\[[AnimateDiff-Lightning](https://huggingface.co/ByteDance/AnimateDiff-Lightning)\]
- **StableVideo: Text-driven Consistency-aware Diffusion Video Editing**, _Chai et al._, ICCV 2023. \[[paper](https://arxiv.org/abs/2308.09592)\]\[[code](https://github.com/rese1f/StableVideo)\]
- **I2VGen-XL: High-Quality Image-to-Video Synthesis via Cascaded Diffusion Models**, _Zhang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.04145)\]\[[code](https://github.com/ali-vilab/VGen)\]
- TF-T2V: **A Recipe for Scaling up Text-to-Video Generation with Text-free Videos**, _Wang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.15770)\]\[[code](https://github.com/ali-vilab/VGen)\]
- **Lumiere: A Space-Time Diffusion Model for Video Generation**, _Bar-Tal et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.12945)\]\[[lumiere-pytorch](https://github.com/lucidrains/lumiere-pytorch)\]
- **Sora: Creating video from text**, _OpenAI_, 2024. \[[blog](https://openai.com/sora)\]\[[Generative Models for Image and Long Video Synthesis](https://digitalassets.lib.berkeley.edu/techreports/ucb/incoming/EECS-2023-100.pdf)\]\[[Generative Models of Images and Neural Networks](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2023/EECS-2023-108.pdf)\]\[[Open-Sora](https://github.com/hpcaitech/Open-Sora)\]\[[Open-Sora-Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan)\]\[[minisora](https://github.com/mini-sora/minisora)\]\[[SoraWebui](https://github.com/SoraWebui/SoraWebui)\]\[[MuseV](https://github.com/TMElyralab/MuseV)\]\[[PhysDreamer](https://github.com/a1600012888/PhysDreamer)\]\[[easyanimate](https://github.com/aigc-apps/easyanimate)\]
- **Sora: A Review on Background, Technology, Limitations, and Opportunities of Large Vision Models**, _Liu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.17177)\]\[[code](https://github.com/lichao-sun/SoraReview)\]
- **Mora: Enabling Generalist Video Generation via A Multi-Agent Framework**, _Yuan et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.13248)\]\[[code](https://github.com/lichao-sun/Mora)\]
- **Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution**, _Dehghani et al._, NeurIPS 2024. \[[paper](https://arxiv.org/abs/2307.06304)\]\[[unofficial code](https://github.com/kyegomez/NaViT)\]
- **VideoPoet: A Large Language Model for Zero-Shot Video Generation**, _Kondratyuk et al._, ICML 2024 Best Paper. \[[paper](https://arxiv.org/abs/2312.14125)\]
- **Latte: Latent Diffusion Transformer for Video Generation**, _Ma et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.03048)\]\[[code](https://github.com/Vchitect/Latte)\]\[[LaVIT](https://github.com/jy0205/LaVIT)\]\[[LaVie](https://github.com/Vchitect/LaVie)\]\[[VBench](https://github.com/Vchitect/VBench)\]
- **Snap Video: Scaled Spatiotemporal Transformers for Text-to-Video Synthesis**, _Menapace et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.14797)\]\[[articulated-animation](https://github.com/snap-research/articulated-animation)\]

- \[[MoneyPrinterTurbo](https://github.com/harry0703/MoneyPrinterTurbo)\]\[[videos](https://github.com/3b1b/videos)\]

### 8. Survey for Multimodal

- **A Survey on Multimodal Large Language Models**, _Yin et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2306.13549)\]\[[code](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models)\]
- **Multimodal Foundation Models: From Specialists to General-Purpose Assistants**, _Li et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.10020)\]\[[cvinw_readings](https://github.com/computer-vision-in-the-wild/cvinw_readings)\]
- **From GPT-4 to Gemini and Beyond: Assessing the Landscape of MLLMs on Generalizability, Trustworthiness and Causality through Four Modalities**, _Lu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.15071)\]\[[Leaderboards](https://openlamm.github.io/Leaderboards)\]
- **Efficient Multimodal Large Language Models: A Survey**, _Jin et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2405.10739)\]\[[code](https://github.com/lijiannuist/Efficient-Multimodal-LLMs-Survey)\]
- **An Introduction to Vision-Language Modeling**, _Bordes et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2405.17247)\]

### 9. Other

- **Fuyu-8B: A Multimodal Architecture for AI Agents** _Bavishi et al._, Adept blog 2023. \[[blog](https://www.adept.ai/blog/fuyu-8b)\]\[[model](https://huggingface.co/adept/fuyu-8b)\]
- **Otter: A Multi-Modal Model with In-Context Instruction Tuning**, _Li et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.03726)\]\[[code](https://github.com/Luodian/Otter)\]
- **OtterHD: A High-Resolution Multi-modality Model**, _Li et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.04219)\]\[[code](https://github.com/Luodian/Otter)\]\[[model](https://huggingface.co/Otter-AI/OtterHD-8B)\]
- CM3leon: **Scaling Autoregressive Multi-Modal Models_Pretraining and Instruction Tuning**, _Yu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.02591)\]\[[Unofficial Implementation](https://github.com/kyegomez/CM3Leon)\]
- **MM-Interleaved: Interleaved Image-Text Generative Modeling via Multi-modal Feature Synchronizer**, _Tian et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.10208)\]\[[code](https://github.com/OpenGVLab/MM-Interleaved)\]
- **CogCoM: Train Large Vision-Language Models Diving into Details through Chain of Manipulations**, _Qi et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.04236)\]\[[code](https://github.com/THUDM/CogCoM)\]
- **SPHINX-X: Scaling Data and Parameters for a Family of Multi-modal Large Language Models**, _Gao et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.05935)\]\[[code](https://github.com/Alpha-VLLM/LLaMA2-Accessory)\]
- **Lumina-T2X: Transforming Text into Any Modality, Resolution, and Duration via Flow-based Large Diffusion Transformers**, _Gao et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2405.05945)\]\[[code](https://github.com/Alpha-VLLM/Lumina-T2X)\]
- **Lumina-mGPT: Illuminate Flexible Photorealistic Text-to-Image Generation with Multimodal Generative Pretraining**, _Liu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2408.02657)\]\[[code](https://github.com/Alpha-VLLM/Lumina-mGPT)\]
- LWM: **World Model on Million-Length Video And Language With RingAttention**, _Liu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.08268)\]\[[code](https://github.com/LargeWorldModel/LWM)\]
- **Chameleon: Mixed-Modal Early-Fusion Foundation Models**, _Chameleon Team_, arxiv 2024. \[[paper](https://arxiv.org/abs/2405.09818)\]\[[code](https://github.com/facebookresearch/chameleon)\]
- **SEED-X: Multimodal Models with Unified Multi-granularity Comprehension and Generation*, _Ge et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2404.14396)\]\[[code](https://github.com/AILab-CVC/SEED-X)\]\[[SEED](https://github.com/AILab-CVC/SEED)\]\[[SEED-Story](https://github.com/TencentARC/SEED-Story)\]

---

## Reinforcement Learning

### 1.Basic for RL

- **Deep Reinforcement Learning: Pong from Pixels**, _Andrej Karpathy_, 2016. \[[blog](https://karpathy.github.io/2016/05/31/rl/)\]\[[reinforcement-learning-an-introduction](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)\]\[[easy-rl](https://github.com/datawhalechina/easy-rl)\]\[[deep-rl-course](https://huggingface.co/learn/deep-rl-course/)\]
- DQN: **Playing Atari with Deep Reinforcement Learning**, _Mnih et al._, arxiv 2013. \[[paper](https://arxiv.org/abs/1312.5602)\]\[[code](https://github.com/higgsfield/RL-Adventure/blob/master/1.dqn.ipynb)\]
- DQNNaturePaper: **Human-level control through deep reinforcement learning**, _Mnih et al._, Nature 2015. \[[paper](https://www.nature.com/articles/nature14236)\]\[[DQN-tensorflow](https://github.com/devsisters/DQN-tensorflow)\]\[[DQN_pytorch](https://github.com/dxyang/DQN_pytorch)\]
- DDQN: **Deep Reinforcement Learning with Double Q-learning**, _Hasselt et al._, AAAI 2016. \[[paper](https://arxiv.org/abs/1509.06461)\]\[[RL-Adventure](https://github.com/higgsfield/RL-Adventure)\]\[[deep-q-learning](https://github.com/keon/deep-q-learning)\]\[[Deep-RL-Keras](https://github.com/germain-hug/Deep-RL-Keras)\]
- **Rainbow: Combining Improvements in Deep Reinforcement Learning**, _Hesssel et al._, AAAI 2018. \[[paper](https://arxiv.org/abs/1710.02298)\]\[[Rainbow](https://github.com/Kaixhin/Rainbow)\]
- DDPG: **Continuous control with deep reinforcement learning**, _Lillicrap et al._, ICLR 2016. \[[paper](https://arxiv.org/abs/1509.02971)\]\[[pytorch-ddpg](https://github.com/ghliu/pytorch-ddpg)\]

- PPO: **Proximal Policy Optimization Algorithms**, _Schulman et al._, arxiv 2017. \[[paper](https://arxiv.org/abs/1707.06347)\]\[[code](https://github.com/openai/baselines)\]\[[trl ppo_trainer](https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py)\]\[[PPO-PyTorch](https://github.com/nikhilbarhate99/PPO-PyTorch)\]\[[implementation-matters](https://github.com/MadryLab/implementation-matters)\]\[[PPOxFamily](https://github.com/opendilab/PPOxFamily)\]

- **Diffusion Models for Reinforcement Learning: A Survey**, _Zhu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.01223)\]\[[code](https://github.com/apexrl/Diff4RLSurvey)\]\[[diffusion_policy](https://github.com/real-stanford/diffusion_policy)\]
- **The Definitive Guide to Policy Gradients in Deep Reinforcement Learning: Theory, Algorithms and Implementations**, _Matthias Lehmann_, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.13662)\]\[[code](https://github.com/Matt00n/PolicyGradientsJax)\]

- \[[tianshou](https://github.com/thu-ml/tianshou)\]\[[rlkit](https://github.com/rail-berkeley/rlkit)\]\[[pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail)\]

### 2. LLM for decision making 

- **Decision Transformer_Reinforcement Learning via Sequence Modeling**, _Chen et al._, NeurIPS 2021. \[[paper](https://arxiv.org/abs/2106.01345)\]\[[code](https://github.com/kzl/decision-transformer)\]
- Trajectory Transformer: **Offline Reinforcement Learning as One Big Sequence Modeling Problem**, _Janner et al._, NeurIPS 2021. \[[paper](https://arxiv.org/abs/2106.02039)\]\[[code](https://github.com/JannerM/trajectory-transformer)\]
- **Guiding Pretraining in Reinforcement Learning with Large Language Models**, _Du et al._, ICML 2023. \[[paper](https://arxiv.org/abs/2302.06692)\]\[[code](https://github.com/yuqingd/ellm)\]
- **Introspective Tips: Large Language Model for In-Context Decision Making**, _Chen et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.11598)\]
- **Q-Transformer: Scalable Offline Reinforcement Learning via Autoregressive Q-Functions**, _Chebotar et al._, CoRL 2023. \[[paper](https://arxiv.org/abs/2309.10150)\]\[[Unofficial Implementation](https://github.com/lucidrains/q-transformer)\]
- **Survey on Large Language Model-Enhanced Reinforcement Learning: Concept, Taxonomy, and Methods**, _Cao et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2404.00282)\]

---

## GNN
 
- \[[GNNPapers](https://github.com/thunlp/GNNPapers)\]\[[dgl](https://github.com/dmlc/dgl)\]\[[Awesome_Graph_Foundation_Models](https://github.com/CurryTang/Awesome_Graph_Foundation_Models)\]
- **A Gentle Introduction to Graph Neural Networks**, _Sanchez-Lengeling et al._, Distill 2021. \[[paper](https://distill.pub/2021/gnn-intro/)\]
- **CS224W: Machine Learning with Graphs**, Stanford. \[[link](http://web.stanford.edu/class/cs224w)\]

- GCN: **Semi-Supervised Classification with Graph Convolutional Networks**, _Kipf and Welling_, ICLR 2017. \[[paper](https://arxiv.org/abs/1609.02907)\]\[[code](https://github.com/tkipf/gcn)\]\[[pygcn](https://github.com/tkipf/pygcn)\]
- GAE: **Variational Graph Auto-Encoders**, _Kipf and Welling_, arxiv 2016. \[[paper](https://arxiv.org/abs/1611.07308)\]\[[code](https://github.com/tkipf/gae)\]\[[gae-pytorch](https://github.com/zfjsail/gae-pytorch)\]
- GAT: **Graph Attention Networks**, _Veličković et al._, ICLR 2018. \[[paper](https://arxiv.org/abs/1710.10903)\]\[[code](https://github.com/PetarV-/GAT)\]\[[pyGAT](https://github.com/Diego999/pyGAT)\]\[[pytorch-GAT](https://github.com/gordicaleksa/pytorch-GAT)\]
- GIN: **How Powerful are Graph Neural Networks?**, _Xu et al._, ICLR 2019. \[[paper](https://arxiv.org/abs/1810.00826)\]\[[code](https://github.com/weihua916/powerful-gnns)\]

- Graphormer: **Do Transformers Really Perform Bad for Graph Representation**, _Ying et al._, NeurIPS 2021. \[[paper](https://arxiv.org/abs/2106.05234)\]\[[code](https://github.com/Microsoft/Graphormer)\]
- **GraphGPT: Graph Instruction Tuning for Large Language Models**, _Tang et al._, SIGIR 2024. \[[paper](https://arxiv.org/abs/2310.13023)\]\[[code](https://github.com/HKUDS/GraphGPT)\]\[[Graph-Bert](https://github.com/jwzhanggy/Graph-Bert)\]
- **OpenGraph: Towards Open Graph Foundation Models**, _Xia et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.01121)\]\[[code](https://github.com/HKUDS/OpenGraph)\]
- **A Survey of Large Language Models for Graphs**, _Ren et al._, KDD 2024. \[[paper](https://arxiv.org/abs/2405.08011)\]\[[code](https://github.com/HKUDS/Awesome-LLM4Graph-Papers)\]

- \[[pytorch_geometric](https://github.com/pyg-team/pytorch_geometric)\]\[[GNN-Recommender-Systems](https://github.com/tsinghua-fib-lab/GNN-Recommender-Systems)\]

### Survey for GNN

---

## Transformer Architecture

- **Attention is All you Need**, _Vaswani et al._, NIPS 2017. \[[paper](https://arxiv.org/abs/1706.03762)\]\[[code](https://github.com/jadore801120/attention-is-all-you-need-pytorch)\]\[[transformer-debugger](https://github.com/openai/transformer-debugger)\]\[[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)\]\[[The Random Transformer](https://osanseviero.github.io/hackerllama/blog/posts/random_transformer/)\]\[[The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)\]\[[Transformers-Tutorials](https://github.com/NielsRogge/Transformers-Tutorials)\]\[[x-transformers](https://github.com/lucidrains/x-transformers)\]
- RoPE: **RoFormer: Enhanced Transformer with Rotary Position Embedding**, _Su et al._, arxiv 2021. \[[paper](https://arxiv.org/abs/2104.09864)\]\[[code](https://github.com/ZhuiyiTechnology/roformer)\]\[[rotary-embedding-torch](https://github.com/lucidrains/rotary-embedding-torch)\]\[[rerope](https://github.com/bojone/rerope)\]\[[blog](https://kexue.fm/archives/9675)\]\[[positional_embedding](https://skylyj.github.io/positional_embedding/)\]\[[longformer](https://github.com/allenai/longformer)\]
- **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints**, _Ainslie et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.13245)\]\[[unofficial code](https://github.com/fkodom/grouped-query-attention-pytorch)\]\[[blog](https://kexue.fm/archives/10091)\]
- **RWKV: Reinventing RNNs for the Transformer Era**, _Peng et al._, EMNLP 2023. \[[paper](https://arxiv.org/abs/2305.13048)\]\[[code](https://github.com/BlinkDL/RWKV-LM)\]\[[ChatRWKV](https://github.com/BlinkDL/ChatRWKV)\]\[[rwkv.cpp](https://github.com/RWKV/rwkv.cpp)\]
- **Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence**, _Peng et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2404.05892)\]\[[code](https://github.com/RWKV/RWKV-LM)\]\[[Awesome-RWKV-in-Vision](https://github.com/Yaziwel/Awesome-RWKV-in-Vision)\]
- **Mamba: Linear-Time Sequence Modeling with Selective State Spaces**, _Gu and Dao_, COLM 2024. \[[paper](https://arxiv.org/abs/2312.00752)\]\[[code](https://github.com/state-spaces/mamba)\]\[[mamba-minimal](https://github.com/johnma2006/mamba-minimal)\]\[[Awesome-Mamba-Papers](https://github.com/yyyujintang/Awesome-Mamba-Papers)\]
- **Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models**, _De et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.19427)\]\[[recurrentgemma](https://github.com/google-deepmind/recurrentgemma)\]
- **Jamba: A Hybrid Transformer-Mamba Language Model**, _Lieber et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.19887)\]\[[model](https://huggingface.co/ai21labs/Jamba-v0.1)\]\[[Samba](https://github.com/microsoft/Samba)\]
- **Neural Network Diffusion**, _Wang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.13144)\]\[[code](https://github.com/NUS-HPC-AI-Lab/Neural-Network-Diffusion)\]\[[GPD](https://github.com/tsinghua-fib-lab/GPD)\]\[[tree-diffusion](https://github.com/revalo/tree-diffusion)\]
- **KAN: Kolmogorov-Arnold Networks**, _Liu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2404.19756)\]\[[code](https://github.com/KindXiaoming/pykan)\]\[[efficient-kan](https://github.com/Blealtan/efficient-kan)\]\[[kan-gpt](https://github.com/AdityaNG/kan-gpt)\]\[[Convolutional-KANs](https://github.com/AntonioTepsich/Convolutional-KANs)\]
- **xLSTM: Extended Long Short-Term Memory**, _Beck et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2405.04517)\]\[[code](https://github.com/NX-AI/xlstm)\]\[[vision-lstm](https://github.com/NX-AI/vision-lstm)\]\[[PyxLSTM](https://github.com/muditbhargava66/PyxLSTM)\]\[[xlstm-cuda](https://github.com/smvorwerk/xlstm-cuda)\]\[[Attention as an RNN](https://arxiv.org/abs/2405.13956)\]\[[ttt-lm-pytorch](https://github.com/test-time-training/ttt-lm-pytorch)\]


