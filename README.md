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
- \[[pytorch-seq2seq](https://github.com/IBM/pytorch-seq2seq)\]

### 3. Pretraining

- **Attention Is All You Need**, _Vaswani et al._, NIPS 2017. \[[paper](https://arxiv.org/abs/1706.03762)\]\[[code](https://github.com/tensorflow/tensor2tensor)\]
- GPT: **Improving language understanding by generative pre-training**, _Radford et al._, preprint 2018.  \[[paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)\]\[[code](https://github.com/openai/finetune-transformer-lm)\]
- GPT-2: **Language Models are Unsupervised Multitask Learners**, _Radford et al._, OpenAI blog 2019. \[[paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)\]\[[code](https://github.com/openai/gpt-2)\]
- GPT-3: **Language Models are Few-Shot Learners**, _Brown et al._, NeurIPS 2020. \[[paper](https://arxiv.org/abs/2005.14165)\]\[[code](https://github.com/openai/gpt-3)\]\[[nanoGPT](https://github.com/karpathy/nanoGPT)\]\[[gpt-fast](https://github.com/pytorch-labs/gpt-fast)\]
- InstructGPT: **Training language models to follow instructions with human feedback**, _Ouyang et al._, NeurIPS 2022. \[[paper](https://arxiv.org/abs/2203.02155)\]\[[MOSS-RLHF](https://github.com/OpenLMLab/MOSS-RLHF)\]
- **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**, _Devlin et al._, arxiv 2018. \[[paper](https://arxiv.org/abs/1810.04805)\]\[[code](https://github.com/google-research/bert)\]
- **RoBERTa: A Robustly Optimized BERT Pretraining Approach**, _Liu et al._, arxiv 2019. \[[paper](https://arxiv.org/abs/1907.11692)\]\[[code](https://github.com/facebookresearch/fairseq)\]
- **What Does BERT Look At_An Analysis of BERT's Attention**, _Clark et al._, arxiv 2019. \[[paper](https://arxiv.org/abs/1906.04341)\]\[[code](https://github.com/clarkkev/attention-analysis)\]
- **DeBERTa: Decoding-enhanced BERT with Disentangled Attention**, _He et al._, ICLR 2021. \[[paper](https://arxiv.org/abs/2006.03654)\]\[[code](https://github.com/microsoft/DeBERTa)\]
- **DistilBERT: a distilled version of BERT_smaller, faster, cheaper and lighter** _Sanh et al._, arxiv 2019. \[[paper](https://arxiv.org/abs/1910.01108)\]\[[code](https://github.com/huggingface/transformers)\]
- **BERT Rediscovers the Classical NLP Pipeline**, _Tenney et al._, arxiv 2019. \[[paper](https://arxiv.org/abs/1905.05950)\]\[[code](https://github.com/nyu-mll/jiant)\]
- **TinyStories: How Small Can Language Models Be and Still Speak Coherent English**, _Eldan and Li_, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.07759)\]\[[code]\]\[[phi-2](https://huggingface.co/microsoft/phi-2)\]
- \[[llm-course](https://github.com/mlabonne/llm-course)\]\[[intro-llm](https://intro-llm.github.io/)\]

#### 3.1 Large Language Model

- **A Survey of Large Language Models**, _Zhao etal._, arxiv 2023. \[[paper](https://arxiv.org/abs/2303.18223)\]\[[code](https://github.com/RUCAIBox/LLMSurvey)\]\[[LLMsPracticalGuide](https://github.com/Mooler0410/LLMsPracticalGuide)\]
- **Efficient Large Language Models: A Survey**, _Wan et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.03863)\]\[[code](https://github.com/AIoT-MLSys-Lab/Efficient-LLMs-Survey)\]
- **Challenges and Applications of Large Language Models**, _Kaddour et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2307.10169)\]
- **A Comprehensive Survey on Pretrained Foundation Models: A History from BERT to ChatGPT**, _Zhou et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2302.09419)\]
- **From Google Gemini to OpenAI Q* (Q-Star): A Survey of Reshaping the Generative Artificial Intelligence (AI) Research Landscape**, _Mclntosh et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.10868)\]
- **A Survey of Resource-efficient LLM and Multimodal Foundation Models**, _Xu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.08092)\]\[[code](https://github.com/UbiquitousLearning/Efficient_Foundation_Model_Survey)\]
- Anthropic: **Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback**, _Bai et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2204.05862)\]\[[code](https://github.com/anthropics/hh-rlhf)\]
- Anthropic: **Constitutional AI: Harmlessness from AI Feedback**, _Bai et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2212.08073)\]\[[code](https://github.com/anthropics/ConstitutionalHarmlessnessPaper)\]
- Anthropic: **Model Card and Evaluations for Claude Models**, Anthropic, 2023. \[[paper](https://www-files.anthropic.com/production/images/Model-Card-Claude-2.pdf)\]
- **BLOOM_A 176B-Parameter Open-Access Multilingual Language Model**, _BigScience Workshop_, arxiv 2022. \[[paper](https://arxiv.org/abs/2211.05100)\]\[[code](https://github.com/bigscience-workshop)\]\[[model](https://huggingface.co/bigscience)\]
- **OPT: Open Pre-trained Transformer Language Models**, _Zhang et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2205.01068)\]\[[code](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT)\]
- Chinchilla: **Training Compute-Optimal Large Language Models**, _Hoffmann et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2203.15556)\]
- Gopher: **Scaling Language Models: Methods, Analysis & Insights from Training Gopher**, _Rae et al._, arxiv 2021. \[[paper](https://arxiv.org/abs/2112.11446)\]
- Codex: **Evaluating Large Language Models Trained on Code**, _Chen et al._, arxiv 2021. \[[paper](https://arxiv.org/abs/2107.03374)\]\[[code](https://github.com/openai/human-eval)\]
- **Colossal-AI: A Unified Deep Learning System For Large-Scale Parallel Training**, _Li et al._, ICPP 2023. \[[paper](https://arxiv.org/abs/2110.14883)\]\[[code](https://github.com/hpcaitech/ColossalAI)\]
- **GPT-NeoX-20B: An Open-Source Autoregressive Language Model**, _Black et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2204.06745)\]\[[code](https://github.com/EleutherAI/gpt-neox)\]
- **Gemini: A Family of Highly Capable Multimodal Models**, _Gemini Team, Google_, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.11805)\]\[[Unofficial Implementation](https://github.com/kyegomez/Gemini)\]
- **GPT-4 Technical Report**, _OpenAI_, arxiv 2023. \[[paper](https://arxiv.org/abs/2303.08774)\]
- **GPT-4V(ision) System Card**, _OpenAI_, OpenAI blog 2023. \[[paper](https://openai.com/research/gpt-4v-system-card)\]
- **Sparks of Artificial General Intelligence_Early experiments with GPT-4**, _Bubeck et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2303.12712)\]
- **The Dawn of LMMs_Preliminary Explorations with GPT-4V(ision)**, _Yang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.17421)\]\[[code](https://github.com/guidance-ai/guidance)\]
- **LaMDA: Language Models for Dialog Applications**, _Thoppilan et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2201.08239)\]\[[LaMDA-rlhf-pytorch](https://github.com/conceptofmind/LaMDA-rlhf-pytorch)\]
- **Code Llama: Open Foundation Models for Code**, _Rozière et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2308.12950)\]\[[code](https://github.com/facebookresearch/codellama)\]
- **LLaMA: Open and Efficient Foundation Language Models**, _Touvron et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2302.13971)\]\[[code](https://github.com/facebookresearch/llama/tree/llama_v1)\]\[[llama.cpp](https://github.com/ggerganov/llama.cpp)\]\[[ollama](https://github.com/jmorganca/ollama)\]
- **Llama 2: Open Foundation and Fine-Tuned Chat Models**, _Touvron et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2307.09288)\]\[[code](https://github.com/facebookresearch/llama)\]\[[llama-recipes](https://github.com/facebookresearch/llama-recipes)\]\[[llama2.c](https://github.com/karpathy/llama2.c)\]
- **TinyLlama: An Open-Source Small Language Model**, _Zhang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.02385)\]\[[code](https://github.com/jzhang38/TinyLlama)\]\[[LiteLlama](https://huggingface.co/ahxt/LiteLlama-460M-1T)\]
- **Stanford Alpaca: An Instruction-following LLaMA Model**, _Taori et al._, Stanford blog 2023. \[[paper](https://crfm.stanford.edu/2023/03/13/alpaca.html)\]\[[code](https://github.com/tatsu-lab/stanford_alpaca)\]\[[Alpaca-Lora](https://github.com/tloen/alpaca-lora)\]
- **Mistral 7B**, _Jiang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.06825)\]\[[code](https://github.com/mistralai/mistral-src)\]\[[model](https://huggingface.co/mistralai)\]
- Minerva: **Solving Quantitative Reasoning Problems with Language Models**, _Lewkowycz et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2206.14858)\]
- **PaLM: Scaling Language Modeling with Pathways**, _Chowdhery et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2204.02311)\]\[[PaLM-pytorch](https://github.com/lucidrains/PaLM-pytorch)\]\[[PaLM-rlhf-pytorch](https://github.com/lucidrains/PaLM-rlhf-pytorch)\]\[[PaLM](https://github.com/conceptofmind/PaLM)\]
- **PaLM 2 Technical Report**, _Anil et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.10403)\]
- **PaLM-E: An Embodied Multimodal Language Model**, _Driess et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2303.03378)\]\[[code](https://github.com/kyegomez/PALM-E)\]
- T5: **Exploring the limits of transfer learning with a unified text-to-text transformer**, _Raffel et al._, Journal of Machine Learning Research 2023. \[[paper](https://arxiv.org/abs/1910.10683)\]\[[code](https://github.com/google-research/text-to-text-transfer-transformer)\]
- **BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension**, _Lewis et al._, ACL 2020. \[[paper](https://arxiv.org/abs/1910.13461)\]\[[code](https://github.com/facebookresearch/fairseq/tree/main/examples/bart)\]
- FLAN: **Finetuned Language Models Are Zero-Shot Learners**, _Wei et al._, ICLR 2022. \[[paper](https://arxiv.org/abs/2109.01652)\]\[[code](https://github.com/google-research/flan)\]
- Scaling Flan: **Scaling Instruction-Finetuned Language Models**, _Chung et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2210.11416)\]\[[mode](https://github.com/google-research/t5x/blob/main/docs/models.md#flan-t5-checkpoints)\]
- **Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context**, _Dai et al._, ACL 2019. \[[paper](https://arxiv.org/abs/1901.02860)\]\[[code](https://github.com/kimiyoung/transformer-xl)\]
- **XLNet: Generalized Autoregressive Pretraining for Language Understanding**, _Yang et al._, NeurIPS 2019. \[[paper](https://arxiv.org/abs/1906.08237)\]\[[code](https://github.com/zihangdai/xlnet)\]
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
- **DB-GPT: Empowering Database Interactions with Private Large Language Models**, _Xue et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.17449)\]\[[code](https://github.com/eosphoros-ai/DB-GPT)\]
- **OpenChat: Advancing Open-source Language Models with Mixed-Quality Data**, _Wang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.11235)\]\[[code](https://github.com/imoneoi/openchat)\]
- **Orca: Progressive Learning from Complex Explanation Traces of GPT-4**, _Mukherjee et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2306.02707)\]
- **PDFTriage: Question Answering over Long, Structured Documents**, _Saad-Falcon et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.08872)\]\[[code]\]
- **Prompt2Model: Generating Deployable Models from Natural Language Instructions**, _Viswanathan et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2308.12261)\]\[[code](https://github.com/neulab/prompt2model)\]
- **Shepherd: A Critic for Language Model Generation**, _Wang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2308.04592)\]\[[code](https://github.com/facebookresearch/Shepherd)\]
- **Alpaca: A Strong, Replicable Instruction-Following Model**, _Taori et al._, Stanford Blog 2023. \[[paper](https://crfm.stanford.edu/2023/03/13/alpaca.html)\]\[[code](https://github.com/tatsu-lab/stanford_alpaca)\]
- Vicuna: **Judging LLM-as-a-judge with MT-Bench and Chatbot Arena**, _Zheng et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2306.05685)\]\[[code](https://github.com/lm-sys/FastChat)\]\[[blog](https://lmsys.org/blog/2023-03-30-vicuna/)\]
- **WizardLM: Empowering Large Language Models to Follow Complex Instructions**, _Xu et al._, ICLR 2024. \[[paper](https://arxiv.org/abs/2304.12244)\]\[[code](https://github.com/nlpxucan/WizardLM)\]
- **WebGLM: Towards An Efficient Web-Enhanced Question Answering System with Human Preferences**, _Liu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2306.07906)\]\[[code](https://github.com/THUDM/WebGLM)\]

##### 3.2.1 AI Agent

- **LLM Powered Autonomous Agents**, _Lilian Weng_, 2023. \[[blog](https://lilianweng.github.io/posts/2023-06-23-agent/)\]
- **A Survey on Large Language Model based Autonomous Agents**, _Wang et al._, \[[paper](https://arxiv.org/abs/2308.11432)\]\[[code](https://github.com/Paitesanshi/LLM-Agent-Survey)\]
- **The Rise and Potential of Large Language Model Based Agents: A Survey**, _Xi et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.07864)\]\[[code](https://github.com/WooooDyy/LLM-Agent-Paper-List)\]
- **Agent AI: Surveying the Horizons of Multimodal Interaction**, _Durante et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.03568)\]

- **Agents: An Open-source Framework for Autonomous Language Agents**, _Zhou et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.07870)\]\[[code](https://github.com/aiwaves-cn/agents)\]
- **AgentBench: Evaluating LLMs as Agents**, _Liu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2308.03688)\]\[[code](https://github.com/THUDM/AgentBench)\]
- **AgentTuning: Enabling Generalized Agent Abilities for LLMs**, _Zeng et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.12823)\]\[[code](https://github.com/THUDM/AgentTuning)\]
- **AgentVerse: Facilitating Multi-Agent Collaboration and Exploring Emergent Behaviors**, _Chen et al._, ICLR 2024. \[[paper](https://arxiv.org/abs/2308.10848)\]\[[code](https://github.com/OpenBMB/AgentVerse/)\]
- **AppAgent: Multimodal Agents as Smartphone Users**, _Zhang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.13771)\]\[[code](https://github.com/mnotgod96/AppAgent)\]
- **Personal LLM Agents: Insights and Survey about the Capability, Efficiency and Security**, _Li et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.05459)\]\[[code](https://github.com/MobileLLM/Personal_LLM_Agents_Survey)\]
- **AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation**, _Wu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2308.08155)\]\[[code](https://github.com/microsoft/autogen)\]
- ChatDev: **Communicative Agents for Software Development**, _Qian et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2307.07924)\]\[[code](https://github.com/OpenBMB/ChatDev)\]\[[gpt-pilot](https://github.com/Pythagora-io/gpt-pilot)\]
- **Generative Agents: Interactive Simulacra of Human Behavior**, _Park et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2304.03442)\]\[[code](https://github.com/joonspk-research/generative_agents)\]\[[GPTeam](https://github.com/101dotxyz/GPTeam)\]
- **MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework**, _Hong et al._, ICLR 2024 Oral. \[[paper](https://arxiv.org/abs/2308.00352)\]\[[code](https://github.com/geekan/MetaGPT)\]
- **TaskWeaver: A Code-First Agent Framework**, _Qiao et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.17541)\]\[[code](https://github.com/microsoft/TaskWeaver)\]

- **Mind2Web: Towards a Generalist Agent for the Web**, _Deng et al._, NeurIPS 2023. \[[paper](https://arxiv.org/abs/2306.06070)\]\[[code](https://github.com/OSU-NLP-Group/Mind2Web)\]
- SeeAct: **GPT-4V(ision) is a Generalist Web Agent, if Grounded**, _Zheng et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.01614)\]\[[code](https://github.com/OSU-NLP-Group/SeeAct)\]

- **RT-1: Robotics Transformer for Real-World Control at Scale**, _Brohan et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2212.06817)\]\[[code](https://github.com/google-research/robotics_transformer)\]
- **RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control**, _Brohan et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2307.15818)\]\[[Unofficial Implementation](https://github.com/kyegomez/RT-2)\]
- **Open X-Embodiment: Robotic Learning Datasets and RT-X Models**, _Open X-Embodiment Collaboration_, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.08864)\]\[[code](https://github.com/google-deepmind/open_x_embodiment)\]
- **Shaping the future of advanced robotics**, Google DeepMind 2024. \[[blog](https://deepmind.google/discover/blog/shaping-the-future-of-advanced-robotics/)\]
- **Mobile ALOHA: Learning Bimanual Mobile Manipulation with Low-Cost Whole-Body Teleoperation**, _Fu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.02117)\]\[[Hardware Code](https://github.com/MarkFzp/mobile-aloha)\]\[[Learning Code](https://github.com/MarkFzp/act-plus-plus)\]

- \[[AutoGPT](https://github.com/Significant-Gravitas/AutoGPT)\]\[[GPT-Engineer](https://github.com/gpt-engineer-org/gpt-engineer)\]\[[AgentGPT](https://github.com/reworkd/AgentGPT)\]
- \[[BabyAGI](https://github.com/yoheinakajima/babyagi)\]\[[SuperAGI](https://github.com/TransformerOptimus/SuperAGI)\]\[[OpenAGI](https://github.com/agiresearch/OpenAGI)\]
- \[[open-interpreter](https://github.com/KillianLucas/open-interpreter)\]\[[Homepage](https://openinterpreter.com/)\]
- **XAgent: An Autonomous Agent for Complex Task Solving**, \[[blog](https://blog.x-agent.net/blog/xagent/)\]\[[code](https://github.com/OpenBMB/XAgent)\]
- \[[crewAI](https://github.com/joaomdmoura/crewAI)\]

##### 3.2.2 Academic

- **Galactica: A Large Language Model for Science**, _Taylor et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2211.09085)\]
- **K2: A Foundation Language Model for Geoscience Knowledge Understanding and Utilization**, _Deng et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2306.05064)\]\[[code](https://github.com/davendw49/k2)\]\[[pdf_parser](https://github.com/Acemap/pdf_parser)\]
- **GeoGalactica: A Scientific Large Language Model in Geoscience**, _Lin et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.00434)\]\[[code](https://github.com/geobrain-ai/geogalactica)\]\[[sciparser](https://github.com/davendw49/sciparser)\]

##### 3.2.3 Code

- **Neural code generation**, CMU 2024 Spring. \[[link](https://cmu-codegen.github.io/s2024/)\]
- \[[Awesome-Code-LLM](https://github.com/codefuse-ai/Awesome-Code-LLM)\]\[[MFTCoder](https://github.com/codefuse-ai/MFTCoder)\]
- **StarCoder: may the source be with you**, _Li et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.06161)\]\[[code](https://github.com/bigcode-project/starcoder)\]\[[bigcode-project](https://github.com/bigcode-project)\]\[[model](https://huggingface.co/bigcode)\]
- **WizardCoder: Empowering Code Large Language Models with Evol-Instruct**, _Luo et al._, ICLR 2024. \[[paper](https://arxiv.org/abs/2306.08568)\]\[[code](https://github.com/nlpxucan/WizardLM/tree/main/WizardCoder)\]
- **Code Generation with AlphaCodium: From Prompt Engineering to Flow Engineering**, _Ridnik et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.08500)\]\[[code](https://github.com/Codium-ai/AlphaCodium)\]
- **DeepSeek-Coder: When the Large Language Model Meets Programming -- The Rise of Code Intelligence**, _Guo et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.14196)\]\[[code](https://github.com/deepseek-ai/DeepSeek-Coder)\]
- **If LLM Is the Wizard, Then Code Is the Wand: A Survey on How Code Empowers Large Language Models to Serve as Intelligent Agents**, _Yang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.00812)\]

##### 3.2.4 Financial Application

- **DocLLM: A layout-aware generative language model for multimodal document understanding**, _Wang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.00908)\]
- **DocGraphLM: Documental Graph Language Model for Information Extraction**, _Wang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2401.02823)\]
- **FinGPT: Open-Source Financial Large Language Models**, _Yang et al._, IJCAI 2023. \[[paper](https://arxiv.org/abs/2306.06031)\]\[[code](https://github.com/AI4Finance-Foundation/FinGPT)\]
- **FinGPT: Instruction Tuning Benchmark for Open-Source Large Language Models in Financial Datasets**, _Wang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.04793)\]\[[code](https://github.com/AI4Finance-Foundation/FinGPT)\]
- **FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance**, _Liu et al._, arxiv 2020. \[[paper](https://arxiv.org/abs/2011.09607)\]\[[code](https://github.com/AI4Finance-Foundation/FinRL)\]
- **DISC-FinLLM: A Chinese Financial Large Language Model based on Multiple Experts Fine-tuning**, _Chen et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.15205)\]\[[code](https://github.com/FudanDISC/DISC-FinLLM)\]
- **XuanYuan 2.0: A Large Chinese Financial Chat Model with Hundreds of Billions Parameters**, _Zhang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.12002)\]\[[code](https://github.com/Duxiaoman-DI/XuanYuan)\]

##### 3.2.5 Information Retrieval

- **ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT**, _Khattab et al._, SIGIR 2020. \[[paper](https://arxiv.org/abs/2004.12832)\]
- **ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction**, _Santhanam et al. _, NAACL 2022. \[[paper](https://arxiv.org/abs/2112.01488)\]\[[code](https://github.com/stanford-futuredata/ColBERT)\]\[[RAGatouille](https://github.com/bclavie/RAGatouille)\]
- **Large Language Models for Information Retrieval: A Survey**, _Zhu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2308.07107)\]\[[code](https://github.com/RUC-NLPIR/LLM4IR-Survey)\]
- **UniGen: A Unified Generative Framework for Retrieval and Question Answering with Large Language Models**, _Li et al._, AAAI 2024. \[[paper](https://arxiv.org/abs/2312.11036)\]
- **INTERS: Unlocking the Power of Large Language Models in Search with Instruction Tuning**, _Zhu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.06532)\]\[[code](https://github.com/DaoD/INTERS)\]
- **SIGIR-AP 2023 Tutorial: Recent Advances in Generative Information Retrieval** \[[link](https://sigir-ap2023-generative-ir.github.io/)\]


##### 3.2.6 Math

- **ToRA: A Tool-Integrated Reasoning Agent for Mathematical Problem Solving**, _Gou et al._, ICLR 2024. \[[paper](https://arxiv.org/abs/2309.17452)\]\[[code](https://github.com/microsoft/ToRA)\]
- **MathVista: Evaluating Math Reasoning in Visual Contexts with GPT-4V, Bard, and Other Large Multimodal Models**, _Lu et al._, ICLR 2024 Oral. \[[paper](https://arxiv.org/abs/2310.02255)\]\[[code](https://github.com/lupantech/MathVista)\]

##### 3.2.7 Medicine and Law

- **ChatLaw: Open-Source Legal Large Language Model with Integrated External Knowledge Bases**, _Cui et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2306.16092)\]\[[code](https://github.com/PKU-YuanGroup/ChatLaw)\]
- Med-PaLM: **Large language models encode clinical knowledge**, _Singhal et al._, Nature 2023. \[[paper](https://www.nature.com/articles/s41586-023-06291-2)\]\[[Unofficial Implementation](https://github.com/kyegomez/Med-PaLM)\]
- AMIE: **Towards Conversational Diagnostic AI**, _Tu et al._, arxiv 2024.  \[[paper](https://arxiv.org/abs/2401.05654)\]\[[AMIE-pytorch](https://github.com/lucidrains/AMIE-pytorch)\]

##### 3.2.8 Recommend System

- **Recommender Systems with Generative Retrieval**, _Rajput et al._, NeurIPS 2022. \[[paper](https://arxiv.org/abs/2305.05065)\]
- \[[recommenders](https://github.com/recommenders-team/recommenders)\]\[[Source code for Twitter's Recommendation Algorithm](https://github.com/twitter/the-algorithm)\]

##### 3.2.9 Tool Learning

- **Tool Learning with Foundation Models**, _Qin et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2304.08354)\]\[[code](https://github.com/OpenBMB/BMTools)\]
- **Toolformer: Language Models Can Teach Themselves to Use Tools**, _Schick et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2302.04761)\]\[[toolformer-pytorch](https://github.com/lucidrains/toolformer-pytorch)\]\[[toolformer](https://github.com/conceptofmind/toolformer)\]
- **ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs**, _Qin et al._, ICLR 2024 Spotlight. \[[paper](https://arxiv.org/abs/2307.16789)\]\[[code](https://github.com/OpenBMB/ToolBench)\]
- **Gorilla: Large Language Model Connected with Massive APIs**, _Patil et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.15334)\]\[[code](https://github.com/ShishirPatil/gorilla)\]
- **GPT4Tools: Teaching Large Language Model to Use Tools via Self-instruction**, _Yang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.18752)\]\[[code](https://github.com/AILab-CVC/GPT4Tools)\]
- **Large Language Models as Tool Makers**, _Cai et al_, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.17126)\]\[[code](https://github.com/ctlllll/LLM-ToolMaker)\]
- **ToolAlpaca: Generalized Tool Learning for Language Models with 3000 Simulated Cases** _Tang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2306.05301)\]\[[code](https://github.com/tangqiaoyu/ToolAlpaca)\]
- **ToolChain\*: Efficient Action Space Navigation in Large Language Models with A\* Search**, _Zhuang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.13227)\]\[[code]\]
- **Chameleon: Plug-and-Play Compositional Reasoning with Large Language Models**, _Lu et al._, NeurIPS 2023. \[[paper](https://arxiv.org/abs/2304.09842)\]\[[code](https://github.com/lupantech/chameleon-llm)\]
- \[[ToolLearningPapers](https://github.com/thunlp/ToolLearningPapers)\]

#### 3.3 LLM Technique

- **How to Train Really Large Models on Many GPUs**, _Lilian Weng_, 2021. \[[blog](https://lilianweng.github.io/posts/2021-09-25-train-large/)\]
- **Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling**, _Biderman et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2304.01373)\]\[[code](https://github.com/EleutherAI/pythia)\]
- **Instruction Tuning with GPT-4**, _Peng et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2304.03277)\]\[[code](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)\]
- **DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines**, _Khattab et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.03714)\]\[[code](https://github.com/stanfordnlp/dspy)\]
- CALM: **LLM Augmented LLMs: Expanding Capabilities through Composition**, _Bansal et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.02412)\]\[[CALM-pytorch](https://github.com/lucidrains/CALM-pytorch)\]
- **Self-Rewarding Language Models**, _Yuan et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.10020)\]\[[unofficial implementation](https://github.com/lucidrains/self-rewarding-lm-pytorch)\]

##### 3.3.1 Alignment

- **Self-Instruct: Aligning Language Models with Self-Generated Instructions**, _Wang et al._, ACL 2023. \[[paper](https://arxiv.org/abs/2212.10560)\]\[[code](https://github.com/yizhongw/self-instruct)\]
- RLHF: \[[hf blog](https://huggingface.co/blog/rlhf)\]\[[OpenAI blog](https://openai.com/research/learning-from-human-preferences)\]\[[alignment blog](https://openai.com/blog/our-approach-to-alignment-research)\]\[[awesome-RLHF](https://github.com/opendilab/awesome-RLHF)\]
- **Secrets of RLHF in Large Language Models** \[[MOSS-RLHF](https://github.com/OpenLMLab/MOSS-RLHF)\]\[[Part I](https://arxiv.org/abs/2307.04964)\]\[[Part II](https://arxiv.org/abs/2401.06080)\]\[[OpenRLHF](https://github.com/OpenLLMAI/OpenRLHF)\]
- DPO: **Direct Preference Optimization: Your Language Model is Secretly a Reward Model**, _Rafailov et al._, NeurIPS 2023 Runner-up Award. \[[paper](https://arxiv.org/abs/2305.18290)\]\[[Unofficial Implementation](https://github.com/eric-mitchell/direct-preference-optimization)\]\[[trl](https://github.com/huggingface/trl)\]
- BPO: **Black-Box Prompt Optimization: Aligning Large Language Models without Model Training**, _Cheng et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.04155)\]\[[code](https://github.com/thu-coai/BPO)\]
- **RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback**, _Lee et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.00267)\]\[[code]\]\[[awesome-RLAIF](https://github.com/mengdi-li/awesome-RLAIF)\]
- **ReMax: A Simple, Effective, and Efficient Reinforcement Learning Method for Aligning Large Language Models**, _Li et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.10505)\]\[[code](https://github.com/liziniu/ReMax)\]
- **Zephyr: Direct Distillation of LM Alignment**, _Tunstall et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.16944)\]\[[code](https://github.com/huggingface/alignment-handbook)\]
- **Policy Optimization in RLHF: The Impact of Out-of-preference Data**, _Li et al._, \[[paper](https://arxiv.org/abs/2312.10584)\]\[[code](https://github.com/liziniu/policy_optimization)\]
- **Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision**, _Burns et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.09390)\]\[[code](https://github.com/openai/weak-to-strong)\]
- SPIN: **Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models**, _Chen et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.01335)\]\[[code]\]
- Anthropic: **Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training**, _Hubinger et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.05566)\]

##### 3.3.2 Context Length

- **StreamingLLM: Efficient Streaming Language Models with Attention Sinks**, _Xiao et al._, ICLR 2024. \[[paper](https://arxiv.org/abs/2309.17453)\]\[[code](https://github.com/mit-han-lab/streaming-llm)\]\[[SwiftInfer](https://github.com/hpcaitech/SwiftInfer)\]\[[SwiftInfer blog](https://hpc-ai.com/blog/colossal-ai-swiftinfer)\]
- **LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression**, _Jiang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.06839)\]\[[code](https://github.com/microsoft/LLMLingua)\]
- **YaRN: Efficient Context Window Extension of Large Language Models**, _Peng et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.00071)\]\[[code](https://github.com/jquesnelle/yarn)\]
- **The What, Why, and How of Context Length Extension Techniques in Large Language Models -- A Detailed Survey**, _Pawar et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.07872)\]

##### 3.3.3 Corpus

- \[[datatrove](https://github.com/huggingface/datatrove)\]

##### 3.3.4 Evaluation

- MMLU: **Measuring Massive Multitask Language Understanding**, _Hendrycks et al._, ICLR 2021.  \[[paper](https://arxiv.org/abs/2009.03300)\]\[[code](https://github.com/hendrycks/test)\]
- HELM: **Holistic Evaluation of Language Models**, _Liang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2211.09110)\]\[[code](https://github.com/stanford-crfm/helm)\]
- **SuperCLUE: A Comprehensive Chinese Large Language Model Benchmark**, _Xu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2307.15020)\]\[[code](https://github.com/CLUEbenchmark/SuperCLUE)\]
- **C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for Foundation Models**, _Huang et al._, NeurIPS 2023. \[[paper](https://arxiv.org/abs/2305.08322)\]\[[code](https://github.com/hkust-nlp/ceval)\]
- **CMMLU: Measuring massive multitask language understanding in Chinese**, _Li et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2306.09212)\]\[[code](https://github.com/haonan-li/CMMLU)\]
- **CMMMU: A Chinese Massive Multi-discipline Multimodal Understanding Benchmark**, _Zhang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.11944)\]\[[code](https://github.com/CMMMU-Benchmark/CMMMU)\]
- \[[Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)\]
- \[[AlpacaEval Leaderboard](https://tatsu-lab.github.io/alpaca_eval/)\]\[[alpaca_eval](https://github.com/tatsu-lab/alpaca_eval)\]
- \[[Chatbot-Arena-Leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)\]\[[blog](https://lmsys.org/blog/2023-05-03-arena/)\]\[[FastChat](https://github.com/lm-sys/FastChat)\]
- \[[OpenCompass](https://github.com/open-compass/opencompass)\]

##### 3.3.5 Hallucination

- **The Dawn After the Dark: An Empirical Study on Factuality Hallucination in Large Language Models**, _Li et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.03205)\]\[[code](https://github.com/RUCAIBox/HaluEval-2.0)\]

##### 3.3.6 Inference

- **Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems**, _Miao et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.15234)\]\[[Awesome-Quantization-Papers](https://github.com/Zhen-Dong/Awesome-Quantization-Papers)\]\[[awesome-model-quantization](https://github.com/htqin/awesome-model-quantization)\]
- **LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale**, _Dettmers et al._, NeurIPS 2022. \[[paper](https://arxiv.org/abs/2208.07339)\]\[[code](https://github.com/TimDettmers/bitsandbytes)\]
- **LLM-FP4: 4-Bit Floating-Point Quantized Transformers**, _Liu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.16836)\]\[[code](https://github.com/nbasyl/LLM-FP4)\]
- **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers**, _Frantar et al._, ICLR 2023. \[[paper](https://arxiv.org/abs/2210.17323)\]\[[code](https://github.com/IST-DASLab/gptq)\]\[[AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)\]
- **QMoE: Practical Sub-1-Bit Compression of Trillion-Parameter Models**, _Frantar et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.16795)\]\[[code](https://github.com/IST-DASLab/qmoe)\]
- **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration**, _Lin et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2306.00978)\]\[[code](https://github.com/mit-han-lab/llm-awq)\]
- **LLM in a flash: Efficient Large Language Model Inference with Limited Memory**, _Alizadeh et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.11514)\]\[[air_llm](https://github.com/lyogavin/Anima/tree/main/air_llm)\]
- **LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models**, _Jiang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.05736)\]\[[code](https://github.com/microsoft/LLMLingua)\]
- **FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU**, _Sheng et al._, ICML 2023. \[[paper](https://arxiv.org/abs/2303.06865)\]\[[code](https://github.com/FMInference/FlexGen)\]
- **PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU**, _Song et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.12456)\]\[[code](https://github.com/SJTU-IPADS/PowerInfer)\]\[[llama.cpp](https://github.com/ggerganov/llama.cpp)\]\[[Anima](https://github.com/lyogavin/Anima)\]
- vllm: **Efficient Memory Management for Large Language Model Serving with PagedAttention**, _Kwon et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.06180)\]\[[code](https://github.com/vllm-project/vllm)\]
- **Fast and Expressive LLM Inference with RadixAttention and SGLang**, _Zheng et al._, Stanford blog 2024. \[[blog](https://lmsys.org/blog/2024-01-17-sglang/)\]\[[code](https://github.com/sgl-project/sglang/)\]
- **Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads**, _Cai et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.10774)\]\[[code](https://github.com/FasterDecoding/Medusa)\]
- **APAR: LLMs Can Do Auto-Parallel Auto-Regressive Decoding**, _Liu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.06761)\]\[[code]\]
- \[[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)\]\[[FasterTransformer](https://github.com/NVIDIA/FasterTransformer)\]\[[TritonServer](https://github.com/triton-inference-server/server)\]
- \[[DeepSpeed-MII](https://github.com/microsoft/DeepSpeed-MII)\]\[[DeepSpeed-FastGen](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen)\]\[[ONNX Runtime](https://github.com/microsoft/onnxruntime)\]\[[onnx](https://github.com/onnx/onnx)\]
- \[[text-generation-inference](https://github.com/huggingface/text-generation-inference)\]\[[quantization](https://huggingface.co/docs/transformers/main/en/quantization)\]
- \[[OpenLLM](https://github.com/bentoml/OpenLLM)\]\[[mlc-llm](https://github.com/mlc-ai/mlc-llm)\]
- \[[LMDeploy](https://github.com/InternLM/lmdeploy)\]
- \[[ggml](https://github.com/ggerganov/ggml)\]\[[exllamav2](https://github.com/turboderp/exllamav2)\]\[[llama.cpp](https://github.com/ggerganov/llama.cpp)\]\[[gpt-fast](https://github.com/pytorch-labs/gpt-fast)\]\[[fastllm](https://github.com/ztxz16/fastllm)\]

##### 3.3.7 MoE

- **Mixture of Experts Explained**, _Sanseviero et al._, Hugging Face Blog 2023. \[[blog](https://huggingface.co/blog/moe)\]
- **Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer**, _Shazeer et al._, arxiv 2017. \[[paper](https://arxiv.org/abs/1701.06538)\]\[[Re-Implementation](https://github.com/davidmrau/mixture-of-experts)\]
- **GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding**, _Lepikhin et al._, arxiv 2020. \[[paper](https://arxiv.org/abs/2006.16668)\]\[[mixture-of-experts](https://github.com/lucidrains/mixture-of-experts)\]
- **MegaBlocks: Efficient Sparse Training with Mixture-of-Experts**, _Gale et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2211.15841)\]\[[code](https://github.com/stanford-futuredata/megablocks)\]
- **Mixture-of-Experts Meets Instruction Tuning:A Winning Combination for Large Language Models**, _Shen et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.14705)\]\[[code]\]
- **Fast Inference of Mixture-of-Experts Language Models with Offloading**, _Eliseev and Mazur_, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.17238)\]\[[code](https://github.com/dvmazur/mixtral-offloading)\]
- **Mixtral of Experts**, _Jiang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2401.04088)\]\[[code](https://github.com/mistralai/mistral-src)\]\[[megablocks-public](https://github.com/mistralai/megablocks-public)\]\[[model](https://huggingface.co/mistralai)\]\[[blog](https://mistral.ai/news/mixtral-of-experts/)\]\[[Chinese-Mixtral-8x7B](https://github.com/HIT-SCIR/Chinese-Mixtral-8x7B)\]
- **DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models**, _Dai et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.06066)\]\[[code](https://github.com/deepseek-ai/DeepSeek-MoE)\]
- \[[llama-moe](https://github.com/pjlab-sys4nlp/llama-moe)\]\[[Aurora](https://github.com/WangRongsheng/Aurora)\]\[[OpenMoE](https://github.com/XueFuzhao/OpenMoE)\]\[[makeMoE](https://github.com/AviSoori1x/makeMoE)\]

##### 3.3.8 PEFT (Parameter-efficient Fine-tuning)

- \[[DeepSpeed](https://github.com/microsoft/DeepSpeed)\]\[[DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples)\]
- \[[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)\]
- \[[Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed)\]\[[Megatron-DeepSpeed](https://github.com/bigscience-workshop/Megatron-DeepSpeed)\]
- \[[PEFT](https://github.com/huggingface/peft)\]\[[Transformer Reinforcement Learning](https://github.com/huggingface/trl)\]\[[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)\]
- \[[mergekit](https://github.com/cg123/mergekit)\]\[[Model Merging](https://huggingface.co/collections/osanseviero/model-merging-65097893623330a3a51ead66)\]

- **LoRA: Low-Rank Adaptation of Large Language Models**, _Hu et al._, arxiv 2021. \[[paper](https://arxiv.org/abs/2106.09685)\]\[[code](https://github.com/microsoft/LoRA)\]\[[LoRA From Scratch](https://lightning.ai/lightning-ai/studios/code-lora-from-scratch)\]
- **QLoRA: Efficient Finetuning of Quantized LLMs**, _Dettmers et al._, NeurIPS 2023 Oral. \[[paper](https://arxiv.org/abs/2305.14314)\]\[[code](https://github.com/artidoro/qlora)\]\[[bitsandbytes](https://github.com/TimDettmers/bitsandbytes)\]\[[unsloth
](https://github.com/unslothai/unsloth)\]
- **Prefix-Tuning: Optimizing Continuous Prompts for Generation**, _Li et al._, ACL 2021. \[[paper](https://arxiv.org/abs/2101.00190)\]\[[code](https://github.com/XiangLi1999/PrefixTuning)\]
- **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**, _Dao et al._, NeurIPS 2022. \[[paper](https://arxiv.org/abs/2205.14135)\]\[[code](https://github.com/Dao-AILab/flash-attention)\]
- **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning**, _Tri Dao_, arxiv 2023. \[[paper](https://arxiv.org/abs/2307.08691)\]\[[code](https://github.com/Dao-AILab/flash-attention)\]
- P-Tuning: **GPT Understands, Too**, _Liu et al._, arxiv 2021. \[[paper](https://arxiv.org/abs/2103.10385)\]\[[code](https://github.com/THUDM/P-tuning)\]
- **P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks**, _Liu et al._, ACL 2022. \[[paper](https://arxiv.org/abs/2110.07602)\]\[[code](https://github.com/THUDM/P-tuning-v2)\]
- **Mixed Precision Training**, _Micikevicius et al._, ICLR 2018. \[[paper](https://arxiv.org/abs/1710.03740)\]
- **8-bit Optimizers via Block-wise Quantization** _Dettmers et al._, ICLR 2022. \[[paper](https://arxiv.org/abs/2110.02861)\]\[[code](https://github.com/timdettmers/bitsandbytes)\]
- **FP8-LM: Training FP8 Large Language Models** _Peng et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.18313)\]\[[code](https://github.com/Azure/MS-AMP)\]

##### 3.3.9 Prompt Learning

- **OpenPrompt: An Open-source Framework for Prompt-learning**, _Ding et al._, arxiv 2021. \[[paper](https://arxiv.org/abs/2111.01998)\]\[[code](https://github.com/thunlp/OpenPrompt)\]
- \[[PromptPapers](https://github.com/thunlp/PromptPapers)\]

- **PAL: Program-aided Language Models**, _Gao et al._, ICML 2023. \[[paper](https://arxiv.org/abs/2211.10435)\]\[[code](https://github.com/reasoning-machines/pal)\]

- **Meta-Prompting: Enhancing Language Models with Task-Agnostic Scaffolding**, _Suzgun and Kalai_, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.12954)\]\[[code](https://github.com/suzgunmirac/meta-prompting)\]

##### 3.3.10 RAG (Retrieval Augmented Generation)

- **Retrieval-Augmented Generation for Large Language Models: A Survey**, _Gao et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.10997)\]\[[code](https://github.com/Tongji-KGLLM/RAG-Survey)\]
- **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**, _Lewis et al._, NeurIPS 2020. \[[paper](https://arxiv.org/abs/2005.11401)\]\[[code](https://github.com/huggingface/transformers/tree/main/examples/research_projects/rag)\]\[[model](https://huggingface.co/facebook/rag-token-nq)\]\[[docs](https://huggingface.co/docs/transformers/main/model_doc/rag)\]\[[FAISS](https://github.com/facebookresearch/faiss)\]
- **Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection**, _Asai et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.11511)\]\[[code](https://github.com/AkariAsai/self-rag)\]
- **Dense Passage Retrieval for Open-Domain Question Answering**, _Karpukhin et al._, EMNLP 2020. \[[paper](https://arxiv.org/abs/2004.04906)\]\[[code](https://github.com/facebookresearch/DPR)\]
- **Internet-Augmented Dialogue Generation** _Komeili et al._, arxiv 2021. \[[paper](https://arxiv.org/abs/2107.07566)\]
- **FreshLLMs: Refreshing Large Language Models with Search Engine Augmentation**, _Vu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.03214)\]\[[code](https://github.com/freshllms/freshqa)\]
- **Chain-of-Note: Enhancing Robustness in Retrieval-Augmented Language Models**, _Yu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.09210)\]
- **ACL 2023 Tutorial: Retrieval-based Language Models and Applications**, _Asai et al._, ACL 2023. \[[link](https://acl2023-retrieval-lm.github.io/)\]
- \[[Advanced RAG Techniques: an Illustrated Overview](https://pub.towardsai.net/advanced-rag-techniques-an-illustrated-overview-04d193d8fec6)\]\[[Chinese Version](https://zhuanlan.zhihu.com/p/674755232)\]
- \[[LangChain](https://github.com/langchain-ai/langchain)\]\[[blog](https://blog.langchain.dev/deconstructing-rag/)\]
- \[[LlamaIndex](https://github.com/run-llama/llama_index)\]\[[A Cheat Sheet and Some Recipes For Building Advanced RAG](https://blog.llamaindex.ai/a-cheat-sheet-and-some-recipes-for-building-advanced-rag-803a9d94c41b)\]
- \[[chatgpt-retrieval-plugin](https://github.com/openai/chatgpt-retrieval-plugin)\]
- \[[Langchain-Chatchat](https://github.com/chatchat-space/Langchain-Chatchat)\]
- **Browse the web with GPT-4V and Vimium** \[[vimGPT](https://github.com/ishan0102/vimGPT)\]

###### Text Embedding

- **Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks**, _Reimers et al._, EMNLP 2019. \[[paper](https://arxiv.org/abs/1908.10084)\]\[[code](https://github.com/UKPLab/sentence-transformers)\]\[[model](https://huggingface.co/sentence-transformers)\]\[[model](https://huggingface.co/sentence-transformers)\]
- **SimCSE: Simple Contrastive Learning of Sentence Embeddings**, _Gao et al._, EMNLP 2021. \[[paper](https://arxiv.org/abs/2104.08821)\]\[[code](https://github.com/princeton-nlp/SimCSE)\]
- OpenAI: **Text and Code Embeddings by Contrastive Pre-Training**, _Neelakantan et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2201.10005)\]\[[blog](https://openai.com/blog/introducing-text-and-code-embeddings)\]
- MRL: **Matryoshka Representation Learning**, _Kusupati et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2205.13147)\]\[[code](https://github.com/RAIVNLab/MRL)\]
- BGE: **C-Pack: Packaged Resources To Advance General Chinese Embedding**, _Xiao et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.07597)\]\[[code](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/baai_general_embedding)\]\[[FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)\]
- LLM-Embedder: **Retrieve Anything To Augment Large Language Models**, _Zhang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.07554)\]\[[code](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_embedder)\]\[[FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)\]
- \[[m3e-base](https://huggingface.co/moka-ai/m3e-base)\]
- **Jina Embeddings 2: 8192-Token General-Purpose Text Embeddings for Long Documents**, _Günther et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.19923)\]\[[model](https://huggingface.co/jinaai/jina-embeddings-v2-base-en)\]
- GTE: **Towards General Text Embeddings with Multi-stage Contrastive Learning**, _Li et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2308.03281)\]\[[model](https://huggingface.co/thenlper/gte-large-zh)\]
- \[[BCEmbedding](https://github.com/netease-youdao/BCEmbedding)\]\[[bce-embedding-base_v1](https://huggingface.co/maidalun1020/bce-embedding-base_v1)\]\[[bce-reranker-base_v1](https://huggingface.co/maidalun1020/bce-reranker-base_v1)\]
- \[[CohereV3](https://huggingface.co/Cohere/Cohere-embed-multilingual-v3.0)\]
- **Improving Text Embeddings with Large Language Models**, _Wang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.00368)\]\[[code](https://github.com/microsoft/unilm/tree/master/e5)\]\[[model](https://huggingface.co/intfloat/e5-mistral-7b-instruct)\]

##### 3.3.11 Reasoning

- Few-Shot-CoT: **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models**, _Wei et al._, NeurIPS 2022. \[[paper](https://arxiv.org/abs/2201.11903)\]\[[chain-of-thought-hub](https://github.com/FranxYao/chain-of-thought-hub)\]
- **Self-Consistency Improves Chain of Thought Reasoning in Language Models**, _Wang et al._, ICLR 2023. \[[paper](https://arxiv.org/abs/2203.11171)\]
- Zero-Shot-CoT: **Large Language Models are Zero-Shot Reasoners**, _Kojima et al._, NeurIPS 2022. \[[paper](https://arxiv.org/abs/2205.11916)\]\[[code](https://github.com/kojima-takeshi188/zero_shot_cot)\]
- Auto-CoT: **Automatic Chain of Thought Prompting in Large Language Models**, _Zhang et al._, ICLR 2023. \[[paper](https://arxiv.org/abs/2210.03493)\]\[[code](https://github.com/amazon-science/auto-cot)\]
- **Multimodal Chain-of-Thought Reasoning in Language Models**, _Zhang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2302.00923)\]\[[code](https://github.com/amazon-science/mm-cot)\]
- **ReAct: Synergizing Reasoning and Acting in Language Models**, _Yao et al._, ICLR 2023. \[[paper](https://arxiv.org/abs/2210.03629)\]\[[code](https://github.com/ysymyth/ReAct)\]
- **Tree of Thoughts: Deliberate Problem Solving with Large Language Models**, _Yao et al._, NeurIPS 2023. \[[paper](https://arxiv.org/abs/2305.10601)\]\[[code](https://github.com/princeton-nlp/tree-of-thought-llm)\]\[[Plug in and Play Implementation](https://github.com/kyegomez/tree-of-thoughts)\]
- **Graph of Thoughts: Solving Elaborate Problems with Large Language Models**, _Besta et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2308.09687)\]\[[code](https://github.com/spcl/graph-of-thoughts)\]
- **Cumulative Reasoning with Large Language Models**, _Zhang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2308.04371)\]\[[code](https://github.com/iiis-ai/cumulative-reasoning)\]
- **Algorithm of Thoughts: Enhancing Exploration of Ideas in Large Language Models**, _Sel et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2308.10379)\]\[[unofficial code](https://github.com/kyegomez/Algorithm-Of-Thoughts)\]
- **Everything of Thoughts: Defying the Law of Penrose Triangle for Thought Generation**, _Ding et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.04254)\]\[[code](https://github.com/microsoft/Everything-of-Thoughts-XoT)\]
- LEMA: **Learning From Mistakes Makes LLM Better Reasoner**, _An et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.20689)\]\[[code](https://github.com/microsoft/LEMA)\]
- **Chain of Code: Reasoning with a Language Model-Augmented Code Emulator**, _Li et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.04474)\]\[[code]\]
- **The Impact of Reasoning Step Length on Large Language Models**, _Jin et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.04925)\]\[[code](https://github.com/jmyissb/The-Impact-of-Reasoning-Step-Length-on-Large-Language-Models)\]

- ReST-EM: **Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models**, _Singh et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.06585)\]\[[unofficial code](https://github.com/lucidrains/ReST-EM-pytorch)\]
- **ReST meets ReAct: Self-Improvement for Multi-Step Reasoning LLM Agent**, _Aksitov et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.10003)\]\[[code]\]
- **Orca 2: Teaching Small Language Models How to Reason**, _Mitra et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.11045)\]\[[code]\]

###### Survey



#### 3.4 LLM Theory

- **Scaling Laws for Neural Language Models**, _Kaplan et al._, arxiv 2020. \[[paper](https://arxiv.org/abs/2001.08361)\]\[[unofficial code](https://github.com/shehper/scaling_laws)\]
- **Emergent Abilities of Large Language Models**, _Wei et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2206.07682)\]
- Chinchilla: **Training Compute-Optimal Large Language Models**, _Hoffmann et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2203.15556)\]
- **Scaling Laws for Autoregressive Generative Modeling**, _Henighan et al._, arxiv 2020. \[[paper](https://arxiv.org/abs/2010.14701)\]
- **Are Emergent Abilities of Large Language Models a Mirage**, _Schaeffer et al._, NeurIPS 2023 Outstanding Paper. \[[paper](https://arxiv.org/abs/2304.15004)\]
- S2A: **System 2 Attention (is something you might need too)**, _Weston et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.11829)\]
- **Scalable Pre-training of Large Autoregressive Image Models**, _El-Nouby et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.08541)\]\[[code](https://github.com/apple/ml-aim)\]

- ROME: **Locating and Editing Factual Associations in GPT**, _Meng et al._, NeurIPS 2022. \[[paper](https://arxiv.org/abs/2202.05262)\]\[[code](https://github.com/kmeng01/rome)\]\[[FastEdit](https://github.com/hiyouga/FastEdit)\]
- **Editing Large Language Models: Problems, Methods, and Opportunities**, _Yao et al._, EMNLP 2023. \[[paper](https://arxiv.org/abs/2305.13172)\]\[[code](https://github.com/zjunlp/EasyEdit)\]
- **A Comprehensive Study of Knowledge Editing for Large Language Models**, _Zhang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.01286)\]\[[code](https://github.com/zjunlp/EasyEdit)\]

#### 3.5 Chinese Model

- \[[Awesome-Chinese-LLM](https://github.com/HqWu-HITCS/Awesome-Chinese-LLM)\]
- **GLM: General Language Model Pretraining with Autoregressive Blank Infilling**, _Du et al._, ACL 2022. \[[paper](https://arxiv.org/abs/2103.10360)\]\[[code](https://github.com/THUDM/GLM)\]\[[ChatGLM3](https://github.com/THUDM/ChatGLM3)\]
- **GLM-130B: An Open Bilingual Pre-trained Model**, _Zeng et al._, ICLR 2023. \[[paper](https://arxiv.org/abs/2210.02414v2)\]\[[code](https://github.com/THUDM/GLM-130B/)\]
- **Baichuan 2: Open Large-scale Language Models**, _Yang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.10305)\]\[[code](https://github.com/baichuan-inc/Baichuan2)\]
- **Qwen Technical Report**, _Bai et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.16609)\]\[[code](https://github.com/QwenLM/Qwen)\]
- **DeepSeek LLM: Scaling Open-Source Language Models with Longtermism**, _Bi et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.02954)\]\[[DeepSeek-LLM](https://github.com/deepseek-ai/DeepSeek-LLM)\]\[[DeepSeek-Coder)](https://github.com/deepseek-ai/DeepSeek-Coder)\]
- **Efficient and Effective Text Encoding for Chinese LLaMA and Alpaca**, Cui et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2304.08177)\]\[[code](https://github.com/ymcui/Chinese-LLaMA-Alpaca)\]\[[Chinese-LLaMA-Alpaca-2](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2)\]\[[baby-llama2-chinese](https://github.com/DLLXW/baby-llama2-chinese)\]
- **TeleChat Technical Report**, _Wang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.03804)\]\[[code](https://github.com/Tele-AI/Telechat)\]
- \[[Yi](https://github.com/01-ai/Yi)\]\[[Orion](https://github.com/OrionStarAI/Orion)\]\[[InternLM](https://github.com/InternLM/InternLM)\]
- \[[MOSS](https://github.com/OpenLMLab/MOSS)\]\[[MOSS-RLHF](https://github.com/OpenLMLab/MOSS-RLHF)\]
- \[[BELLE](https://github.com/LianjiaTech/BELLE)\]\[[Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM)\]
- \[[FlagAlpha/Llama2-Chinese](https://github.com/FlagAlpha/Llama2-Chinese)\]\[[LinkSoul-AI/Chinese-Llama-2-7b](https://github.com/LinkSoul-AI/Chinese-Llama-2-7b)\]
- \[[Firefly](https://github.com/yangjianxin1/Firefly)\]\[[GPT2-chitchat](https://github.com/yangjianxin1/GPT2-chitchat)\]
- Alpaca-CoT: **An Empirical Study of Instruction-tuning Large Language Models in Chinese**, _Si et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.07328)\]\[[code](https://github.com/PhoebusSi/Alpaca-CoT)\]

---

## CV

- **CS231n: Deep Learning for Computer Vision** \[[link](http://cs231n.stanford.edu/)\]

### 1. Basic for CV
- AlexNet: **ImageNet Classification with Deep Convolutional Neural Networks**, _Krizhevsky et al._, NIPS 2012. \[[paper](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)\]
- VGG: **Very Deep Convolutional Networks for Large-Scale Image Recognition**, _Simonyan et al._, ICLR 2015. \[[paper](https://arxiv.org/abs/1409.1556)\]
- **GoogLeNet_Going Deeper with Convolutions**, _Szegedy et al._, CVPR 2015. \[[paper](https://arxiv.org/abs/1409.4842)\]
- **ResNet_Deep Residual Learning for Image Recognition**, _He et al._, CVPR 2016 Best Paper. \[[paper](https://arxiv.org/abs/1512.03385)\]\[[code](https://github.com/KaimingHe/deep-residual-networks)\]
- DenseNet: **Densely Connected Convolutional Networks**, _Huang et al._, CVPR 2017 Oral. \[[paper](https://arxiv.org/abs/1608.06993)\]\[[code](https://github.com/liuzhuang13/DenseNet)\]
- **EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks**, _Tan et al._, ICML 2019. \[[paper](https://arxiv.org/abs/1905.11946)\]\[[code](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)\]\[[EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)\]
- BYOL: **Bootstrap your own latent: A new approach to self-supervised Learning**, _Grill et al._, arxiv 2020. \[[paper](https://arxiv.org/abs/2006.07733)\]\[[code](https://github.com/google-deepmind/deepmind-research/tree/master/byol)\]\[[byol-pytorch](https://github.com/lucidrains/byol-pytorch)\]

### 2. Contrastive Learning

### 3. CV Application

- CodeFormer: **Towards Robust Blind Face Restoration with Codebook Lookup Transformer**, _Zhou et al._, NeurIPS 2022. \[[paper](https://arxiv.org/abs/2206.11253)\]\[[code](https://github.com/sczhou/CodeFormer)\]
- **BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers**, _Li et al._, ECCV 2022. \[[paper](https://arxiv.org/abs/2203.17270)\]\[[code](https://github.com/fundamentalvision/BEVFormer)\]
- UniAD: **Planning-oriented Autonomous Driving**, _Hu et al._, CVPR 2023 Best Paper. \[[paper](https://arxiv.org/abs/2212.10156)\]\[[code](https://github.com/OpenDriveLab/UniAD)\]

- **PhotoMaker: Customizing Realistic Human Photos via Stacked ID Embedding**, _Li et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.04461)\]\[[code](https://github.com/TencentARC/PhotoMaker)\]
- **InstantID: Zero-shot Identity-Preserving Generation in Seconds**, _Wang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.07519)\]\[[code](https://github.com/InstantID/InstantID)\]
- **ReplaceAnything as you want: Ultra-high quality content replacement**, \[[link](https://aigcdesigngroup.github.io/replace-anything/)\]

### 4. Foundation Model

- ViT: **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**, _Dosovitskiy et al._, ICLR 2021. \[[paper](https://arxiv.org/abs/2010.11929)\]\[[code](https://github.com/google-research/vision_transformer)\]\[[Pytorch Implementation](https://github.com/lucidrains/vit-pytorch)\]
- DeiT: **Training data-efficient image transformers & distillation through attention**, _Touvron et al._, ICML 2021. \[[paper](https://arxiv.org/abs/2012.12877)\]\[[code](https://github.com/facebookresearch/deit)\]
- **ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision**, _Kim et al._, ICML 2021. \[[paper](https://arxiv.org/abs/2102.03334)\]\[[code](https://github.com/dandelin/vilt)\]
- **Swin Transformer: Hierarchical Vision Transformer using Shifted Windows**, _Liu et al._, ICCV 2021. \[[paper](https://arxiv.org/abs/2103.14030)\]\[[code](https://github.com/microsoft/Swin-Transformer)\]
- MAE: **Masked Autoencoders Are Scalable Vision Learners**, _He et al._, CVPR 2022. \[[paper](https://arxiv.org/abs/2111.06377)\]\[[code](https://github.com/facebookresearch/mae)\]
- LVM: **Sequential Modeling Enables Scalable Learning for Large Vision Models**, _Bai et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.00785)\]\[[code](https://github.com/ytongbai/LVM)\]
- GLEE: **General Object Foundation Model for Images and Videos at Scale**, _Wu wt al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.09158)\]\[[code](https://github.com/FoundationVision/GLEE)\]
- **Tokenize Anything via Prompting**, _Pan et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.09128)\]\[[code](https://github.com/baaivision/tokenize-anything)\]
- **Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model** _Zhu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.09417)\]\[[code](https://github.com/hustvl/Vim)\]\[[VMamba](https://github.com/MzeroMiko/VMamba)\]
- **Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data**, _Yang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.10891)\]\[[code](https://github.com/LiheYoung/Depth-Anything)\]

### 5. Generative Model (GAN and VAE)

- GAN: **Generative Adversarial Networks**, _Goodfellow et al._, arxiv 2014. \[[paper](https://arxiv.org/abs/1406.2661)\]\[[code](https://github.com/goodfeli/adversarial)\]\[[Pytorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN)\]
- StyleGAN3: **Alias-Free Generative Adversarial Networks**, _Karras etal._, NeurIPS 2021. \[[paper](https://arxiv.org/abs/2106.12423)\]\[[code](https://github.com/NVlabs/stylegan3)\]
- GigaGAN: **Scaling up GANs for Text-to-Image Synthesis**, _Kang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2303.05511)\]\[[code](https://github.com/lucidrains/gigagan-pytorch)\]
- \[[pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)\]
- VAE: **Auto-Encoding Variational Bayes**, _Kingma et al._, arxiv 2013. \[[paper](https://arxiv.org/abs/1312.6114)\]\[[code](https://github.com/jaanli/variational-autoencoder)\]\[[Pytorch-VAE](https://github.com/AntixK/PyTorch-VAE)\]
- VQ-VAE: **Neural Discrete Representation Learning**, _Oord et al._, NIPS 2017. \[[paper](https://arxiv.org/abs/1711.00937)\]\[[code](https://github.com/AntixK/PyTorch-VAE/blob/master/models/vq_vae.py)\]\[[vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch)\]
- VQ-VAE-2: **Generating Diverse High-Fidelity Images with VQ-VAE-2**, Razavi et al._, arxiv 2019. \[[paper](https://arxiv.org/abs/1906.00446)\]\[[code](https://github.com/rosinality/vq-vae-2-pytorch)\]

### 6. Image Editing

### 7. Object Detection

### 8. Semantic Segmentation

- **U-Net: Convolutional Networks for Biomedical Image Segmentation**, _Ronneberger et al._, MICCAI 2015. \[[paper](https://arxiv.org/abs/1505.04597)\]\[[code](https://github.com/milesial/Pytorch-UNet)\]
- **Segment Anything**, _Kirillov et al._, ICCV 2023. \[[paper](https://arxiv.org/abs/2304.02643)\]\[[code](https://github.com/facebookresearch/segment-anything)\]

### 9. Video

- **VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training**, _Tong et al._, NeurIPS 2022 Spotlight. \[[paper](https://arxiv.org/abs/2203.12602)\]\[[code](https://github.com/MCG-NJU/VideoMAE)\]
- **MagicVideo-V2: Multi-Stage High-Aesthetic Video Generation**, _Wang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.04468)\]

### 10. Survey for CV

---

## Multimodal

### 1. Audio

- Whisper: **Robust Speech Recognition via Large-Scale Weak Supervision**, _Radford et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2212.04356)\]\[[code](https://github.com/openai/whisper)\]\[[whisper.cpp](https://github.com/ggerganov/whisper.cpp)\]\[[faster-whisper](https://github.com/SYSTRAN/faster-whisper)\]
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
- **LLaSM: Large Language and Speech Model**, _Shu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2308.15930)\]\[[code](https://github.com/LinkSoul-AI/LLaSM)\]

- **Github Repositories**
- [coqui-ai/TTS](https://github.com/coqui-ai/TTS)
- [suno-ai/bark](https://github.com/suno-ai/bark)
- [WhisperSpeech](https://github.com/collabora/WhisperSpeech)
- [https://github.com/netease-youdao/EmotiVoice](https://github.com/netease-youdao/EmotiVoice)
- [https://github.com/pyannote/pyannote-audio](https://github.com/pyannote/pyannote-audio)
- [https://github.com/alibaba-damo-academy/FunASR](https://github.com/alibaba-damo-academy/FunASR)
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
- **VLMo: Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts**, _Bao et al._, NeurIPS 2022. \[[paper](https://arxiv.org/abs/2111.02358)\]\[[code](https://github.com/microsoft/unilm/tree/master/vlmo)\]
- **BEiT: BERT Pre-Training of Image Transformers**, _Bao et al._, ICLR 2022 Oral presentation. \[[paper](https://arxiv.org/abs/2106.08254)\]\[[code](https://github.com/microsoft/unilm/tree/master/beit)\]
- BeiT-V3: **Image as a Foreign Language: BEiT Pretraining for All Vision and Vision-Language Tasks**, _Wang et al._, CVPR 2023. \[[paper](https://arxiv.org/abs/2208.10442)\]\[[code](https://github.com/microsoft/unilm/tree/master/beit3)\]

### 3. Clip

- CLIP: **Learning Transferable Visual Models From Natural Language Supervision**, _Radford et al._, ICML 2021. \[[paper](https://arxiv.org/abs/2103.00020)\]\[[code](https://github.com/OpenAI/CLIP)\]\[[clip-as-service](https://github.com/jina-ai/clip-as-service)\]\[[open_clip](https://github.com/mlfoundations/open_clip)\]
- DALL-E2: **Hierarchical Text-Conditional Image Generation with CLIP Latents**, _Ramesh et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2204.06125)\]\[[code](https://github.com/lucidrains/DALLE2-pytorch)\]
- **HiCLIP: Contrastive Language-Image Pretraining with Hierarchy-aware Attention**, _Geng et al._, ICLR 2023. \[[paper](https://arxiv.org/abs/2303.02995)\]\[[code](https://github.com/jeykigung/HiCLIP)\]
- **Chinese CLIP: Contrastive Vision-Language Pretraining in Chinese**, _Yang et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2211.01335)\]\[[code](https://github.com/OFA-Sys/Chinese-CLIP)\]
- **Alpha-CLIP: A CLIP Model Focusing on Wherever You Want**, _Sun et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.03818)\]\[[code](https://github.com/SunzeY/AlphaCLIP)\]
- MMVP: **Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs**, _Tong et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.06209)\]\[[code](https://github.com/tsb0601/MMVP)\]

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
- LCM: **Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference**, _Luo et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.04378)\]\[[code](https://github.com/luosiallen/latent-consistency-model)\]
- **LCM-LoRA: A Universal Stable-Diffusion Acceleration Module**, _Luo et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.05556)\]\[[code](https://github.com/luosiallen/latent-consistency-model)\]
- **StreamDiffusion: A Pipeline-level Solution for Real-time Interactive Generation**, _Kodaira et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.12491)\]\[[code](https://github.com/cumulo-autumn/StreamDiffusion)\]
- **Video Diffusion Models**, _Ho et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2204.03458)\]\[[code](https://github.com/lucidrains/video-diffusion-pytorch)\]
- **Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets**, _Blattmann et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.15127)\]\[[code](https://github.com/Stability-AI/generative-models)\]
- **Consistency Models**, _Song et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2303.01469)\]\[[code](https://github.com/openai/consistency_models)\]\[[Consistency Decoder](https://github.com/openai/consistencydecoder)\]
- **A Survey on Video Diffusion Models**, _Xing et al._, srxiv 2023. \[[paper](https://arxiv.org/abs/2310.10647)\]\[[code](https://github.com/ChenHsing/Awesome-Video-Diffusion-Models)\]
- **Diffusion Models: A Comprehensive Survey of Methods and Applications**, _Yang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2209.00796)\]\[[code](https://github.com/YangLing0818/Diffusion-Models-Papers-Survey-Taxonomy)\]
- **Language Model Beats Diffusion -- Tokenizer is Key to Visual Generation**, _Yu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.05737)\]
- **The Chosen One: Consistent Characters in Text-to-Image Diffusion Models**, _Avrahami et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.10093)\]\[[code](https://github.com/ZichengDuan/TheChosenOne)\]
- **UniDiffuser: One Transformer Fits All Distributions in Multi-Modal Diffusion**, _Bao et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2303.06555)\]\[[code](https://github.com/thu-ml/unidiffuser)\]
- l-DAE: **Deconstructing Denoising Diffusion Models for Self-Supervised Learning**, _Chen et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.14404)\]

- **Github Repositories**
- \[[stable-diffusion](https://github.com/CompVis/stable-diffusion)\]
- \[[stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)\]
- \[[stablediffusion](https://github.com/Stability-AI/stablediffusion?tab=readme-ov-file)\]
- \[[Awesome-Diffusion-Models](https://github.com/diff-usion/Awesome-Diffusion-Models)\]
- \[[Fooocus](https://github.com/lllyasviel/Fooocus)\]
- \[[ComfyUI](https://github.com/comfyanonymous/ComfyUI)\]\[[streamlit](https://github.com/streamlit/streamlit)\]\[[gradio](https://github.com/gradio-app/gradio)\]
- \[[diffusers](https://github.com/huggingface/diffusers)\]

### 5. Multimodal LLM

- LLaVA: **Visual Instruction Tuning**, _Liu et al._, NeurIPS 2023 Oral. \[[paper](https://arxiv.org/abs/2304.08485)\]\[[code](https://github.com/haotian-liu/LLaVA)\]

- **A Survey on Multimodal Large Language Models**, Yin et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2306.13549)\]\[[code](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models)\]
- **Flamingo: a Visual Language Model for Few-Shot Learning**, _Alayrac et al._, NeurIPS 2022. \[[paper](https://arxiv.org/abs/2204.14198)\]\[[open-flamingo](https://github.com/mlfoundations/open_flamingo)\]\[[flamingo-pytorch](https://github.com/lucidrains/flamingo-pytorch)\]
- **BuboGPT: Enabling Visual Grounding in Multi-Modal LLMs**, _Zhao et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2307.08581)\]\[[code](https://github.com/magic-research/bubogpt)\]
- **CogVLM: Visual Expert for Pretrained Language Models**, _Wang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.03079)\]\[[code](https://github.com/THUDM/CogVLM)\]\[[VisualGLM-6B](https://github.com/THUDM/VisualGLM-6B)\]
- **DreamLLM: Synergistic Multimodal Comprehension and Creation**, _Dong et al._, ICLR 2024 Spotlight. \[[paper](https://arxiv.org/abs/2309.11499)\]\[[code](https://github.com/RunpeiDong/DreamLLM)\]
- **Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond**, _Bai et al._, arxiv 2023.\[[paper](https://arxiv.org/abs/2308.12966)\]\[[code](https://github.com/QwenLM/Qwen-VL)\]
- **InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks**, _Chen et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.14238)\]\[[code](https://github.com/OpenGVLab/InternVL)\]
- **ShareGPT4V: Improving Large Multi-Modal Models with Better Captions**, _Chen et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.12793)\]\[[code](https://github.com/InternLM/InternLM-XComposer/tree/main/projects/ShareGPT4V)\]
- **TinyGPT-V: Efficient Multimodal Large Language Model via Small Backbones**, _Yuan et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.16862)\]\[[code](https://github.com/DLYuanGod/TinyGPT-V)\]
- Vary-toy: **Small Language Model Meets with Reinforced Vision Vocabulary**, _Wei et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.12503)\]\[[code](https://github.com/Ucas-HaoranWei/Vary-toy)\]

### 6. Text2Image

- DALL-E: **Zero-Shot Text-to-Image Generation**, _Ramesh et al._, arxiv 2021. \[[paper](https://arxiv.org/abs/2102.12092)\]\[[code](https://github.com/openai/DALL-E)\]
- DALL-E3: **Improving Image Generation with Better Captions**, _Betker et al._, OpenAI 2023. \[[paper](https://cdn.openai.com/papers/dall-e-3.pdf)\]\[[code](https://github.com/openai/consistencydecoder)\]\[[blog](https://openai.com/dall-e-3)\]
- ControlNet: **Adding Conditional Control to Text-to-Image Diffusion Models**, _Zhang et al._, ICCV 2023 Marr Prize. \[[paper](https://arxiv.org/abs/2302.05543)\]\[[code](https://github.com/lllyasviel/ControlNet)\]
- **AnyText: Multilingual Visual Text Generation And Editing**, _Tuo et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.03054)\]\[[code](https://github.com/tyxsspa/AnyText)\]
- RPG: **Mastering Text-to-Image Diffusion: Recaptioning, Planning, and Generating with Multimodal LLMs**, _Yang et al._, arxiv 20224. \[[paper](https://arxiv.org/abs/2401.11708)\]\[[code](https://github.com/YangLing0818/RPG-DiffusionMaster)\]
- **LAION-5B: An open large-scale dataset for training next generation image-text models**, _Schuhmann et al._, NeurIPS 2022. \[[paper](https://arxiv.org/abs/2210.08402)\]\[[code](https://github.com/LAION-AI/laion-datasets)\]\[[blog](https://laion.ai/blog/laion-5b/)\]
- Imagen: **Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding**, _Saharia et al._, NeurIPS 2022. \[[paper](https://arxiv.org/abs/2205.11487)\]\[[unofficial code](https://github.com/lucidrains/imagen-pytorch)\]
- **Instruct-Imagen: Image Generation with Multi-modal Instruction**, _Hu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.01952)\]
- **TextDiffuser: Diffusion Models as Text Painters**, _Chen et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.10855)\]\[[code](https://github.com/microsoft/unilm/tree/master/textdiffuser)\]
- **TextDiffuser-2: Unleashing the Power of Language Models for Text Rendering**, _Chen et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.16465)\]\[[code](https://github.com/microsoft/unilm/tree/master/textdiffuser-2)\]
- **PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis**, _Chen et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.00426)\]\[[code](https://github.com/PixArt-alpha/PixArt-alpha)\]
- **PIXART-δ: Fast and Controllable Image Generation with Latent Consistency Models**, _Chen et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.05252)\]\[[code](https://github.com/PixArt-alpha/PixArt-alpha)\]
- **IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models**, _Ye et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2308.06721)\]\[[code](https://github.com/tencent-ailab/IP-Adapter)\]

### 7. Text2Video

- **Animate Anyone: Consistent and Controllable Image-to-Video Synthesis for Character Animation**, _Hu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.17117)\]\[[code](https://github.com/HumanAIGC/AnimateAnyone)\]\[[Open-AnimateAnyone](https://github.com/guoqincode/Open-AnimateAnyone)\]\[[Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone)\]
- **DreaMoving: A Human Video Generation Framework based on Diffusion Models**, _Feng et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.05107)\]\[[code](https://github.com/dreamoving/dreamoving-project)\]
- **MagicAnimate:Temporally Consistent Human Image Animation using Diffusion Model**, _Xu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.16498)\]\[[code](https://github.com/magic-research/magic-animate)\]
- **FlowVid: Taming Imperfect Optical Flows for Consistent Video-to-Video Synthesis**, _Liang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.17681)\]\[[code](https://github.com/Jeff-LiangF/FlowVid)\]

- \[[Awesome-Video-Diffusion](https://github.com/showlab/Awesome-Video-Diffusion)\]
- **Video Diffusion Models**, _Ho et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2204.03458)\]\[[video-diffusion-pytorch](https://github.com/lucidrains/video-diffusion-pytorch)\]
- **Make-A-Video: Text-to-Video Generation without Text-Video Data**, _Singer et al._, arxiv 2022. \[[paper](https://arxiv.org/abs/2209.14792)\]\[[make-a-video-pytorch](https://github.com/lucidrains/make-a-video-pytorch)\]
- **Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation**, _Wu et al._, ICCV 2023. \[[paper](https://arxiv.org/abs/2212.11565)\]\[[code](https://github.com/showlab/Tune-A-Video)\]
- **Text2Video-Zero: Text-to-Image Diffusion Models are Zero-Shot Video Generators**, _Khachatryan et al._, ICCV 2023 Oral. \[[paper](https://arxiv.org/abs/2303.13439)\]\[[code](https://github.com/Picsart-AI-Research/Text2Video-Zero)\]

- **AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning**, _Guo et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2307.04725)\]\[[code](https://github.com/guoyww/AnimateDiff)\]
- **I2VGen-XL: High-Quality Image-to-Video Synthesis via Cascaded Diffusion Models**, _Zhang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.04145)\]\[[code](https://github.com/ali-vilab/i2vgen-xl)\]
- TF-T2V: **A Recipe for Scaling up Text-to-Video Generation with Text-free Videos**, _Wang et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.15770)\]\[[code](https://github.com/ali-vilab/i2vgen-xl)\]
- **Lumiere: A Space-Time Diffusion Model for Video Generation**, _Bar-Tal et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.12945)\]

### 8. Survey for Multimodal

### 9. Other

- **Otter: A Multi-Modal Model with In-Context Instruction Tuning**, _Li et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.03726)\]\[[code](https://github.com/Luodian/Otter)\]
- **Fuyu-8B: A Multimodal Architecture for AI Agents** _Bavishi et al._, Adept blog 2023. \[[blog](https://www.adept.ai/blog/fuyu-8b)\]\[[model](https://huggingface.co/adept/fuyu-8b)\]
- **OtterHD: A High-Resolution Multi-modality Model**, _Li et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2311.04219)\]\[[code](https://github.com/Luodian/Otter)\]\[[model](https://huggingface.co/Otter-AI/OtterHD-8B)\]
- CM3leon: **Scaling Autoregressive Multi-Modal Models_Pretraining and Instruction Tuning**, _Yu et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2309.02591)\]\[[Unofficial Implementation](https://github.com/kyegomez/CM3Leon)\]

---

## Reinforcement Learning

### 1.Basic for RL

- PPO: **Proximal Policy Optimization Algorithms**, _Schulman et al._, arxiv 2017. \[[paper](https://arxiv.org/abs/1707.06347)\]\[[code](https://github.com/openai/baselines)\]\[[PPO-PyTorch](https://github.com/nikhilbarhate99/PPO-PyTorch)\]\[[implementation-matters](https://github.com/MadryLab/implementation-matters)\]

### 2. LLM for decision making 

- **Decision Transformer_Reinforcement Learning via Sequence Modeling**, _Chen et al._, NeurIPS 2021. \[[paper](https://arxiv.org/abs/2106.01345)\]\[[code](https://github.com/kzl/decision-transformer)\]
- Trajectory Transformer: **Offline Reinforcement Learning as One Big Sequence Modeling Problem**, _Janner et al._, NeurIPS 2021. \[[paper](https://arxiv.org/abs/2106.02039)\]\[[code](https://github.com/JannerM/trajectory-transformer)\]
- **Guiding Pretraining in Reinforcement Learning with Large Language Models**, _Du et al._, ICML 2023. \[[paper](https://arxiv.org/abs/2302.06692)\]\[[code](https://github.com/yuqingd/ellm)\]
- **Introspective Tips: Large Language Model for In-Context Decision Making**, _Chen et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.11598)\]
- **Q-Transformer: Scalable Offline Reinforcement Learning via Autoregressive Q-Functions**, _Chebotar et al._, CoRL 2023. \[[paper](https://arxiv.org/abs/2309.10150)\]\[[Unofficial Implementation](https://github.com/lucidrains/q-transformer)\]

---

## GNN
 
- **Do Transformers Really Perform Bad for Graph Representation**, _Ying et al._, NeurIPS 2021. \[[paper](https://arxiv.org/abs/2106.05234)\]\[[code](https://github.com/Microsoft/Graphormer)\]

### Survey for GNN

---

## Transformer Architecture

- **Attention is All you Need**, _Vaswani et al._, NIPS 2017. \[[paper](https://arxiv.org/abs/1706.03762)\]\[[code](https://github.com/jadore801120/attention-is-all-you-need-pytorch)\]\[[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)\]\[[The Random Transformer](https://osanseviero.github.io/hackerllama/blog/posts/random_transformer/)\]\[[The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)\]\[[Transformers-Tutorials](https://github.com/NielsRogge/Transformers-Tutorials)\]\[[x-transformers](https://github.com/lucidrains/x-transformers)\]
- RoPE: **RoFormer: Enhanced Transformer with Rotary Position Embedding**, _Su et al._, arxiv 2021. \[[paper](https://arxiv.org/abs/2104.09864)\]\[[code](https://github.com/ZhuiyiTechnology/roformer)\]\[[rotary-embedding-torch](https://github.com/lucidrains/rotary-embedding-torch)\]\[[blog](https://kexue.fm/archives/9675)\]
- **RWKV: Reinventing RNNs for the Transformer Era**, _Peng et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2305.13048)\]\[[code](https://github.com/BlinkDL/RWKV-LM)\]
- **Mamba: Linear-Time Sequence Modeling with Selective State Spaces**, _Gu and Dao_, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.00752)\]\[[code](https://github.com/state-spaces/mamba)\]


