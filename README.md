# Awesome-Prompt-Learning-CV
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

This repository is a collection of awesome things about **Prompt Learning**, including papers, code, etc.

If you would like to contribute to our repository or have any questions/advice, see [Contributing & Contact](#contributing--contact).

# Contents
- [Awesome-Prompt-Learning-CV](#Awesome-Prompt-Learning-CV)
- [Contents](#contents)
- [Papers](#papers)
- [Publications](#publications)
- [Contributing \& Contact](#contributing--contact)

# Papers
> We list papers, implementation code (the unofficial code is marked with *), etc, in the order of year and from journals to conferences.


### Prompt Learning(accepted): *Updating by Linbin Wang*
* **CoOp**: Learning to Prompt for Vision-Language Models (*NTU, Singapore*) [[IJCV 2022](https://arxiv.org/pdf/2109.01134)] [[PyTorch](https://github.com/KaiyangZhou/CoOp)]
* **CoCoOp**: Conditional Prompt Learning for Vision-Language Models (*NTU, Singapore*) [[CVPR 2022](http://openaccess.thecvf.com/content/CVPR2022/papers/Zhou_Conditional_Prompt_Learning_for_Vision-Language_Models_CVPR_2022_paper.pdf)] [[Code](https://github.com/KaiyangZhou/CoOp)]
* **MaPLe**: Multi-modal Prompt Learning [[CVPR 2023](https://arxiv.org/pdf/2210.03117)] [[Code](https://github.com/muzairkhattak/multimodal-prompt-learning)]
* **ProDA**: Prompt Distribution Learning, CVPR, 2022 (*Huawei*). [[Paper](https://arxiv.org/abs/2205.03340)]
* **VPT**: Visual Prompt Tuning, ECCV, 2022 (*Cornell*). [[Paper](https://arxiv.org/abs/2203.12119)][[PyTorch](https://github.com/kmnp/vpt)]
* **PerVL**: This is my unicorn, Fluffy: Personalizing frozen vision-language representations, ECCV, 2022 (*NVIDIA*). [[Paper](https://arxiv.org/abs/2204.01694)][[PyTorch](https://github.com/NVlabs/PALAVRA)]
* **OrdinalCLIP**: OrdinalCLIP: Learning Rank Prompts for Language-Guided Ordinal Regression, NeurIPS, 2022 (*Tsinghua University*). [[Paper](https://arxiv.org/abs/2206.02338)][[PyTorch](https://github.com/xk-huang/OrdinalCLIP)]
* **CoOp**: Learning to Prompt for Vision-Language Models, IJCV, 2022 (*NTU, Singapore*). [[Paper](https://arxiv.org/abs/2109.01134)][[PyTorch](https://github.com/KaiyangZhou/CoOp)]
* **DeFo**: Learning to Decompose Visual Features with Latent Textual Prompts, ICLR, 2023 (*UIUC*). [[Paper](https://arxiv.org/abs/2210.04287)]
* **PLOT**: Prompt Learning with Optimal Transport for Vision-Language Models, ICLR, 2023 (*CMU*). [[Paper](https://arxiv.org/abs/2210.01253)]
* **?**: Visual Classification via Description from Large Language Models, ICLR, 2023 (*Columbia*). [[Paper](https://arxiv.org/abs/2210.07183)]
* **CSP**: Learning to Compose Soft Prompts for Compositional Zero-Shot Learning, ICLR, 2023 (*Brown University*). [[Paper](https://arxiv.org/abs/2204.03574)][[PyTorch](https://github.com/BatsResearch/csp)]
* **CaFo**: Prompt, Generate, then Cache: Cascade of Foundation Models makes Strong Few-shot Learners, CVPR, 2023 (*Shanghai AI Lab*). [[Paper](https://arxiv.org/abs/2303.02151)][[PyTorch](https://github.com/ZrrSkywalker/CaFo)]
* **?**: Multimodal Prompting with Missing Modalities for Visual Recognition, CVPR, 2023 (*NYCU*). [[Paper](https://arxiv.org/abs/2303.03369)][[PyTorch (in construction)](https://github.com/YiLunLee/Missing_aware_prompts)][[Website](https://yilunlee.github.io/missing_aware_prompts/)]
* **DAM-VP**: Diversity-Aware Meta Visual Prompting, CVPR, 2023 (*USTC*). [[Paper](https://arxiv.org/abs/2303.08138)][[Code (in construction)](https://github.com/shikiw/DAM-VP)]
* **ILM-VP**: Understanding and Improving Visual Prompting: A Label-Mapping Perspective, CVPR, 2023 (*Michigan State*). [[Paper](https://arxiv.org/abs/2211.11635)][[PyTorch](https://github.com/OPTML-Group/ILM-VP)]
* **KgCoOp**: Visual-Language Prompt Tuning with Knowledge-guided Context Optimization, CVPR, 2023 (*CAS*). [[Paper](https://arxiv.org/abs/2303.13283)][[PyTorch](https://github.com/htyao89/KgCoOp)]
* **BlackVIP**: BlackVIP: Black-Box Visual Prompting for Robust Transfer Learning, CVPR, 2023 (*University of Seoul*). [[Paper](https://arxiv.org/abs/2303.14773)][[PyTorch (in construction)](https://github.com/changdaeoh/BlackVIP)]
* **EXPRES**: Learning Expressive Prompting With Residuals for Vision Transformers, CVPR, 2023 (*Amazon*). [[Paper](https://arxiv.org/abs/2303.15591)]
* **?**: Learning to Name Classes for Vision and Language Models, CVPR, 2023 (*Huawei*). [[Paper](https://arxiv.org/abs/2304.01830)]
* **PMF**: Efficient Multimodal Fusion via Interactive Prompting, CVPR, 2023 (*Zhejiang University*). [[Paper](https://arxiv.org/abs/2304.06306)]
* **MaPLe**: MaPLe: Multi-modal Prompt Learning, CVPR, 2023 (*MBZUAI*). [[Paper](https://arxiv.org/abs/2210.03117)][[PyTorch](https://github.com/muzairkhattak/multimodal-prompt-learning)]
* **POUF**: POUF: Prompt-oriented unsupervised fine-tuning for large pre-trained models, ICML, 2023 (*UT Austin*). [[Paper](https://arxiv.org/abs/2305.00350)][[PyTorch](https://github.com/korawat-tanwisuth/POUF)]
* **TPT**: Test-Time Prompt Tuning for Zero-Shot Generalization in Vision-Language Models, (*UMD*). [[NIPS 2022](https://arxiv.org/abs/2209.07511)][[PyTorch]( https://azshue.github.io/TPT/)]
* **K-LITE**：K-LITE: Learning Transferable Visual Models with External Knowledge(*Microsoft*) [[NeurIPS 2022](https://openreview.net/forum?id=gERv_uy69IA)][[PyTorch](https://github.com/microsoft/klite)]
* **CALIP**：CALIP: Zero-Shot Enhancement of CLIP with Parameter-free Attention(*Peking University*) [[AAAI 2023](https://arxiv.org/abs/2209.14169)][[PyTorch](https://github.com/ziyuguo99/calip)]
* **ProReg**：Debiased Fine-Tuning for Vision-Language Models by Prompt Regularization(*Nanyang Technological University*)[[AAAI 2023](https://arxiv.org/abs/2301.12429)]
* **NLIP**：NLIP: Noise-robust Language-Image Pre-training(*Sun Yat-sen University*)[[AAAI 2023](https://arxiv.org/abs/2212.07086)]
* **ZS-SBIR**：CLIP for All Things Zero-Shot Sketch-Based Image Retrieval,Fine-Grained or Not(*University of Surrey*)[[CVPR 2023](https://arxiv.org/abs/2303.13440)][[PyTorch](https://github.com/aneeshan95/Sketch_LVM)]
* **?**：Multimodality Helps Unimodality:Cross-Modal Few-Shot Learning with Multimodal Models(*Carnegie Mellon University*)[[CVPR 2023](https://arxiv.org/abs/2301.06267)][[PyTorch](https://github.com/linzhiqiu/cross_modal_adaptation)]
* **SP**：Semantic Prompt for Few-Shot Image Recognition(*University of Science and Technology of China*)[[CVPR 2023](https://arxiv.org/abs/2303.14123)][[PyTorch](https://github.com/WentaoChen0813/SemanticPrompt)]
* **CPT**：CONTRASTIVE PROMPT TUNING IMPROVES GENERALIZATION IN VISION-LANGUAGE MODELS(*MIT-IBM Watson AI Lab*)[[ICLR 2023](https://openreview.net/profile?id=~Aadarsh_Sahoo1)]
* **MixGen**：MixGen: A New Multi-Modal Data Augmentation(*Institute of Information Engineering, CAS*)[[CVPR 2023](https://arxiv.org/abs/2206.08358)][[PyTorch](https://github.com/amazon-science/mix-generation)]
* **SLIP**：SLIP: Self-supervision meets Language-Image Pre-training(*UC Berkeley*)[[ECCV 2022](https://arxiv.org/abs/2112.12750)][[PyTorch](https://github.com/facebookresearch/SLIP)]
* **?**：Rethinking the Value of Prompt Learning for Vision-Language Models(*Chinese Academy of Sciences*)[[ICLR 2023](https://openreview.net/forum?id=1FsdIfRngtw)]
* **SubPT**：Understanding and Mitigating Overfitting in Prompt Tuning for Vision-Language Models(*Chinese Academy of Sciences*)[[IEEE 2023](https://arxiv.org/abs/2211.02219)][[PyTorch](https://github.com/machengcheng2016/Subspace-Prompt-Learning)]
* **CLIP-like**：Delving into the Openness of CLIP(*Peking University*)[[ICLR 2023](https://openreview.net/forum?id=9OoFFWDPDQ)][[Code](https://github.com/lancopku/clip-openness)]
* **PTP**：Position-guided Text Prompt for Vision-Language Pre-training(*National University of Singapore*)[[IEEE 2022](https://arxiv.org/abs/2212.09737)][[PyTorch](https://github.com/sail-sg/ptp)]
* **DPT**: Dual Modality Prompt Tuning for Vision-Language Pre-Trained Model(*Northwestern Polytechnical University*) [[TMM 2023](https://arxiv.org/abs/2208.08340)][[Code](https://github.com/fanrena/dpt)]
* **PROTO-CLIP**:PROTO-CLIP: Vision-Language Prototypical Network for Few-Shot Learning(*The University of Texas at Dallas*)[[arXiv 2307](https://arxiv.org/abs/2307.03073)][[Code](https://github.com/IRVLUTD/Proto-CLIP)]

### Prompt Learning(arxiv): *Updating by Jiajia Zhang*

* **LAMM**: Label Alignment for Multi-Modal Prompt Learning(*Shanghai Jiao Tong University*) [[arXiv 2312](https://arxiv.org/abs/2312.08212)] [[Pytorch](https://github.com/gaojingsheng/LAMM)]
* **?**: Align Your Prompts: Test-Time Prompting with Distribution Alignment for Zero-Shot Generalization(*Mohamed Bin Zayed University of AI*) [[arXiv 2311](https://arxiv.org/abs/2311.01459)] [[Pytorch](https://github.com/jameelhassan/PromptAlign)]
* **FD-Align**: Feature Discrimination Alignment for Fine-tuning Pre-Trained Models in Few-Shot Learning (*SCCE*) [[arXiv 2310](https://arxiv.org/abs/2310.15105)] [[Pytorch](https://github.com/skingorz/FD-Align)]
* **?**: Prompting Scientific Names for Zero-Shot Species Recognition(*Texas A&M University*) [[arXiv 2310](https://arxiv.org/abs/2310.09929)]
* **CXR-CLIP**: Toward Large Scale Chest X-ray Language-Image Pre-training (*Kakaobrain*) [[arXiv 2310](https://arxiv.org/abs/2310.13292)]
* **FGPrompt**: Fine-grained Goal Prompting for Image-goal Navigation(*South China University of Technology*) [[arXiv 2310](https://arxiv.org/abs/2310.07473)]
* **DA-CLIP**: Controlling Vision-Language Models for Universal Image Restoration(*Uppsala University*) [[arXiv 2310](https://arxiv.org/abs/2310.01018)]
* **FedTPG**: Text-driven Prompt Generation for Vision-Language Models in Federated Learning(*Bosch Center for AI*) [[arXiv 2310](https://arxiv.org/abs/2310.06123)]
* **AutoCLIP**: Auto-tuning Zero-Shot Classifiers for Vision-Language Models(*Bosch Center for Artificial Intelligence*) [[arXiv 2309](https://arxiv.org/abs/2309.16414)]
* **?**: Tuning Multi-mode Token-level Prompt Alignment across Modalities(*Xidian University*) [arXiv 2309](https://arxiv.org/pdf/2309.13847)]
* **BPT**：CLIP-based Synergistic Knowledge Transfer for Text-based Person Retrieval(*Shenzhen International Graduate Schoo*) [[arXiv 2309](https://arxiv.org/abs/2309.09496)]
* **?**：Language Models as Black-Box Optimizers for Vision-Language Models(*CMU*) [[arXiv 2309](https://arxiv.org/abs/2309.05950)]
* **PRE**: Vision-Language Prompt Learning with Reparameterization Encoder(*QMUL*)[[arXiv 2309](https://arxiv.org/abs/2309.07760)] [[Pytorch](https://github.com/minhanh151/respro)]
* **LoGoPrompt**: Synthetic Text Images Can Be Good Visual Prompts for Vision-Language Models(*School of Information Science and Technology, ShanghaiTech University*)[[arXiv 2309](https://arxiv.org/abs/2309.01155)]
* **?**: Image-Object-Specific Prompt Learning for Few-Shot Class-Incremental Learning(*Kaist Advanced Institute of Science and Technology*) [[arXiv 2309](https://arxiv.org/abs/2309.02833)]
* **DuAl-PT**: Context-Aware Prompt Tuning for Vision-Language Model with Dual-Alignment(*Shanghai Jiao Tong University*) [[arXiv 2309](https://arxiv.org/abs/2309.04158)]
* **DPL**: Decoupled Prompt Learning for Vision-Language Models(*Nanjing University*) [[arXiv 2308](https://arxiv.org/abs/2308.10061)]
* **ALIP**: Adaptive Language-Image Pre-training with Synthetic Caption(*DeepGlint*) [[arXiv 2308](https://arxiv.org/abs/2308.08428)] [[Pytorch](https://github.com/deepglint/alip)]
* **ICPC**: Instance-Conditioned Prompting with Contrastive Learning for Semantic Segmentation(*Alibaba Group*) [[arXiv 2308](https://arxiv.org/abs/2308.07078)]
* **UEO**: Towards Realistic Unsupervised Fine-tuning with CLIP(*CASIA*) [[arXiv 2308](https://arxiv.org/abs/2308.12919)]
* **UP-Adapter**: Unsupervised Prototype Adapter for Vision-Language Models(*Southern University of Science and Technology*) [[arXiv 2308](https://arxiv.org/abs/2308.11507)]
* **KAPT**: Knowledge-Aware Prompt Tuning for Generalizable Vision-Language Models(*Qilu University of Technology*) [[arXiv 2308](https://arxiv.org/abs/2308.11186)]
* **RPO**：Read-only Prompt Optimization for Vision-Language Few-shot Learning(*Korea University*)[[arXiv 2308](https://arxiv.org/abs/2308.14960)]
* **UP-Adapter**：Unsupervised Prototype Adapter for Vision-Language Models(*Southern University of Science and Technology*)[[arXiv 2308](https://arxiv.org/abs/2308.11507)][[PyTorch](https://github.com/mlvlab/RPO)]
* **PromptSRC**: Self-regulating Prompts: Foundational Model Adaptation without Forgetting(*Mohamed bin Zayed University of AI*) [[arXiv 2307](https://arxiv.org/abs/2307.06948)] [[Pytorch](https://github.com/muzairkhattak/promptsrc)]
* **VDP-Adapter** :Enhancing CLIP with GPT-4 Harnessing Visual Descriptions as Prompts (*Dublin City University*)[[arXiv 2307](https://arxiv.org/abs/2307.11661)] [[Pytorch](https://github.com/mayug/VDT-Adapter)]
* **ICAE**: In-context Autoencoder for Context Compression in a Large Language Model(*Microsoft*)[[arXiv 2307](https://arxiv.org/abs/2307.06945)]
* **?**: Leveraging Vision-Language Foundation Models for Fine-Grained Downstream Tasks(*Conservatoire National des Arts et Métiers, CEDRIC, Paris, France*) [[arXiv 2307](https://arxiv.org/abs/2307.06795)]
* **?**: Unsupervised Calibration through Prior Adaptation for Text Classification using Large Language Models(*Instituto de Investigaci´on en Ciencias de la Computaci´on, CONICET-UBA, Argentina*) [[arXiv 2307](https://arxiv.org/abs/2307.06713)]
* **LDP**: Language-driven Dual-Pixel Image Defocus Deblurring Network(*Beijing Institute of Technology*) [[arXiv 2307](https://arxiv.org/abs/2307.09815)]
* **DVPT**: Dynamic Visual Prompt Tuning of Large Pre-trained Models for Medical Image Analysis(*Nankai University*) [[arXiv 2307](https://arxiv.org/abs/2307.09787)]
* **MoP-CLIP**: A Mixture of Prompt-Tuned CLIP Models for Domain Incremental Learning(*Ola Ahmad*) [[arXiv 2307](https://arxiv.org/abs/2307.05707)]
* **MuDPT**: Multi-modal Deep-symphysis Prompt Tuning for Large Pre-trained Vision-Language Models (*College of Computer Science and Technology,changsha*) [[arXiv 2306](https://arxiv.org/abs/2306.11400)]
* **DIFFender**: Diffusion-Based Adversarial Defense against Patch Attacks in the Physical World (*Institute of Artificial Intelligence, Beihang University*) [[arXiv 2306](https://arxiv.org/abs/2306.09124)]
* **?**: Soft-prompt Tuning for Large Language Models to Evaluate Bias(*Vector Institute for AI*) [[arXiv 2306](https://arxiv.org/abs/2306.04735)]
* **TKDP**: Threefold Knowledge-enriched Deep Prompt Tuning for Few-shot Named Entity Recognition(*JOURNAL OF LATEX CLASS FILES*) [[arXiv 2306](https://arxiv.org/abs/2306.03974)]
* **ProTeCt**: Prompt Tuning for Hierarchical Consistency(*Department of Electrical and Computer Engineering
University of California*)[[arXiv 2306](https://arxiv.org/abs/2306.02240)]
* **LLaVAR**: Enhanced Visual Instruction Tuning for Text-Rich Image Understanding(*Georgia Tech*)[[arXiv 2306](https://arxiv.org/abs/2306.17107)]
* **NPT**: Bridging the Gap: Neural Collapse Inspired Prompt Tuning for Generalization under Class Imbalance(*Zhejiang University*)[[arXiv 2306](https://arxiv.org/abs/2306.15955)]
* **SADA**:  Few-Shot Learning with Visual Distribution Calibration and Cross-Modal Distribution Alignment(*Beihang University*)[arXiv 2305](https://arxiv.org/abs/2305.11439)][Pytorch](https://github.com/bhrqw/sada)]
* **?**: Residual Prompt Tuning: Improving Prompt Tuning with Residual Reparameterization(*Meta AI*) [[arXiv 2305](https://arxiv.org/abs/2305.03937)] [[Pytorch](https://github.com/arazd/residualprompts)]
* **BSL**: Black-box Prompt Tuning with Subspace Learning (*Tsinghua University*) [[arXiv 2305](https://arxiv.org/abs/2305.03518)]
* **PTP**: Boosting Stability and Performance of Prompt Tuning with Perturbation-Based Regularizer (*University of Maryland*) [[arXiv 2305](https://arxiv.org/abs/2305.02423)]
* **Instruction-ViT**: Instruction-ViT: Multi-Modal Prompts for Instruction Learning in ViT (*University of Electronic Science and Technology of China*). [[arXiv 2305](https://arxiv.org/abs/2305.00201)]
* **VPGTrans**: Transfer Visual Prompt Generator across LLMs (*NUS*). [[arXiv 2305](https://arxiv.org/abs/2305.01278)][[PyTorch](https://github.com/VPGTrans/VPGTrans)][[Website](https://vpgtrans.github.io/)]
* **DRPT**: DRPT: Disentangled and Recurrent Prompt Tuning for Compositional Zero-Shot Learning (*Hong Kong Polytechnic University*). [[arXiv 2305](https://arxiv.org/abs/2305.01239)][[Code (in construction)](https://github.com/Forest-art/DRPT-torch)]
* **VCoT**: Visual Chain of Thought: Bridging Logical Gaps with Multimodal Infillings (*UCSB*). [[arXiv 2305](https://arxiv.org/abs/2305.02317)]
* **PMPO**: Multi-Prompt with Depth Partitioned Cross-Modal Learning (*CAS*). [[arXiv 2305](https://arxiv.org/abs/2305.06221)]
* **Aurora**: Mode Approximation Makes Good Vision-Language Prompts (*Peking*). [[2305](https://arxiv.org/abs/2305.08381)][[PyTorch](https://github.com/WillDreamer/Aurora)]
* **DSD**: Discriminative Diffusion Models as Few-shot Vision and Language Learners (*Google*). [[arXiv 2305](https://arxiv.org/abs/2305.10722)]
* **APE**: Not All Features Matter: Enhancing Few-shot CLIP with Adaptive Prior Refinement(*City University of Hong Kong*) [[arXiv 2304](https://arxiv.org/abs/2304.01195)] [[Pytorch](https://github.com/yangyangyang127/ape)]
* **XSGD**: Efficiently Aligned Cross-Lingual Transfer Learning for Conversational Tasks using Prompt-Tuning (*Salesforce AI*) [[arXiv 2304](https://arxiv.org/abs/2304.01295v1)]
* **SAQI**: Towards Rebust Text-Prompted Semantic Criterion for In-the-Wild Video Quality Assessment(*IEEE*) [[arXiv 2304](https://arxiv.org/abs/2304.14672)] [[Pytorch](https://github.com/vqassessment/bvqi)]
* **D2CSE**: Difference-aware Deep continuous prompts for Contrastive Sentence Embeddings(*Samsung SDS*) [[arXiv 2304](https://arxiv.org/abs/2304.08991)]
* **IDPT**: Instance-aware Dynamic Prompt Tuning for Pre-trained Point Cloud Models(*Tsinghua University*) [[arXiv 2304](https://arxiv.org/abs/2304.07221)] [[Pytorch](https://github.com/zyh16143998882/IDPT)]
* **AutoSplice**: A Text-prompt Manipulated Image Dataset for Media Forensics(*University at Buffalo*) [[arXiv 2304](https://arxiv.org/abs/2304.06870)] [[Pytorch](https://github.com/shanface33/autosplice_dataset)]
* **POMP**: Prompt Pre-Training with Twenty-Thousand Classes for Open-Vocabulary Visual Recognition (*Amazon*). [[arXiv 2304](https://arxiv.org/abs/2304.04704)][[PyTorch](https://github.com/amazon-science/prompt-pretraining)]
* **?**: What does CLIP know about a red circle? Visual prompt engineering for VLMs (*Oxford*). [[arXiv 2304](https://arxiv.org/abs/2304.06712)]
* **Robust-ProL**: Towards Robust Prompts on Vision-Language Models (*Google*). [[arXiv 2304](https://arxiv.org/abs/2304.08479)]
* **ProVP**: Progressive Visual Prompt Learning with Contrastive Feature Re-formation (*vivo, China*). [[arXiv 2304](https://arxiv.org/abs/2304.08386)]
* **?**: Chain of Thought Prompt Tuning in Vision Language Models (*Peking University*). [[arXiv 2304](https://arxiv.org/abs/2304.07919)]
* **LION**：LION: Implicit Vision Prompt Tuning(*Peking University*)[[arXiv 2303](https://arxiv.org/abs/2303.09992)]
* **SeMap**: From Visual Prompt Learning to Zero-Shot Transfer: Mapping Is All You Need (*CISPA, Germany*). [[arXiv 2303](https://arxiv.org/abs/2303.05266)]
* **R-Tuning**: R-Tuning: Regularized Prompt Tuning in Open-Set Scenarios (*Shanghai Jiao Tong*). [[arXiv 2303](https://arxiv.org/abs/2303.05122)]
* **VPTM**: Rethinking Visual Prompt Learning as Masked Visual Token Modeling (*Shanghai Jiao Tong*). [[arXiv 2303](https://arxiv.org/abs/2303.04998)]
* **GRAM**: Gradient-Regulated Meta-Prompt Learning for Generalizable Vision-Language Models (*Huawei*). [[arXiv 2303](https://arxiv.org/abs/2303.06571)]
* **PBPrompt**: Patch-Token Aligned Bayesian Prompt Learning for Vision-Language Models (*Xidian University*). [[arXiv 2303](https://arxiv.org/abs/2303.09100)]
* **CTP-TFT**: Task-Oriented Multi-Modal Mutual Leaning for Vision-Language Models (*Baidu*). [[arXiv 2303](https://arxiv.org/abs/2303.17169)]
* **ZPE**: A Simple Zero-shot Prompt Weighting Technique to Improve Prompt Ensembling in Text-Image Models (*Google*). [[arXiv 2302](https://arxiv.org/abs/2302.06235)]
* **?**: Task Bias in Vision-Language Models (*Columbia*). [[arXiv 2212](https://arxiv.org/abs/2212.04412)]
* **?**：CLIP Itself is a Strong Fine-tuner: Achieving 85.7% and 88.0% Top-1 Accuracy with ViT-B and ViT-L on ImageNet(*University of Science and Technology of China*)[[arXiv 2212](https://arxiv.org/abs/2212.06138)][[PyTorch](https://github.com/lightdxy/ft-clip)]
* **?**：Unleashing the Power of Visual Prompting At the Pixel Level(*Shanghai Jiao Tong University*)[[arXiv 2212](https://arxiv.org/abs/2212.10556)][[Code](https://github.com/ucsc-vlaa/evp)]
* **TaskRes**: Task Residual for Tuning Vision-Language Models (*NUS*). [[arXiv 2211](https://arxiv.org/abs/2211.10277)][[Code (in construction)](https://github.com/geekyutao/TaskRes)]
* **MVLPT**: Multitask Vision-Language Prompt Tuning (*Berkeley*). [[arXiv 2211](https://arxiv.org/abs/2211.11720)][[PyTorch](https://github.com/sIncerass/MVLPT)]
* **TaI-DP**: Texts as Images in Prompt Tuning for Multi-Label Image Recognition (*Tomorrow Advancing Life (TAL)*). [[arXiv 2211](https://arxiv.org/abs/2211.12739)][[PyTorch](https://github.com/guozix/TaI-DPT)]
* **?**：Bayesian Prompt Learning for Image-Language Model Generalization(*University of Amsterdam*)[[arXiv 2210](https://arxiv.org/abs/2210.02390)]
* **PGN**: Prompt Generation Networks for Efficient Adaptation of Frozen Vision Transformers (*University of Amsterdam*). [[arXiv 2210](https://arxiv.org/abs/2210.06466)][[PyTorch](https://github.com/jochemloedeman/PGN)]
* **UPT**: Unified Vision and Language Prompt Learning (*NTU, Singapore*). [[arXiv 2210](https://arxiv.org/abs/2210.07225)][[Code (in construction)](https://github.com/yuhangzang/UPT)]
* **CPL**: CPL: Counterfactual Prompt Learning for Vision and Language Models (*UC Santa Cruz*). [[arXiv 2210](https://arxiv.org/abs/2210.10362)]
* **PTP**: Prompting through Prototype: A Prototype-based Prompt Learning on Pretrained Vision-Language Models (*Baidu*). [[arXiv 2210](https://arxiv.org/abs/2210.10841)]
* **LASP**: Language-Aware Soft Prompting for Vision & Language Foundation Models (*Samsung*). [[arXiv 2210](https://arxiv.org/abs/2210.01115)][[Website](https://www.adrianbulat.com/lasp)]
* **VPT**: Variational prompt tuning improves generalization of vision-language models (*Samsung*). [[arXiv 2210](https://arxiv.org/abs/2210.02390)]
* **DoPrompt**：Prompt Vision Transformer for Domain Generalization(*National University of Singapore*)[[arXiv 2208](https://arxiv.org/abs/2208.08914)][[PyTorch](https://github.com/zhengzangw/DoPrompt)]
* **SoftCPT**：Prompt Tuning with Soft Context Sharing for Vision-Language Models(*National Laboratory of Pattern Recognition*)[[arXiv 2208](https://arxiv.org/abs/2208.13474)][[Code](https://github.com/kding1225/softcpt)]
* **?**：Prompt-to-Prompt Image Editing with Cross Attention Control(*Google Research*)[[arXiv 2208](https://arxiv.org/abs/2208.01626)][[Code](https://github.com/google/prompt-to-prompt)]
* **DPT**：Dual Modality Prompt Tuning for Vision-Language(*Chinese Academy of Sciences*)[[arXiv 2208](https://arxiv.org/abs/2208.08340)][[PyTorch](https://github.com/fanrena/dpt)]
* **CAVPT**: Class-Aware Visual Prompt Tuning for Vision-Language Pre-Trained Model (*Northwestern Polytechnical University, China*). [[arXiv 2208](https://arxiv.org/abs/2208.08340)][[Code](https://github.com/fanrena/dpt)]
* **ProGrad**：Prompt-aligned Gradient for Prompt Tuning(*Nanyang Technological University*)[[arXiv 2205](https://arxiv.org/abs/2205.14865)][[PyTorch](https://github.com/beierzhu/prompt-align)]
* **UPL**：Unsupervised Prompt Learning for Vision-Language Models(*Peking University*)[[arXiv 2204](https://arxiv.org/abs/2204.03649)][[Code](https://github.com/tonyhuang2022/UPL)]
* **Visual-Prompting**: Exploring Visual Prompts for Adapting Large-Scale Models (*MIT*). [[arXiv 2203](https://arxiv.org/abs/2203.17274)][[PyTorch](https://github.com/hjbahng/visual_prompting)][[Website](https://hjbahng.github.io/visual_prompting/)]
* **DAPL**：Domain Adaptation via Prompt Learning(*Tsinghua University*)[[arXiv 2202](https://arxiv.org/abs/2202.06687)][[Code](https://github.com/LeapLabTHU/DAPrompt)]
* **OOD**：Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution(*Stanford University*) [[arxiv 2202](https://arxiv.org/abs/2202.10054)][[PyTorch](https://github.com/Aaditya-Singh/Low-Shot-Robustness)]
* **AP**：Amortized Prompt: Lightweight Fine-Tuning for CLIP in Domain Generalization(*The University of Tokyo*)[[arXiv 2111](https://arxiv.org/abs/2111.12853v1)]
* **Tip-Adapter**：Tip-Adapter: Training-free CLIP-Adapter for Better Vision-Language Modeling(*Shanghai AI Laboratory*) [[arxiv 2111](https://arxiv.org/abs/2111.03930)][[PyTorch](https://github.com/gaopengcuhk/tip-adapter)]
* **DPL**：Domain Prompt Learning for Efficiently Adapting CLIP to Unseen Domains(*The University of Tokyo*)[[arXiv 2111](https://arxiv.org/abs/2111.12853)][[PyTorch](https://github.com/shogi880/DPLCLIP)]
* **CLIP-Adapter**: Better Vision-Language Models with Feature Adapters (*Shanghai AI Lab*) [[arXiv 2110](https://arxiv.org/abs/2110.04544)][[PyTorch](https://github.com/gaopengcuhk/CLIP-Adapter)]

### CLIP Variants: *Updating by Siyu He*
* **VL-T5**: Unifying Vision-and-Language Tasks via Text Generation(UNC Chapel Hill).[[ICML 2021](https://arxiv.org/abs/2102.02779)] [[PyTorch](https://github.com/j-min/VL-T5)]
* **DenseCLIP**: DenseCLIP: Language-Guided Dense Prediction with Context-Aware Prompting(Tsinghua University).[[CVPR 2022](https://arxiv.org/abs/2112.01518)] [[PyTorch](https://github.com/raoyongming/DenseCLIP)]
* **BeamCLIP**: Transferring Pre-trained Multimodal Representations with Cross-modal Similarity Matching (*LG*). [[NeurIPS 2022](https://openreview.net/forum?id=j2Vtg_jhKZ)]
* **UniCLIP**: UniCLIP: Unified Framework for Contrastive Language-Image Pre-training(LG AI Research).[[NeurIPS 2022](https://arxiv.org/abs/2209.13430)]
* **FLIP**: Scaling Language-Image Pre-training via Masking(*Meta AI*).[[CVPR 2023](https://arxiv.org/abs/2212.00794)] [[Pytorch](https://github.com/facebookresearch/flip)]
* **MaskCLIP**: MaskCLIP: Masked Self-Distillation Advances Contrastive Language-Image Pretraining(*USTC*).[[CVPR 2023](https://arxiv.org/abs/2208.12262)] [[PyTorch](https://github.com/LightDXY/MaskCLIP)]
* **Frozen**: Multimodal Few-Shot Learning with Frozen Language Models(DeepMind).[[arXiv 2106](https://arxiv.org/abs/2106.13884)]
* **CPT**: CPT: Colorful Prompt Tuning for Pre-trained Vision-Language Models(Tsinghua University).[[arXiv 2109](https://arxiv.org/abs/2109.11797)] [[PyTorch](https://github.com/thunlp/CPT)]
* **MAnTiS**: Multimodal Conditionality for Natural Language Generation.[[arXiv 2109](https://arxiv.org/abs/2109.01229)]
* **CAE v2**:CAE v2: Context Autoencoder with CLIP Targets(Baidu VIS).[[arXiv 2211](https://arxiv.org/abs/2211.09799)]
* **A-CLIP**: Attentive Mask CLIP(Microsoft Research Asia).[[arXiv 2212](https://arxiv.org/abs/2212.08653)]
* **EVA-CLIP**: EVA-CLIP: Improved Training Techniques for CLIP at Scale(Beijing Academy of Artificial Intelligence).[[arXiv 2303](https://arxiv.org/abs/2303.15389)] [[PyTorch](https://github.com/baaivision/EVA/tree/master/EVA-CLIP)]
* **LaCLIP**: Improving CLIP Training with Language Rewrites(*Google*).[[arXiv 2305](https://arxiv.org/abs/2305.20088)] [[PyTorch](https://github.com/LijieFan/LaCLIP)]
* **DLIP**: DLIP: Distilling Language-Image Pre-training(Xiamen University).[[arXiv 2308](https://arxiv.org/abs/2308.12956)]
* **ALIP**: ALIP: Adaptive Language-Image Pre-training with Synthetic Caption(DeepGlint).[[arXiv 2308](https://arxiv.org/abs/2308.08428)] [[PyTorch](https://github.com/deepglint/ALIP)]

### In-context learning: *Updating by Yansheng Gao*
* **CPL**: CPL: Counterfactual Prompt Learning for Vision and Language Models, arXiv, 2022 (*UC Santa Cruz*). [[Paper](https://arxiv.org/abs/2210.10362)]
* **Oscar** Multimodal Few-Shot Learning with Frozen Language Models, Nips, 2021 (DeepMind). [[paper](https://proceedings.neurips.cc/paper/2021/file/01b7575c38dac42f3cfb7d500438b875-Paper.pdf)]
* **HierKD** Open-Vocabulary One-Stage Detection with Hierarchical Visual-Language, CVPR, 2022 (MAIS & NLPR). [[paper](https://arxiv.org/abs/2203.10593)] [[Pytorch](https://github.com/mengqiDyangge/HierKD)]
* **Survey** What learning algorithm is in-context learning? Investigations with linear models, arxiv, 2022(Google Reasearch). [[paper](https://arxiv.org/pdf/2211.15661.pdf)] [[Pytorch](https://github.com/ekinakyurek/google-research/tree/master/incontext)]

### Domain Adaptation + Prompt Learning: *Updating by Yongguang Li*
* **CTTA**: Decorate the Newcomers Visual Domain Prompt for Continual Test Time Adaptation [[AAAI 2023](https://arxiv.org/pdf/2212.04145)]
* **AP**：Amortized Prompt: Lightweight Fine-Tuning for CLIP in Domain Generalization(*The University of Tokyo*)[[arXiv 2111](https://arxiv.org/abs/2111.12853v1)]
* **DAPL**：Domain Adaptation via Prompt Learning(*Tsinghua University*)[[arXiv 2202](https://arxiv.org/abs/2202.06687)][[Code](https://github.com/LeapLabTHU/DAPrompt)]
* **MIRO**：Domain Generalization by Mutual-Information Regularization with Pre-trained Models(*Kakao Brain*)[[ECCV 2022](https://arxiv.org/abs/2203.10789)][[Code](https://github.com/kakaobrain/miro)]
* **DPL**：Domain Prompt Learning for Efficiently Adapting CLIP to Unseen Domains(*The University of Tokyo*)[[arXiv 2111](https://arxiv.org/abs/2111.12853)][[PyTorch](https://github.com/shogi880/DPLCLIP)]
* **DOT**：Making the Best of Both Worlds: A Domain-Oriented Transformer for Unsupervised Domain Adaptation(*Beijing Institute of Technology*)[[ACMMM 2022](https://arxiv.org/abs/2208.01195)][[PyTorch](https://github.com/bit-da/domain-oriented-transformer)]
* **MPA**：Multi-Prompt Alignment for Multi-Source Unsupervised Domain Adaptation(*Fudan University*)[[ICLR 2023](https://arxiv.org/abs/2209.15210)]
* **PADA**：PADA: Example-based Prompt Learning for on-the-fly Adaptation to Unseen Domains(*Technion - Israel Institute of Technology*)[[TACL 2022](https://arxiv.org/abs/2102.12206)][[PyTorch](https://github.com/eyalbd2/PADA)]
* **DoPrompt**：Prompt Vision Transformer for Domain Generalization(*National University of Singapore*)[[arXiv 2208](https://arxiv.org/abs/2208.08914)][[PyTorch](https://github.com/zhengzangw/DoPrompt)]
* **DePT**：Visual Prompt Tuning for Test-time Domain Adaptation(*Rutgers University*)[[ICLR 2023](https://arxiv.org/abs/2210.04831)]
* **IPL**：Zero-Shot Generative Model Adaptation via Image-Specific Prompt Learning(*Tsinghua University*)[[CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Guo_Zero-Shot_Generative_Model_Adaptation_via_Image-Specific_Prompt_Learning_CVPR_2023_paper.html)][[PyTorch](https://github.com/Picsart-AI-Research/IPL-Zero-Shot-Generative-Model-Adaptation)]
* **DomainGen**：CLIP the Gap: A Single Domain Generalization Approach for Object Detection(*CVLab, EPFL*)[[CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Vidit_CLIP_the_Gap_A_Single_Domain_Generalization_Approach_for_Object_CVPR_2023_paper.pdf)][[PyTorch](https://github.com/vidit09/domaingen)]
* **RADA-prompt**：Domain Prompt Tuning via Meta Relabeling for Unsupervised Adversarial Adaptation(*Eastern Institute of Technology*)[[TMM 2023](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10115031)]
* **AD-CLIP**:AD-CLIP: Adapting Domains in Prompt Space Using CLIP (Indian Institute of Technology Bombay)[[arxiv 2308](https://arxiv.org/abs/2308.05659)]
* **PADCLIP** : PADCLIP: Pseudo-labeling with Adaptive Debiasing in CLIP for Unsupervised Domain Adaptation[[ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Lai_PADCLIP_Pseudo-labeling_with_Adaptive_Debiasing_in_CLIP_for_Unsupervised_Domain_ICCV_2023_paper.pdf)]
* **PromptStyler** : PromptStyler: Prompt-driven Style Generation for Source-free Domain Generalization[[ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Cho_PromptStyler_Prompt-driven_Style_Generation_for_Source-free_Domain_Generalization_ICCV_2023_paper.html)]
* Open-Set Domain Adaptation with Visual-Language Foundation Models[[arXiv 2307](https://arxiv.org/abs/2307.16204)]
# Contributing & Contact
Feel free to contribute to our repository.

- If you woulk like to *correct mistakes*, please do it directly;
- If you would like to *add/update papers*, please follow the existing format;
- If you have any *questions or advice*, please contact us by email (summitlsf@outlook.com) or GitHub issues.

Thank you for your support!




## References
* Online Resources:
