Papers to review:
https://github.com/chrisliu298/awesome-llm-unlearning

# Learn What You Want To Unlearn
- Uses both the original model and the unlearned model in its inversion
- finds the forgotten information... but in the original model?
- Tries to recover the data

# Certified Minimax Unlearning with Generalization Rates and Deletion Capacity
- Very relevant to theoretical proof sections
- Offers a new definition of (ε,δ)-Certified Machine Unlearning

# Position: LLM Unlearning Benchmarks are Weak Measures of Progress
- forget/retain dependencies  
- overfitting to the test queries
- Multiple pieces of information in the forget set
- Looks interesting but might be more limited to language

# EXTRACTING UNLEARNED INFORMATION FROM LLMS WITH ACTIVATION STEERING
- Uses Steering Vector to find lost information
- Steering vector is defined as vector of activations between "anonymized" version of the data points and the query concatenating the unlearned objective

# BENCHMARKING VISION LANGUAGE MODEL UNLEARNING VIA FICTITIOUS FACIAL IDENTITY DATASET
- generates a fictious dataset of faces to benchmark unlearning

# EVALUATING DEEP UNLEARNING IN LARGE LANGUAGE MODELS
- Uses "deep unlearning which considers the logical deductions between facts."
- metric recall associated with this
- interesting and may be worth looking into but language centric (fact) based

# Unlearning in- vs. out-of-distribution data in LLMs under gradient-based methods
- Metric for unlearning in generative models
- Seems to provide unlearning definition similar to ours
	- Sec 2.1
- Needs further look @brennon

# RESTOR: Knowledge Recovery through Machine Unlearning
- Uses corruptions to evaluate unlearning
- Evaluates if the model still retains the original knowledge independent of the forget set
- Seems to be similar to checkpointing based unlearning, as it returns the model to "before" it saw bad facts
- Has 4 emperical unlearning methods
- detailed analysis of how the models differ

# Protecting Privacy in Multimodal Large Language Models with MLLMU-Bench
- Unlearning Benchmark
- Checks if maintains original information by performance on retain data

#  Cross-model Control: Improving Multiple Large Language Models in One-time Training
- Trains multiple LLMs at once
- Might be relevant for future emperical work


# Assignments
## Phil 
Unlearning as multi-task optimization: A normalized gradient difference approach with an adaptive learning rate

Learning and Unlearning of Fabricated Knowledge in Language Models

Applying sparse autoencoders to unlearn knowledge in language models

CLEAR: Character Unlearning in Textual and Visual Modalities

WAGLE: Strategic Weight Attribution for Effective and Modular Unlearning in Large Language Models

## Brennon

UnStar: Unlearning with Self-Taught Anti-Sample Reasoning for LLMs
Going backwards - 10

Does your LLM truly unlearn? An embarrassingly simple approach to recover unlearned knowledge

When Machine Unlearning Meets Retrieval-Augmented Generation (RAG): Keep Secret or Forget Knowledge?

Evaluating Deep Unlearning in Large Language Models
Author(s): Ruihan Wu, Chhavi Yadav, Russ Salakhutdinov, Kamalika Chaudhuri
Date: 2024-10
Venue: -
Code: GitHub
Unlearning Backdoor Attacks for LLMs with Weak-to-Strong Knowledge Distillation
Author(s): Shuai Zhao, Xiaobao Wu, Cong-Duy Nguyen, Meihuizi Jia, Yichao Feng, Luu Anh Tuan
Date: 2024-10
Venue: -
Code: GitHub
Breaking Chains: Unraveling the Links in Multi-Hop Knowledge Unlearning
Author(s): Minseok Choi, ChaeHun Park, Dohyun Lee, Jaegul Choo
Date: 2024-10
Venue: -
Code: GitHub
Mechanistic Unlearning: Robust Knowledge Unlearning and Editing via Mechanistic Localization
Author(s): Phillip Guo, Aaquib Syed, Abhay Sheshadri, Aidan Ewart, Gintare Karolina Dziugaite
Date: 2024-10
Venue: -
Code: -
LLM Unlearning via Loss Adjustment with Only Forget Data
Author(s): Yaxuan Wang, Jiaheng Wei, Chris Yuhao Liu, Jinlong Pang, Quan Liu, Ankit Parag Shah, Yujia Bao, Yang Liu, Wei Wei
Date: 2024-10
Venue: -
Code: -
CodeUnlearn: Amortized Zero-Shot Machine Unlearning in Language Models Using Discrete Concept
Author(s): YuXuan Wu, Bonaventure F. P. Dossou, Dianbo Liu
Date: 2024-10
Venue: -
Code: -
Do Unlearning Methods Remove Information from Language Model Weights?
Author(s): Aghyad Deeb, Fabien Roger
Date: 2024-10
Venue: -
Code: GitHub
A Closer Look at Machine Unlearning for Large Language Models
Author(s): Xiaojian Yuan, Tianyu Pang, Chao Du, Kejiang Chen, Weiming Zhang, Min Lin
Date: 2024-10
Venue: -
Code: GitHub


## Neil
Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning
Author(s): Chongyu Fan, Jiancheng Liu, Licong Lin, Jinghan Jia, Ruiqi Zhang, Song Mei, Sijia Liu
Date: 2024-10
Venue: -
Code: GitHub
Dissecting Fine-Tuning Unlearning in Large Language Models
Author(s): Yihuai Hong, Yuelin Zou, Lijie Hu, Ziqian Zeng, Di Wang, Haiqin Yang
Date: 2024-10
Venue: EMNLP 2024
Code: GitHub
NegMerge: Consensual Weight Negation for Strong Machine Unlearning
Author(s): Hyoseo Kim, Dongyoon Han, Junsuk Choe
Date: 2024-10
Venue: -
Code: GitHub
A Probabilistic Perspective on Unlearning and Alignment for Large Language Models
Author(s): Yan Scholten, Stephan Günnemann, Leo Schwinn
Date: 2024-10
Venue: -
Code: -
Mitigating Memorization In Language Models
Author(s): Mansi Sakarvadia, Aswathy Ajith, Arham Khan, Nathaniel Hudson, Caleb Geniesse, Kyle Chard, Yaoqing Yang, Ian Foster, Michael W. Mahoney
Date: 2024-10
Venue: -
Code: -
Answer When Needed, Forget When Not: Language Models Pretend to Forget via In-Context Knowledge Unlearning
Author(s): Shota Takashiro, Takeshi Kojima, Andrew Gambardella, Qi Cao, Yusuke Iwasawa, Yutaka Matsuo
Date: 2024-10
Venue: -
Code: -
An Adversarial Perspective on Machine Unlearning for AI Safety
Author(s): Jakub Łucki, Boyi Wei, Yangsibo Huang, Peter Henderson, Florian Tramèr, Javier Rando
Date: 2024-09
Venue: -
Code: GitHub
Alternate Preference Optimization for Unlearning Factual Knowledge in Large Language Models
Author(s): Anmol Mekala, Vineeth Dorna, Shreya Dubey, Abhishek Lalwani, David Koleczek, Mukund Rungta, Sadid Hasan, Elita Lobo
Date: 2024-09
Venue: -
Code: -
LLM Surgery: Efficient Knowledge Unlearning and Editing in Large Language Models
Author(s): Akshaj Kumar Veldanda, Shi-Xiong Zhang, Anirban Das, Supriyo Chakraborty, Stephen Rawls, Sambit Sahu, Milind Naphade
Date: 2024-09
Venue: -
Code: -
MEOW: MEMOry Supervised LLM Unlearning Via Inverted Facts
Author(s): Tianle Gu, Kexin Huang, Ruilin Luo, Yuanqi Yao, Yujiu Yang, Yan Teng, Yingchun Wang
Date: 2024-09
Venue: -
Code: - GitHub