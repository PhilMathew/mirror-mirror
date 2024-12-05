

# Assignments
## Phil 
Unlearning as multi-task optimization: A normalized gradient difference approach with an adaptive learning rate

Learning and Unlearning of Fabricated Knowledge in Language Models

Applying sparse autoencoders to unlearn knowledge in language models

CLEAR: Character Unlearning in Textual and Visual Modalities

WAGLE: Strategic Weight Attribution for Effective and Modular Unlearning in Large Language Models

## Brennon

## UnStar: Unlearning with Self-Taught Anti-Sample Reasoning for LLMs
- To unlearn, UNSTAR intentionally provides incorrect answers and their justifications as an anti-sample. For instance, it generates Where did Harry Potter study? Ilvermorny. Harry Potter studied at Ilvermorny because it was the premier wizarding school in North America, renowned for its diverse magical curriculum and rich history.
- previous unlearning techniques can inadvertently disrupt the LLM’s broader knowledge
- to address this challenge, we propose fine-grained targeted unlearning, which allows for the selective removal of specific associations. In the aforementioned example, other related facts—such as that Harry Potter is a wizard and Hogwarts is a boarding school of magic for young wizards—should not be forgotten.
- *approximate unlearning* (Chundawat et al. (2023a)), which focuses on reversed loss functions, reduces the influence of target data points through parameter-level updates, significantly lowering computational costs [as compared to exact unlearning].
- The predominant approach utilizes gradient ascent to maximize prediction loss on the data to be forgotten (Jang et al. (2022); Yao et al. (2023)). Other techniques involve training the model to produce alternative responses, such as “I don’t know” (Ishibashi & Shimodaira (2023)), random labels (?), or predictions based on perturbed inputs (Eldan & Russinovich (2023)). Additionally, recent studies have investigated task arithmetic (Ilharco et al. (2023); Barbulescu & Triantafillou (2024)) and training-free methods for unlearning by incorporating specific instructions or in-context examples (Thaker et al. (2024); Pawelczyk et al. (2024)). However, these methods often lack the granularity required for fine-tuned control over what specific information is forgotten, which is where our approach—utilizing anti-samples—proposes a more refined solution.

## Does your LLM truly unlearn? An embarrassingly simple approach to recover unlearned knowledge
- Despite the effectiveness of current unlearning methods, little attention has been given to whether existing unlearning methods for LLMs truly achieve forgetting or merely hide the knowledge, which current unlearning benchmarks fail to detect.
- This paper reveals that applying quantization to models that have undergone unlearning can restore the "forgotten" information.
- Our results highlight a fundamental tension between preserving the utility of the unlearned model and preventing knowledge recovery through quantization, emphasizing the challenge of balancing these two objectives.
- Generally, machine unlearning for LLMs aims to remove the memorization of specific knowledge while maximally preserving utility. Among the advanced unlearning methods, gradient ascent (GA) (Yao et al., 2023) and negative preference optimization (NPO) (Zhang et al., 2024) are the most foundational.
- GA aims to minimize the likelihood of making correct predictions on a forget dataset by applying gradient ascent to the cross-entropy loss. On the other hand, NPO treats the forget set as negative preference data, adapting the offline DPO (Rafailov et al., 2024) objective to adjust the model to assign a lower likelihood to the forget set. Since GA and NPO are not designed for utility preservation, several regularization techniques (Shi et al., 2024b; Maini et al., 2024) are typically combined with unlearning to preserve utility.
- For example, given a retain dataset, techniques such as gradient descent on the retain dataset (Zhang et al., 2024; Maini et al., 2024) and minimizing the KL divergence between the unlearned model’s and the target model’s probability distributions on inputs from the retain dataset (Zhang et al., 2024; Maini et al., 2024) are introduced to enhance the utility of the unlearned model.
- As shown in Table 1, applying the unlearning method GA_KLR on the BOOKS dataset (Shi et al., 2024b) results in the unlearned model retaining only 13% of its original knowledge. However, when the unlearned model undergoes quantization, knowledge retention recovers to approximately 89%.
- The similarity between Q(funlearn) and Q(ftarget) indicates that the quantized unlearned model may inadvertently retain knowledge from the forget set, even though the full-precision unlearned model successfully eliminates such information.
- **Brennon:** this shows that SOTA unlearning fiddles with the low-order bits, not the high-order bits.

## When Machine Unlearning Meets Retrieval-Augmented Generation (RAG): Keep Secret or Forget Knowledge?
- The key challenge is the difficulty of modifying the parameters in LLMs due to the huge volumes of parameters. Second, the output space of LLMs is much larger than the classification space, which hinders the verification of LLM unlearning. The two challenges block the implementation of exact LLM unlearning. Therefore, approximate unlearning for LLM has become a research hotspot (Offset unlearning for large language models, In-context unlearning: Language models as few-shot unlearners, Who’s harry potter? approximate unlearning in llms)
- However, these LLM unlearning schemes have inherent weaknesses. First, the schemes involving fine-tuning LLM introduce considerable computational overhead, like gradient ascent, so they cannot adapt to frequent unlearning requests. Second, closed-source LLMs only provide their interfaces, making some schemes unsuitable. Third, LLMs possess emergent abilities, which means that forgetting must extend beyond specific data to include related content.
- Shumailov et al. discovered that LLMs can recall previously forgotten knowledge through their in-context abilities, exposing a vulnerability in existing unlearning schemes to this ‘un-unlearning’ phenomenon.
- Fortunately, leveraging RAG to keep the forgotten target confidential can function similarly to unlearning it.

## Evaluating Deep Unlearning in Large Language Models
- Prior work in unlearning in large language models have looked at this problem, but for the target fact, the focus has been on removing the target itself. However, this can be superficial – LLMs not only know single facts in isolation, but many connected facts – and very often, the fact that has been unlearnt can be deduced from facts that are already known by the model.
- If the LLM only unlearns the target fact but retains A and B, this is insufficient as an adversary who extracts A and B from the LLM can deduce the target fact.
- Deep unlearning is formulated by stating a set of facts and logical rules that connect the facts. The fact is deeply unlearnt if the target fact cannot be deduced from the retained facts in the LLM through the given logical rules.
- **Brennon:** I think they are just making two points: (1) the point that defining the forget set is tricky and (2) you have to hold against adaptive adversarial queries.

## Unlearning Backdoor Attacks for LLMs with Weak-to-Strong Knowledge Distillation
- How can backdoor features be unlearned without compromising model performance by leveraging PEFT (parameter-efficient fine-tuning)?
- We propose a novel unlearning algorithm to defend against backdoor attacks, Weak-to-Strong-Defense (W2SDefense), which enables a poisoned student model to unlearn backdoors through knowledge distillation from a clean teacher model.
- **Brennon:** I think they are trying to use the fact that cryptographic backdoors are _evasive._ They are built using the unforgeability property of digital signatures; it is infeasible to trigger the backdoor without the signing key. Thus, distillation is unlikely to preserve the backdoor.  

## Breaking Chains: Unraveling the Links in Multi-Hop Knowledge Unlearning
- Prior research has largely focused on minimizing the probabilities of specific token sequences by reversing the language modeling objective. However, these methods still leave LLMs vulnerable to adversarial attacks that exploit indirect references. In this work, we examine the limitations of current unlearning techniques in effectively erasing a particular type of indirect prompt: multi-hop queries. Our findings reveal that existing methods fail to completely remove multi-hop knowledge when one of the intermediate hops is unlearned.
- We express a fact as a triple (s,r,o), where s is the subject, r the relation, and o the object.
- In this section, we present Multi-Hop unlearning via UNCertainty tHreshold (MUNCH), a simple yet effective method to enhance the performance of unlearning multi-hop facts.
- With the rise of LLMs, new methods have emerged to enable forgetting specific token sequences, such as through gradient ascent (Jang et al., 2023), additional retention strategies (Lee et al., 2024), and negative preference optimization (Zhang et al., 2024c). However, Choi et al. (2024) identified a significant limitation in existing knowledge unlearning techniques: they fail to generalize across different languages, leaving models vulnerable to attacks in low-resource languages. This reveals that when token sequences are substituted or aliased with alternative sequences, the unlearning process can be circumvented.
- **Brennon:** unclear how their unlearning technique works. Seems to be very related to _Evaluating Deep Unlearning in Large Language Models_ (above). The key contribution to the lit review is that most LLM learning is viewed as lowering the probability of a fixed sequence of tokens. But why do this at all? Just do a regex of "things to forget" over the generated response.

## Mechanistic Unlearning: Robust Knowledge Unlearning and Editing via Mechanistic Localization
- Some recent work has explored editing or unlearning techniques that rely on mechanistic interpretability methods attempting to trace which components of a network store specific facts (Meng et al., 2023). These methods, such as causal tracing or attribution patching, focus on measuring how output or task accuracy is affected when clean/corrupted input is patched into specific components. However, the effectiveness of these “output-tracing” (OT) techniques for editing has been questioned by Hase et al. (2023). Our research confirms these doubts, finding that localized editing and unlearning of facts based on several existing OT methods often perform equal to or worse than simply updating the entire model.
- We perform a rigorous evaluation of several standard unlearning approaches on factual recall tasks and show they fail to generalize to prompting/output distribution shifts and adversarial relearning.
- We identify mechanisms for factual lookup and attribute extraction on Gemma-7B and Gemma-2-9B. We demonstrate that gradient-based unlearning and editing localized on the factual lookup mechanism is more robust and generalizes better than OT localizations and baselines across multiple datasets and models.
- ***One formalization of unlearning to match a retrained-from-scratch model is due to Ginart et al. (2019), and is closely inspired by differential privacy (Dwork et al., 2014).***
- **Brennon:** summary is that we don't know where facts live in LLM parameters, otherwise these deletion/patching techniques would work. ***We need to read Ginart et al.***

## CodeUnlearn: Amortized Zero-Shot Machine Unlearning in Language Models Using Discrete Concept
- In this study, we propose a novel amortized unlearning approach using codebook features and Sparse Autoencoders (SAEs). By leveraging a bottleneck to decompose the activation space and regulate information flow, our method efficiently unlearns targeted information while preserving the model’s performance on unrelated data. To the best of our knowledge, this is the first work that successfully enables unlearning specific topics with contextual relevance in an LLM, marking a significant step towards real-world applications of machine unlearning.
- Existing solutions, such as Sharded, Isolated, Sliced, and Aggregated (SISA) training Bourtoule et al. (2020), primarily involve partitioning the training data into disjoint shards and retraining models on these individual shards.
- Unlike traditional unlearning techniques that rely on retraining portions of the model, zero-shot unlearning seeks to directly eliminate the influence of specific data points or pieces of information from the model’s learned representation—without additional computational steps or parameter adjustments.
- We demonstrate how Vector Quantization (VQ) can structure the latent space, facilitating the selective removal of information in an amortized manner.
- **Brennon:** interesting idea, basically you take advantage of the transformer architecture's internal embedding structure: "the multi-head attention output is discretized using a discrete embedding vocabulary, referred to as the codebook. This approach prevents information leakage via the residual connection, ensuring that the codebook effectively regulates and interprets the network’s behavior. Zero-shot machine unlearning is achieved by removing the discrete codes in the codebook that correspond to the targeted information." So basically you are lobotomizing the embedding.

## Do Unlearning Methods Remove Information from Language Model Weights?
- We propose an adversarial evaluation method to test for the removal of information from model weights: we give an attacker access to some facts that were supposed to be removed, and using those, the attacker tries to recover other facts from the same distribution that cannot be guessed from the accessible facts. We show that using fine-tuning on the accessible facts can recover 88% of the pre-unlearning accuracy when applied to current unlearning methods, revealing the limitations of these methods in removing information from the model weights.
- We present a framework for evaluating the extent to which unlearning methods remove knowledge from the weights. We create new datasets and modify existing ones to fit the desired criteria of our framework. Using our framework and these datasets, we are able to quantify the amount of knowledge that was hidden but not removed from model weights.
- **Brennon:** I like the adversarial framework, which is missing from other literature. Seems to be very related to _Breaking Chains: Unraveling the Links in Multi-Hop Knowledge Unlearning_ and _Evaluating Deep Unlearning in Large Language Models_ (both above).

## A Closer Look at Machine Unlearning for Large Language Models
- Specifically, the behavior that untargeted unlearning attempts to approximate is unpredictable and may involve hallucinations, and existing regularization is insufficient for targeted unlearning. To alleviate these issues, we propose using the objective of maximizing entropy (ME) for untargeted unlearning and incorporate answer preservation (AP) loss as regularization for targeted unlearning.
- We note that most prior studies (Maini et al., 2024; Ji et al., 2024; Jia et al., 2024; Jin et al., 2024; Shi et al., 2024a) primarily rely on ROUGE (Lin, 2004) as the sole metric for evaluating the output of unlearned models. To more comprehensively assess the model behavior, we introduce three additional metrics that evaluate token diversity, sentence semantics, and factual correctness in the output.
- ***Brennon:*** Reviews GA and NPO, along with fine-tuning techniques for "IDK" responses.
- By combining different forget losses and regularization losses, we can obtain seven baseline methods, namely GA+GD, GA+KL, NPO+GD, NPO+KL, DPO+GD, DPO+KL and IDK+GD.
- In traditional classification tasks, since the model size is relatively small and the labeled dataset is unambiguous, it is possible for researchers to train a ideal retain model from scratch for evaluation purpose under the unlearning objective, that is, ***compare the indistinguishability of the unlearned model and the ideal retain model on some metrics***, such as the accuracy on $D_\text{forget}$ (Nguyen et al., 2022). In the context of LLMs, the computational cost of retraining is prohibitive for most people.
- In practice, we minimize the KL divergence between the predicted distribution for each token and a uniform distribution with vocabulary size.
- ***Brennon:*** appears to argue "LLMs too expensive to evaluate against a control, so why bother? `#yolo`" Weird.


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