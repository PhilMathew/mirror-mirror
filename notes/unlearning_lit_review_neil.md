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

# Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning
- Heuristic Approach
- Uses Negative Preference Optimization

# Dissecting Fine-Tuning Unlearning in Large Language Models
- Evaluation of the shortcomings of finetuning (heuristic) based approaches to unlearning
- Shows that this alters model retrieval process but does not remove knowledge

# NegMerge: Consensual Weight Negation for Strong Machine Unlearning
- Run multiple task vectors
- merge the consistenct components of multiple task vectors
- Use this for task vector based machine unlearning

# A Probabilistic Perspective on Unlearning and Alignment for Large Language Models
- Determinstic evaluations of unlearning fail
- novel metric with "high probability" guarantees concerning model output
- WOrth a deeper look
- Has multiple definitions of "leaking" information that are interesting

# Mitigating Memorization In Language Models
- Three heuristic ways to decrease memorization

# Answer When Needed, Forget When Not: Language Models Pretend to Forget via In-Context Knowledge Unlearning
- Shows that LLMs often contain information but choose to not act on it
- Deliberately finetune a model to do this
- Heuristic based unlearning

# An Adversarial Perspective on Machine Unlearning for AI Safety
- Can use jailbreaking techniques to break unlearning
- Neil note: Does cash's critique apply?

# Alternate Preference Optimization for Unlearning Factual Knowledge in Large Language Models
 - Heuristic based 

# LLM Surgery: Efficient Knowledge Unlearning and Editing in Large Language Models
