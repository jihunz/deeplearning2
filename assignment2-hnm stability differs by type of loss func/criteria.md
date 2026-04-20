Advanced Study 02
Due Apr 20, 23:59:59 (No delayed submissions are allowed)
Deep Learning Advanced Study Assignment
1. Objective
The goal of this assignment is to investigate a deep learning question of personal interest through self-directed study.
You will select one focused question related to the course material and explore it through theoretical analysis and small-scale empirical validation.
2. Topic Selection
Choose one question-based topic that is clearly related to the lecture content covered from the beginning of the course material 03. Tips and Techniques up to Ranking Loss.
Your topic does not need to be limited to concepts explicitly written on the slides.
You may choose a related advanced topic that extends the covered material, as long as the connection to the lecture scope is clear.
Examples of acceptable topic directions

loss functions for classification
entropy, cross-entropy, and KL divergence
forward KL vs. reverse KL
hinge loss and margin-based learning
pairwise ranking loss and triplet loss
metric learning and retrieval
hard negative mining
label smoothing
softmax classification vs. embedding learning
3. Topic Requirement
Your topic should be written as a question, not just as a keyword or concept name.
Preferred style
Why does the direction of KL divergence matter?
How does triplet loss differ from pairwise ranking loss?
Why is ranking loss useful for retrieval tasks?
How does hard negative mining improve metric learning?
A question-based topic leads to a more focused and rigorous report.
Recommended Approach
1. Literature and Tool Use
You may use:
internet search
AI tools such as ChatGPT
PyTorch or other deep learning toolkits
2. What Your Study Should Include
A strong report should combine:
conceptual background
theoretical or mathematical analysis
comparison with related methods
a small programming-based experiment
interpretation of the results
3. Experimental Expectation
You are not expected to build a large-scale system.
However, your report should include a small but meaningful experiment that helps test or illustrate your main question.
Examples:
compare two loss functions on the same classification task
compare forward KL and reverse KL on toy distributions
compare pairwise loss and triplet loss in embedding learning
study the effect of margin size
compare random triplet sampling and hard negative mining
Submission Requirements
1. Format
Submit a PDF report
Filename convention: A_01_(yourname).pdf
2. Required Components
Your report should include the following sections:
1) Research Question
State the specific question your report aims to answer.
2) Motivation
Explain:
why the question is interesting,
how it is connected to the lecture material,
and why you chose it.
3) Background
Introduce the concepts, definitions, and mathematical setting necessary to understand the topic.
4) Main Analysis
Present the core theoretical discussion of the report.
This may include:
derivation,
comparison,
interpretation,
geometric intuition,
discussion of strengths and limitations,
explanation of practical implications.
5) Experimental Verification
Design and conduct a small experiment related to your question.
Clearly describe:
dataset,
model,
objective/loss,
experimental setting,
evaluation metric,
and results.
6) Interpretation
Explain what the results mean.
Discuss:
whether the results support your theoretical expectation,
what insights were confirmed,
what limitations or unexpected behaviors appeared.
7) Conclusion
Provide a direct answer to the original question and summarize the main findings.
8) Appendix
Include:
AI chat log
program code
additional plots, tables, or results
optional supplementary derivations
Important Notes
1. Topic Quality
A good topic is:
focused,
clearly connected to the covered lecture range,
narrow enough to analyze in depth,
and suitable for both theory and experiment.
2. What to Avoid
Avoid topics that are:
too broad,
purely descriptive,
unrelated to the lecture scope,
or lacking a clear question to answer.
Weak topic example
Cross-Entropy Loss
Stronger version
Why is cross-entropy more suitable than MSE for classification?
3. Use of AI
You may use AI tools for:
idea exploration,
conceptual clarification,
coding assistance,
debugging,
and report organization.
However, you remain responsible for:
understanding the content,
checking correctness,
interpreting results,
and writing a coherent report.
Suggested Evaluation Criteria
1. Clarity of Research Question
Is the question specific and meaningful?
Is the scope appropriate?
2. Theoretical Understanding
Does the report explain the concepts accurately?
Is the analysis logically developed?
3. Quality of Experiment
Is the experiment relevant to the question?
Is the setup appropriate and interpretable?
4. Interpretation
Does the report explain what the results actually show?
Does it connect empirical findings back to theory?
5. Overall Quality
Is the report well organized?
Is the discussion rigorous and coherent?
Example Topic Candidates
Theory-oriented
Why does the direction of KL divergence matter?
What is the relationship between entropy, cross-entropy, and KL divergence?
What role does the margin play in hinge loss and ranking loss?
Experiment-oriented
Pairwise ranking loss vs. triplet loss
Hard negative mining for triplet loss
Label smoothing and cross-entropy
Softmax classification vs. metric learning for retrieval
Application-oriented
Why ranking loss is useful for face recognition
Why embedding learning is useful for retrieval tasks
When similarity learning is more appropriate than closed-set classification