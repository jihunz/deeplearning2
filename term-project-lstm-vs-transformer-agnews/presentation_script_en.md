# Presentation Script — LSTM vs Transformer (AG News) · condensed

## Time budget (importance-weighted)

| # | Slide | Time | Priority |
|---|---|---|---|
| 1 | Title | 0:10 | ○ |
| 2 | Background | 0:25 | ◎ |
| 3 | Dataset | 0:20 | ○ |
| 4 | Models (mechanism) | 1:00 | ★ |
| 5 | Experiments | 0:20 | ○ |
| 6 | Main results | 0:55 | ★ |
| 7 | Errors | 0:30 | ◎ |
| 8 | Ablation | 0:50 | ★ |
| 9 | Extension (mechanism) | 0:45 | ★ |
| 10 | Conclusion | 0:45 | ◎ |
| | Speaking total | ~5:40 | + transitions ~0:45 → ~6:25 |

★ core (4·6·8·9) · ◎ important (2·7·10) · ○ compressed (1·3·5). Slides 4·6·8·10 carry the mechanism argument.

---

## Script (first person · unified academic terminology)

### Slide 1 · Title (0:10)
Good afternoon, I'm Jihun Jang from ETRI School. I compared an LSTM and a Transformer Encoder on AG News topic classification, under identical, controlled conditions.

### Slide 2 · Background (0:25)
My goal was not to maximize accuracy, but to explain why the two architectures differ. Both are trained from scratch, and I held everything identical except the encoder, so any difference can be attributed to the architecture. I ask how they differ in performance, convergence, data efficiency, and errors, and what causes it.

### Slide 3 · Dataset (0:20)
AG News, split into 108,000 training, 12,000 validation, and 7,600 test examples, with four balanced classes. Preprocessing is shared: word-level tokens, a vocabulary built from training only to prevent leakage, and max length 128 with padding masked out.

### Slide 4 · Models (1:00) ★
Now the models. Both share one backbone: Embedding, Encoder, masked mean pooling, and a Linear layer; I swapped only the encoder, so the difference is purely in how the encoder builds each token's representation. The LSTM is recurrent: it reads the sequence step by step, carrying a hidden state that compresses everything seen so far, so information has to pass through a single fixed-size channel; in practice this lets it lean heavily on a few strong tokens. The Transformer uses self-attention: every token attends to every other token directly and in parallel, so each representation aggregates the whole sequence with no sequential bottleneck, spreading evidence across many tokens. This contrast is what I use to explain the results: the Transformer aggregates evidence broadly, while the LSTM concentrates it. The parameter counts are within about 14 percent, 3.22 versus 2.83 million, so this is a comparison of mechanism, not size.

### Slide 5 · Experiments (0:20)
For a fair comparison, the only variable is the encoder; everything else is fixed. I trained with Adam, learning rate 0.001, batch 64, up to 8 epochs, seed 42, selecting on validation and testing once. I also ran a data-efficiency ablation at 5, 25, 50, and 100 percent, measuring accuracy, macro F1, loss, and confusion matrices.

### Slide 6 · Main results (0:55) ★
The Transformer reached 0.910 accuracy versus the LSTM's 0.833, about 7.7 points; but the cause matters more than the number. Look at the loss curves: the LSTM's training loss falls almost to zero while its validation loss climbs from 0.5 to about 2.0, textbook overfitting, whereas the Transformer keeps validation loss flat and is already at 88 percent after the first epoch. This is the mechanism showing through. Because recurrence concentrates the decision on a few tokens, the LSTM ends up memorizing training-specific cues, fitting the training set but generalizing poorly; the Transformer, aggregating evidence across many tokens, has no single feature to latch onto and overfit, so it stays stable. Since I select the best checkpoint by validation, the LSTM's final model is taken from epoch 2. So the gap is a difference in generalization, and it comes straight from how each encoder aggregates information.

### Slide 7 · Errors (0:30) ◎
Both models confuse Business and Sci/Tech the most, from shared vocabulary, semantic overlap. Of the 428 cases both got wrong, 61 percent are on that boundary, ambiguous labels, so it's the task, not the model. But the LSTM alone was wrong on 841 cases, 3.2 times the Transformer, spread across all classes, a sign of weak generalization even on easy examples.

### Slide 8 · Ablation (0:50) ★
Next, the data-efficiency ablation. I expected the Transformer to need more data; the opposite held. Its accuracy rises monotonically, from 0.798 at 5 percent to 0.910 at full data, while the LSTM peaks at just 25 percent, 0.846, and then flattens. Again, this follows from the mechanism: once the LSTM has enough data to lock onto its preferred tokens, more data does not help, because the limit is its concentrated representation, not the amount of data. The Transformer's distributed representation keeps absorbing new evidence, so it keeps scaling. And since it led even at 5 percent, I rejected my hypothesis that it is more data-hungry.

### Slide 9 · Extension · mechanism (0:45) ★
I probed the mechanism directly. Removing positional encoding left accuracy unchanged, 0.910 versus 0.911, so the task is essentially bag-of-words; word order barely matters. Then, on one Sports article, I compared which tokens each model relied on. The LSTM put almost all of its weight on "giddy", an off-topic word, and misclassified the article; the Transformer spread its weight over topical tokens like medley and 400, and was correct. This is the mechanism caught in the act: the LSTM bet almost everything on a single token, while the Transformer spread its bet across the evidence, which is exactly why it is more robust and generalizes better.

### Slide 10 · Conclusion (0:45) ◎
To conclude. Under a controlled comparison the Transformer won, 0.910 to 0.833, and the cause traces cleanly back to mechanism. Self-attention aggregates evidence across many tokens, which makes it robust, keeps its validation loss flat, and lets it scale with data; recurrence funnels the decision through a sequential bottleneck onto a few tokens, which causes overfitting and early saturation. So one mechanistic difference explains every result: the accuracy gap, the loss curves, and the data-efficiency curve. I also confirmed the task is bag-of-words, so the gap is about robustness, not word order. The main limitation is the single seed; future work is multiple seeds, stronger LSTM regularization, and order-sensitive tasks. Thank you.

---

## Delivery tips
- **Pace**: slow on the core slides (6·8·9), faster on the compressed ones (1·3·5). The end of slide 6 is roughly your halfway point.
- **Words to stress**: "only the encoder is the variable" (4), "overfitting" (6), "hypothesis rejected" (8), "bag-of-words" (9).
- **Q&A prep**: definition of macro F1-score (the diagonal of the confusion matrix is the true positives), the meaning of the best checkpoint, and the single-seed power limitation.
