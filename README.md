# AmazonHelp Customer-Support Reply Agent

## What this project does

This prototype classifies AmazonHelp customer messages into a 9-intent taxonomy, retrieves similar historical customer/support conversations with FAISS, and generates a concise reply with Gemini when the case passes escalation rules.

The current notebook uses:

- **Data source:** Kaggle Customer Support on Twitter (`twcs.csv`)
- **Brand:** `AmazonHelp`
- **Historical English pairs used for retrieval:** 823 customer → AmazonHelp conversations
- **Labeling sample:** 500 English customer messages
- **Intent classifier:** `sentence-transformers/all-MiniLM-L6-v2` embeddings + balanced logistic regression
- **RAG:** FAISS inner-product search over normalized embeddings, top-k=3
- **Reply generator:** `gemini-3.5-flash-lite`
- **Escalation:** human review for high-risk intents (`payment_problem`, `refund_request`, `account_problem`, `order_cancellation`) or low classifier confidence / low retrieval similarity

## Headline result

The strongest external evaluation number in the notebook is:

> **63.87% intent accuracy on a 191-example golden evaluation set (122/191 correct).**

The notebook also reports **66% accuracy on a random 80/20 split of the 500 automatically labeled examples**. That is an internal development metric, not the headline quality claim.

## Reproduce the headline number

### Fastest path

Use the original Kaggle notebook with the same data inputs, run through the classifier training cells, then run the golden-dataset evaluation cell. The golden evaluation itself is deterministic once the classifier has been trained.

1. Open `agent-for-cutomer-queries.ipynb` in Kaggle.
2. Attach these Kaggle datasets/inputs:
   - Customer Support on Twitter (`twcs/twcs.csv`)
   - Golden dataset (`amazon_support_intent_golden_eval.csv`)
3. Run cells in order through the classifier-training cell.
4. Run the golden evaluation cell.
5. Confirm:
   - `Total: 191`
   - `Correct: 122`
   - `Wrong: 69`
   - `Accuracy: 63.87%`

### Reproduction caveat

A clean-room, sub-15-minute reproduction is **not fully guaranteed by the current notebook** because the notebook dynamically creates the 500-example training labels with an external Gemini call and does not save the trained classifier or the generated labels as versioned artifacts. The notebook output does, however, contain the exact reported headline result.

For a true <15-minute reproducibility package, the next iteration should persist:

- `labeled_train.csv` (the 500 generated labels)
- classifier + embedding model artifacts
- the exact golden CSV
- one pinned environment file
- one evaluation command that skips LLM labeling

The evaluation harness in this package is designed around that artifact-based workflow.

## Intent taxonomy

```text
order_tracking
order_cancellation
refund_request
delivery_problem
payment_problem
account_problem
product_problem
technical_problem
general_inquiry
```

## Current evaluation facts

| Evaluation | Size | Accuracy | Interpretation |
|---|---:|---:|---|
| Random 80/20 split of generated labels | 100 test | 66.00% | Development metric; label quality depends on Gemini-generated labels |
| Golden evaluation | 191 | **63.87%** | Main external-style evaluation used in the notebook |

Golden-set class support is:

- account_problem: 18
- delivery_problem: 25
- general_inquiry: 18
- order_cancellation: 20
- order_tracking: 28
- payment_problem: 20
- product_problem: 22
- refund_request: 24
- technical_problem: 16

## Baselines

Two simple baselines can be computed directly from the golden-set class distribution:

1. **Majority-class baseline:** always predict `order_tracking` (the largest class, 28/191) → **14.66% expected accuracy**.
2. **Uniform random baseline:** choose one of the 9 intents uniformly → **11.11% expected accuracy**.

The prototype therefore improves accuracy by roughly **49.21 percentage points over majority-class** and **52.76 points over uniform random**.

These are deliberately simple baselines. A stronger benchmark should add a TF-IDF + linear classifier and/or a zero-shot LLM classifier with the same test set.

## How to run the evaluation harness

```bash
python evaluation_harness.py \
  --golden amazon_support_intent_golden_eval.csv \
  --predictions golden_predictions.csv
```

`golden_predictions.csv` must contain:

```text
message_id,message,true_intent,predicted_intent
```

To evaluate generated replies with an LLM judge, add a JSONL file with one record per case:

```json
{"id":"1","customer_message":"...","reply":"...","human_scores":{"helpfulness":4,"groundedness":4,"tone":5,"actionability":4,"policy_safety":5},"human_overall":4}
```

Then run:

```bash
export GEMINI_API_KEY="..."
python evaluation_harness.py \
  --golden amazon_support_intent_golden_eval.csv \
  --predictions golden_predictions.csv \
  --replies replies.jsonl \
  --judge-model gemini-3.5-flash-lite
```

The harness computes automated classification metrics and, when human scores exist, reports judge-vs-human agreement using exact-score agreement, mean absolute error, and Spearman rank correlation for the overall score.

## LLM-as-judge rubric

Score each generated reply from **1 (poor) to 5 (excellent)** on:

| Dimension | 1 | 3 | 5 |
|---|---|---|---|
| Helpfulness | Does not address issue | Partially useful | Directly addresses the customer’s need |
| Groundedness | Invents unsupported facts | Mostly grounded | Uses only customer/retrieved evidence |
| Tone | Off-brand, rude, robotic | Acceptable | Empathetic, concise, support-appropriate |
| Actionability | No useful next step | Generic next step | Clear, relevant next action |
| Safety / escalation | Unsafe or misses obvious risk | Borderline | Correctly avoids risky claims and escalation mistakes |

The judge must return structured JSON. The harness stores both the per-dimension scores and a short rationale.

## What the headline number does *not* prove

The 63.87% number is **intent classification accuracy**, not end-to-end customer-reply success. It does not directly measure whether replies are helpful, whether retrieval is correct, whether escalation is optimal, or whether customers would accept the generated response.

It also comes from a 191-example golden set and a model trained on 500 automatically labeled examples. The golden set is therefore useful evidence, but it is not enough to claim production-grade accuracy.


