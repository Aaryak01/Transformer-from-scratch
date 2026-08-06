# Transformer From Scratch

A GPT-style decoder-only transformer — tokenizer, architecture, and
training loop all implemented from scratch in PyTorch, no pretrained
models or Hugging Face model classes involved. Pretrained on Wikitext-103,
then fine-tuned on real SEC 10-K/10-Q filings scraped directly from EDGAR.

<!--
  TODO(Aarya): if you have a two-sentence project summary you want up top
  (matching the style of the Sec-Analyst-LLM README), paste it here.
-->

## What this demonstrates

Everything here was built from the ground up, coded alongside Andrej
Karpathy's ["Let's build GPT from scratch"](https://www.youtube.com/watch?v=kCc8FmEb1nY)
and then extended into a full pretrain → fine-tune pipeline on a real,
non-toy domain:

- **A byte-pair-encoding tokenizer**, trained from scratch on a mixed
  corpus (no pretrained tokenizer, no reused vocabulary).
- **A decoder-only transformer architecture** — multi-head self-attention
  with causal masking, residual connections, pre-layer-norm, feed-forward
  blocks — implemented directly with `torch.nn` primitives (see
  [`model.py`](model.py)), not assembled from a library's pre-built model
  class.
- **A real pretraining run**: next-token prediction on Wikitext-103,
  with checkpointing/resume (Colab sessions time out) and loss tracked to
  convergence.
- **A real fine-tuning run**: continuing training on a narrow, genuinely
  different domain (SEC filings) at a deliberately lower learning rate,
  with visible, measurable domain adaptation in both the loss curve and
  the generated text.

## Architecture

| | |
|---|---|
| Parameters | **33.5M** |
| Embedding dimension (`n_embd`) | 512 |
| Attention heads (`n_head`) | 8 |
| Transformer blocks (`n_layer`) | 8 |
| Context length (`block_size`) | 256 tokens |
| Dropout | 0.2 |
| Vocabulary size | 8,000 (custom-trained BPE) |
| Optimizer | AdamW |

Full implementation, with comments explaining each component, in
[`model.py`](model.py) — `Head` → `MultiHeadAttention` → `FeedForward` →
`Block` → `GPTLanguageModel`.

## Training process

**1. Data collection** ([`01_data_collection.ipynb`](01_data_collection.ipynb))
Wikitext-103 (1.8M rows, via Hugging Face `datasets`) for pretraining, plus
158 real SEC 10-K/10-Q filings (10 companies, filed 2022 onward) scraped
directly from SEC EDGAR for tokenizer training and fine-tuning — 34.8MB of
real filing text, not a toy dataset.

**2. Tokenizer training** ([`02_tokenizer.ipynb`](02_tokenizer.ipynb))
BPE tokenizer trained from scratch on the SEC filings plus a Wikitext-103
sample, vocabulary size 8,000.

**3. Pretraining** ([`03_pretrain.ipynb`](03_pretrain.ipynb))
37.5M Wikitext-103 tokens, 90/10 train/val split, trained to step 14,999
(lr 3e-4). **Final loss: train 3.7288, val 3.8494.** Sample generation at
this stage shows correct subword structure and rough Wikipedia-style
phrasing, but not long-range coherence — expected at 33.5M parameters and
this data scale.

**4. Fine-tuning** ([`04_finetune.ipynb`](04_finetune.ipynb))
Continuing from the step-14999 pretrained checkpoint, fine-tuned on 7.4M
SEC-filing tokens only, 3,000 steps at lr 3e-5 (10x lower than
pretraining — nudge, don't overwrite). **Loss dropped from 5.7818 → 2.7697**
(train) over the run, with the sharpest drop in the first 250 steps. Given
the start of a real filing sentence, the fine-tuned model continues with
financially-plausible vocabulary and structure — dollar figures,
fiscal-period references, legal/compliance phrasing — that only appears in
the SEC-filing corpus, not the Wikipedia pretraining data. Full generated
samples and the corpus quality analysis (91.2% prose / 8.8% table-like
content) are in that notebook, run and saved with real output.

## Bonus: Q&A format fine-tuning

[`05_qa_finetune.ipynb`](05_qa_finetune.ipynb) is a second fine-tuning
experiment starting from the same pretrained checkpoint, this time on
[SQuAD](https://huggingface.co/datasets/squad) (~8M tokens of
`"Question: ...\nAnswer: ..."` pairs from real Wikipedia articles) instead
of SEC filings. It tests something different from the SEC run: not domain
vocabulary adaptation, but whether the model can learn a *format* — given
a question, produce an answer-shaped continuation and stop — as a
question distinct from whether that answer is *correct*.

**Real results (3,000 steps, Colab T4):** loss dropped from train 4.93 →
2.48, val 4.94 → 2.64, with the sharpest drop in the first 250 steps and
train/val staying close throughout (no overfitting). Format learning was a
**partial success** — output is short and actually terminates, unlike the
base/SEC-tuned model's tendency to ramble — but several generations embed
a second, hallucinated `Question:`/`Answer:` pair mid-response, so what it
learned is closer to *"this text involves Q&A exchanges"* than the
narrower *"stop cleanly after exactly one answer."* Factual accuracy
**did not work**, as expected at this scale: `"In what year did World War
II end?"` produced `"1950"` — plausible-shaped (a number, roughly the
right era), factually wrong. Most other answers, on both SQuAD-style and
deliberately novel questions, read closer to incoherent word combinations
than real answers.

**Honest bottom line:** this demonstrates format learning, not question
answering. The model reliably produces short, answer-shaped, terminated
output instead of unbounded rambling — that part worked. It does not
reliably produce correct, or even generally coherent, answers. Those are
separate claims; this run is real evidence for the first and against the
second. Full write-up, with the reasoning behind each conclusion, is in
the notebook's own results section.

## What this is *not*

Being upfront about scope, the same way the application README this model
was later used in is: at 33.5M parameters and single-digit millions of
fine-tuning tokens, this is nowhere near a production-quality language
model. Generated text shows clear domain adaptation but not fluent,
factually-reliable output — that's expected at this scale and wasn't the
point. The point was building every layer of the pipeline (tokenizer,
attention, training loop, checkpointing, fine-tuning) by hand and
understanding it end to end.

## Relationship to the SEC Filing Analyst application

This model was originally built as the technical foundation for
[**Sec-Analyst-LLM**](https://github.com/Aaryak01/Sec-Analyst-LLM), a
full-stack RAG chatbot for querying real SEC filings. Being fully honest
about how the two connect: **that application's live responses currently
come from a TF-IDF retrieval pipeline + Cohere's hosted LLM (`command-a`),
not from this from-scratch model.** This repo exists on its own because
that's genuinely a separate, complete piece of work — "I built and trained
a transformer from scratch" is true and demonstrated end-to-end here,
independent of which model answers a given production request in the
deployed app.

## Reproducing this

Every notebook was run in Google Colab with a T4 GPU (Colab's free tier).
To reproduce:

1. Open each notebook (`01` → `04`, in order — `05` is optional, branches
   off after `03`) in Colab.
2. Runtime → Change runtime type → select a GPU (T4 is sufficient).
3. Mount your own Google Drive when prompted (each notebook does this) —
   data, tokenizer, and checkpoints are saved there between sessions since
   Colab's local disk is wiped when a session ends.
4. Run top to bottom. Pretraining (`03`) takes a few hours to reach step
   15,000 on a T4; both fine-tuning notebooks (`04`, `05`) are much faster
   (3,000 steps on a single-digit-millions-of-tokens corpus each).

Or adapt the notebooks to run locally / on your own GPU — the only
Colab-specific lines are the `drive.mount(...)` calls and `!pip install`
cells at the top of each notebook; swap the `/content/drive/MyDrive/...`
paths for local ones and everything else runs as-is.

**Checkpoint weights aren't included in this repo** (`model_step14999.pt`
is 419MB, over GitHub's 100MB file limit) — re-run `03_pretrain.ipynb` to
regenerate the pretrained checkpoint, then `04_finetune.ipynb` for the
fine-tuned one. `tokenizer/tokenizer.json` (484KB) *is* included, since
retraining it isn't necessary to load or continue training the model
architecture, and it's small enough to commit directly.

## Repo layout

```
.
├── model.py                   Architecture: Head, MultiHeadAttention, FeedForward, Block, GPTLanguageModel
├── 01_data_collection.ipynb   Wikitext-103 download + SEC EDGAR scraping
├── 02_tokenizer.ipynb         BPE tokenizer training (vocab_size=8000)
├── 03_pretrain.ipynb          Pretraining on Wikitext-103
├── 04_finetune.ipynb          Fine-tuning on SEC filings
├── 05_qa_finetune.ipynb       Bonus: Q&A format fine-tuning on SQuAD (real results -- see above)
├── tokenizer/tokenizer.json   Trained tokenizer (checkpoints not included -- see above)
└── requirements.txt
```

## Tech stack

PyTorch · Hugging Face `tokenizers` (BPE training) · Hugging Face
`datasets` (Wikitext-103, SQuAD) · `requests` + `beautifulsoup4` (SEC
EDGAR scraping) · `pandas` · trained on Google Colab (T4 GPU)

## License

MIT — see [LICENSE](LICENSE).
