# Low-Resource G2P Transformer

Character-level grapheme-to-phoneme conversion for English and Spanish using a
PyTorch encoder-decoder Transformer. The model uses a bidirectional encoder
with grouped-query attention and RoPE, and a causal decoder with cross-attention
over the encoder output.

The accompanying report evaluates monolingual and multilingual systems under
strict train/test grapheme deduplication, with additional memorisation analysis
for overlapping train/test items.

## Repository Layout

```text
.
├── data/                 # English and Spanish TSV splits
├── figures/              # Report figures
├── report/               # ACL LaTeX report and references
├── results/              # Saved metrics, histories, and error analyses
├── scripts/              # SLURM launch scripts
├── src/
│   ├── data.py           # Tokenisation, splits, datasets
│   ├── model.py          # Encoder-decoder Transformer
│   ├── train.py          # Training loop
│   ├── evaluate.py       # Metrics and error analysis
│   ├── analyse.py        # Plotting and summaries
│   └── run.py            # CLI entry point
└── requirements.txt
```

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The modelling code uses only PyTorch and the Python standard library. Pandas and
matplotlib are used for analysis and plotting.

## Data

The repository expects SIGMORPHON-style TSV files in `data/`:

```text
eng_train.tsv   eng_val.tsv   eng_test.tsv
spa_train.tsv   spa_val.tsv   spa_test.tsv
```

Each row contains a grapheme form and a whitespace-tokenised IPA transcription.
The loader also supports nested paths such as `data/en/train.tsv`.

## Reproducing Runs

Run commands from the repository root.

### English

```bash
python src/run.py \
  --data_dir data \
  --lang en \
  --strategy dedup \
  --output_dir runs/en_dedup
```

### Spanish

```bash
python src/run.py \
  --data_dir data \
  --lang spa \
  --strategy dedup \
  --output_dir runs/spa_dedup
```

### Multilingual English + Spanish

```bash
python src/run.py \
  --data_dir data \
  --lang en spa \
  --strategy dedup \
  --multilingual \
  --output_dir runs/multi_dedup
```

### Memorisation Analysis

Use the tag strategy to retain train/test overlap and label test items as seen
or unseen:

```bash
python src/run.py \
  --data_dir data \
  --lang en \
  --strategy tag \
  --output_dir runs/en_tag
```

## Evaluation Outputs

Each run writes:

```text
config.json
history.json
test_metrics.json
error_analysis.json
test_predictions.tsv
src_vocab.txt
tgt_vocab.txt
best_model.pt
top_checkpoints.json
```

Metrics include exact match, phone error rate, weighted edit distance, character
Levenshtein distance, and seen/unseen breakdowns where applicable.

## Analysis

Generate plots and compare completed runs:

```bash
python src/analyse.py --output_dir runs/en_dedup
python src/analyse.py --output_dir runs/en_dedup runs/spa_dedup runs/multi_dedup --compare
```

## Report

The ACL-style report is in:

```text
report/g2p_report.tex
```

The required ACL files are included in `report/`:

```text
acl.sty
acl_natbib.bst
references.bib
```

Compile from inside `report/`:

```bash
pdflatex g2p_report.tex
bibtex g2p_report
pdflatex g2p_report.tex
pdflatex g2p_report.tex
```

## Saved Results

The `results/` directory contains the metrics and analysis files used for the
submitted report. Large checkpoint files are not included.
