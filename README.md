# Which Regression Model Predicts Student Exam Scores Best — and Which One Survives Messy Data?

My first machine learning research project. As an IGCSE student whose university
admission depends on predicted grades, I wanted to know if AI regression models
can actually predict student scores reliably — and what happens to them when the
data is dirty, like real-life data always is.

## The experiment

Three models — **multiple linear regression**, **random forest**, and **SVR** —
tested on three versions of the same 5,000-student dataset:

1. **Clean** — a perfect, zero-errors version (ground truth)
2. **Cleaned** — a corrupted version repaired by my cleaning pipeline
   (duplicates, impossible values, inconsistent labels, missing data)
3. **Raw** — the corrupted version with no cleaning at all

Same pipeline, same 80/20 split, same metrics (MAE, R²) every time — only the
data changes. A dummy regressor (always predicts the mean) is the baseline in
every condition.

## Main findings

- **Three categories, three different winners.** Linear regression won accuracy
  (MAE 6.63 clean / 6.84 cleaned — and 6.40 with a hand-made interaction
  feature). Random forest won robustness: on raw dirty data the ranking
  flipped — forest 7.13 vs linear 8.68 MAE. SVR won recovery (92% of its
  damage was repaired by cleaning).
- **Studying pays more when you sleep.** The effect of study hours on scores is
  much steeper for students sleeping 6.5+ hours — an interaction that a single
  scatter plot can't show, and that improved linear regression when added as a
  feature.
- **Cleaning matters.** My pipeline recovered ~90% of the damage that dirty
  data caused to the value-based models.
- **Hyperparameters matter more than the algorithm.** SVR ranged from 7.0 to
  9.9 MAE across five C values — a bigger gap than between the algorithms
  themselves.

**Takeaway:** model choice depends on data quality and data shape, not
sophistication. If you can clean your data, a simple model wins; if you're
stuck with dirt, the robust forest is safer.

## Files

| File | What it is |
|---|---|
| `research_paper.pdf` | The full write-up with figures |
| `research_notebook.ipynb` | All code: cleaning, models, experiments |
| `student_performance_CLEAN.csv` | The perfect twin dataset (5,000 students) |
| `student_performance_research.csv` | The corrupted twin (5,050 rows, errors included) |

## How to run

Open the notebook in [Google Colab](https://colab.research.google.com), upload
both CSVs (folder icon → upload), then Runtime → **Run all**. Requires only
pandas, numpy, matplotlib, and scikit-learn (all preinstalled in Colab).

## Honesty note

The datasets are synthetic (AI-generated with deliberately injected, known
corruption) — that's what makes the controlled clean-vs-dirty comparison
possible, and it's also the main limitation: real-world dirt is unknown and
nastier. AI assistance was used for debugging, verification, and language
editing; all experiments, decisions, and findings are my own.
