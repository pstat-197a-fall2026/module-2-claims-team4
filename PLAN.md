# Project Plan

Due: Friday, Nov 13

## Who does what

| Member | Tasks |
|---|---|
| Member 1 | Preliminary tasks 1 and 2 (add header text to the scraping; bigram model). Share the cleaned data with the group. |
| Member 2 | Preliminary task 3 (a simple neural network that beats the task 1 model). Write the first summary (tasks 1–3). |
| Member 3 | Main task: neural networks for binary and multiclass, in Python. Save the models and write a prediction script. |
| Member 4 | Other models to compare against (SVM, logistic regression). Make the accuracy, sensitivity and specificity tables. |
| Member 5 | Build the final predictions file. Write the second summary. Final cleanup and check. |

## Things everyone should use

- **Same data split:** `set.seed(110122)` and an 70/10/20 split. Save the IDs to `data/split-ids.csv` so everyone tests on the same pages.
- **Same cleaned data:** Member 1 updates `scripts/preprocessing.R` to include headers and saves:
  - `data/claims-clean.RData` (training data)
  - `data/claims-test-clean.RData` (test data)
  - CSV copies of both for Python
- **Branches:** each person works on their own branch and merges into `main` when ready.

## Python setup (main task)

- Make a virtual environment and install `tensorflow`, `pandas`, `scikit-learn` and `pyreadr`. Add `.venv/` to `.gitignore` and commit a `requirements.txt`.
- `scripts/nn_train.py`: trains the models. Try these in order:
  1. A simple model like the template's
  2. Add bigrams, dropout and early stopping
  3. An LSTM or GRU
- Save the models to `results/nn-binary.keras` and `results/nn-multiclass.keras`.
- `scripts/nn_predict.py`: makes test predictions and saves them to `results/nn-test-preds.csv`.
- `scripts/prediction.R`: reads that CSV and saves `results/preds-group4.RData` with columns `.id`, `bclass.pred` and `mclass.pred`.

## Timeline

| Date | Goal |
|---|---|
| Fri 11/6 | Meet and agree on the data split and branches |
| Sat 11/7 | Cleaned data shared |
| Sat 11/7 – Mon 11/9 | Preliminary tasks and model experiments |
| Tue 11/10 | Pick the final binary and multiclass models |
| Wed 11/11 | Save the final models and make test predictions |
| Thu 11/12 | Write and render both summaries |
| Fri 11/13 | Final review and submit |

## What we turn in

1. `writeups/tasks-summary.html`: summary of preliminary tasks 1–3
2. `writeups/prediction-summary.html`: summary of the main task
3. `results/preds-group4.RData`: test predictions
4. The saved models in `results/` and the prediction scripts in `scripts/`

## Before submitting

- Run the prediction scripts from a fresh copy of the repo and check that `preds-group4.RData` loads with the right columns and no missing values.
- Render both writeups and check that each is about one page.

## Open questions

- ~~The cleaning step drops some test pages.~~ Resolved: the final pipeline parses every page, so `preds-group4.RData` has a prediction for all 929 test pages.
