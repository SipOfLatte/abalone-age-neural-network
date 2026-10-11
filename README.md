# Abalone Age Classification with Dense Neural Networks

Classifies abalone into four age groups from physical measurements (UCI Abalone
data) with dense neural networks in scikit-learn, and studies how the number of
hidden neurons, the learning rate, the number of hidden layers and the optimiser
(SGD vs Adam) affect performance. The full write-up is in
[Abalone_NN_Report.pdf](Abalone_NN_Report.pdf).

## Files

| Path | Contents |
|------|----------|
| `model.py` | Main code. Reads and cleans the abalone data, creates the four ring-age classes, makes the fixed 60/40 stratified split, and runs Steps 2 to 6 (hidden neurons, learning rate, number of hidden layers, Adam vs SGD, final model with confusion matrix and per-class ROC/AUC). Results are written to `results/`. |
| `plots.py` | Produces every figure used in the report from the data and the files in `results/`. Figures are saved to `figures/`. |
| `data/` | `abalone.data` and `abalone.names` from the UCI repository. |
| `results/` | Per-run results, summaries, per-epoch training histories and the final model results from the 10-run study used in the report. |
| `figures/` | Figures used in the report (PDF and PNG). |
| `report/` | LaTeX source of the report (IEEE conference template). |

## Running

```bash
python model.py                  # one run per configuration
python model.py --runs 10        # setting used for the report
python model.py --convergence    # adds the 1000-epoch trial used to pick the training budget
python plots.py                  # after model.py has been run with --convergence
```

`model.py` reads the raw UCI file from `data/abalone.data`, or from the path
given with `--data`. Both scripts are run from the project folder.

Requirements: Python 3.9+, numpy, pandas, scipy, scikit-learn, matplotlib.

## Notes

- Age in years = rings + 1.5. Classes: 1 (age <= 7), 2 (7 < age <= 10),
  3 (10 < age <= 15), 4 (age > 15).
- The train/test split uses `random_state` 42 for every experiment; run r of a
  configuration uses `random_state` r for the initial weights and batch order.
- The optimal setting found in each step is passed to the next step
  automatically, so a single run may pick different values from the 10-run
  study reported. For example, one run selects 15 hidden neurons, while the
  mean over ten runs selects 20.
