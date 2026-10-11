"""
Abalone ring-age classification (4 classes) with dense neural networks.

Usage
python model.py                          # N_RUNS = 1 for submission
python model.py --runs 10                # for report
python model.py --runs 1 --convergence   # also run the epoch-budget trial

"""

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             confusion_matrix, f1_score, log_loss, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Training stops after a fixed number of epochs on purpose,
# scikit-learn's "did not converge" warning is expected and only adds noise.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# settings
N_RUNS = 1              # runs per configuration
EPOCHS = 300            # training length chosen from the 1000 epoch convergence trial
BATCH_SIZE = 32         # examples per SGD/Adam update (about 78 updates per epoch)
SPLIT_SEED = 42         #fix seed for consistency
TEST_SIZE = 0.40  # 60/40 training split

HIDDEN_NEURONS = [5, 10, 15, 20]                     
LEARNING_RATES = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]  
ADAM_LR = 0.001                                      # Adam's usual default rate

COLUMNS = ["Sex", "Length", "Diameter", "Height", "Whole_weight",
           "Shucked_weight", "Viscera_weight", "Shell_weight", "Rings"]
NUMERIC = COLUMNS[1:8]  # the seven continuous measurements
CLASS_NAMES = ["Class 1 (age <= 7)", "Class 2 (7 < age <= 10)",
               "Class 3 (10 < age <= 15)", "Class 4 (age > 15)"]


# Data

def load_data(path):
    df = pd.read_csv(path, header=None)
    # UCI file starts straight away with a sex code
    # Anything else in the first cell means the first row is a header !
    if str(df.iloc[0, 0]).strip().upper() not in {"M", "F", "I"}:
        df = pd.read_csv(path)
    df = df.iloc[:, -len(COLUMNS):]     # ignore a leading index column if present
    df.columns = COLUMNS
    df["Sex"] = df["Sex"].astype(str).str.strip().str.upper()
    return df


def clean_data(df):
    """Drop missing/duplicate rows and impossible or extreme heights. 

    Two specimens have Height = 0, and two lie far beyond Tukey's outer
    fence (Q3 + 3 IQR): 0.515 and 1.13, against a 99th percentile of about
    0.22. (mentioned in report). These are treated as recording errors and removed.
    """
    report = {"raw_rows": len(df),
              "missing": int(df.isna().sum().sum()),
              "duplicates": int(df.duplicated().sum())}
    df = df.dropna().drop_duplicates()
    q1, q3 = df["Height"].quantile([0.25, 0.75])
    bad_height = (df["Height"] <= 0) | (df["Height"] > q3 + 3 * (q3 - q1))
    report["bad_height"] = int(bad_height.sum())
    df = df.loc[~bad_height].reset_index(drop=True)
    report["clean_rows"] = len(df)
    return df, report


def add_age_class(df):
    #Convert rings to age in years (+1.5) and bin into the four classes.

    df = df.copy()
    df["Age"] = df["Rings"] + 1.5
    conditions = [df["Age"] <= 7, df["Age"] <= 10, df["Age"] <= 15]
    df["Class"] = np.select(conditions, [0, 1, 2], default=3)
    return df


def prepare_features(df):
    # 10 network inputs, split 60/40 and standardise

    X = pd.get_dummies(df[["Sex"] + NUMERIC], columns=["Sex"], dtype=float)
    y = df["Class"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SPLIT_SEED)

    scaler = StandardScaler().fit(X_train[NUMERIC])
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[NUMERIC] = scaler.transform(X_train[NUMERIC])
    X_test[NUMERIC] = scaler.transform(X_test[NUMERIC])
    return (X_train.to_numpy(), X_test.to_numpy(), y_train, y_test,
            list(X.columns))



# Model and training

def build_model(hidden, solver="sgd", lr=0.01, seed=0):
    #dense network: ReLU hidden layer and a softmax output with 4 neurons

    return MLPClassifier(hidden_layer_sizes=hidden, activation="relu",
                         solver=solver, learning_rate_init=lr,
                         learning_rate="constant", momentum=0.0,
                         nesterovs_momentum=False, batch_size=BATCH_SIZE,
                         alpha=0.0, shuffle=True, random_state=seed)


def train_model(model, data, epochs=EPOCHS):
    # Train for a fixed number of epochs and record the learning curves.

    X_train, X_test, y_train, y_test = data
    classes = np.unique(y_train)
    history = np.zeros((epochs, 4))   # columns: train_loss, test_loss, train_acc, test_acc
    for epoch in range(epochs):
        model.partial_fit(X_train, y_train, classes=classes)
        p_train = model.predict_proba(X_train)
        p_test = model.predict_proba(X_test)
        history[epoch] = [log_loss(y_train, p_train, labels=classes),
                          log_loss(y_test, p_test, labels=classes),
                          accuracy_score(y_train, p_train.argmax(1)),
                          accuracy_score(y_test, p_test.argmax(1))]
    return model, history

# 


def evaluate(model, data):
    # Accuracy and macro-F1 train/test evaluation

    X_train, X_test, y_train, y_test = data
    pred_train, pred_test = model.predict(X_train), model.predict(X_test)
    return {"train_acc": accuracy_score(y_train, pred_train),
            "test_acc": accuracy_score(y_test, pred_test),
            "train_f1": f1_score(y_train, pred_train, average="macro"),
            "test_f1": f1_score(y_test, pred_test, average="macro")}


def mean_ci(values, level=0.95):
 
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        return values.mean(), 0.0
    t = stats.t.ppf(0.5 + level / 2, n - 1)
    return values.mean(), t * values.std(ddof=1) / np.sqrt(n)


def run_experiment(name, data, hidden, solver, lr, n_runs, epochs=EPOCHS):

    rows, histories, models = [], [], []
    for run in range(n_runs):
        model = build_model(hidden, solver, lr, seed=run)
        model, history = train_model(model, data, epochs)
        rows.append({"config": name, "hidden": str(hidden), "solver": solver,
                     "lr": lr, "run": run, **evaluate(model, data)})
        histories.append(history)
        models.append(model)
    runs = pd.DataFrame(rows)
    summary = summarise(runs)
    print(f"  {name:<28} train {summary['train_acc_mean']:.4f} "
          f"+/- {summary['train_acc_ci']:.4f}   test "
          f"{summary['test_acc_mean']:.4f} +/- {summary['test_acc_ci']:.4f}")
    return runs, summary, np.stack(histories), models


def summarise(runs):
   
    out = {k: runs[k].iloc[0] for k in ["config", "hidden", "solver", "lr"]}
    out["n_runs"] = len(runs)
    for metric in ["train_acc", "test_acc", "train_f1", "test_f1"]:
        m, ci = mean_ci(runs[metric])
        out[f"{metric}_mean"], out[f"{metric}_ci"] = m, ci
    return out
 #One summary row per configuration


# Steps 2 to 6

class Study:
    """
    runs step 2-6, setting with the highest mean test accuracy is kept and
    handed to the next step, as the brief describes. Since no validation set, the test set is used for
    these choices.
    """

    def __init__(self, data, n_runs, out_dir, epochs=EPOCHS):
        self.data, self.n_runs, self.out, self.epochs = data, n_runs, out_dir, epochs
        self.summaries = []

    def _run(self, step, name, hidden, solver, lr):
        #Run one configuration, remember its summary and save its learning curves for plotting
        runs, summary, hist, models = run_experiment(
            name, self.data, hidden, solver, lr, self.n_runs, self.epochs)
        runs.insert(0, "step", step)
        summary = {"step": step, **summary}
        self.summaries.append(summary)
        np.save(os.path.join(self.out, f"history_{name}.npy"), hist)
        return runs, summary, models

    def _save_runs(self, step, frames):
        pd.concat(frames).to_csv(
            os.path.join(self.out, f"runs_{step}.csv"), index=False)

    def step2_hidden_neurons(self, lr=0.01):
        # one hidden layer of 5, 10, 15 or 20 neurons, trained with SGD at a starting learning rate of 0.01. Returns the best size
        print("\nStep 2: number of hidden neurons (SGD, one hidden layer)")
        frames, best = [], None
        for h in HIDDEN_NEURONS:
            runs, s, _ = self._run("step2", f"sgd_1x{h}_lr{lr}", (h,), "sgd", lr)
            frames.append(runs)
            if best is None or s["test_acc_mean"] > best[1]:
                best = (h, s["test_acc_mean"])
        self._save_runs("step2", frames)
        print(f"  -> optimal hidden neurons: {best[0]}")
        return best[0]

    def step3_learning_rate(self, h):

        """the SGD learning rate, with the best size from previous step.
        Returns the best rate and its test accuracy for step 4 """


        print(f"\nStep 3: learning rate (SGD, one hidden layer of {h})")
        frames, best = [], None
        for lr in LEARNING_RATES:
            runs, s, _ = self._run("step3", f"sgd_1x{h}_lr{lr}", (h,), "sgd", lr)
            frames.append(runs)
            if best is None or s["test_acc_mean"] > best[1]:
                best = (lr, s["test_acc_mean"])
        self._save_runs("step3", frames)
        print(f"  -> optimal learning rate: {best[0]}")
        return best[0], best[1]

    def step4_hidden_layers(self, h, lr, one_layer_acc):
        """Step 4 adds a second hidden layer of the same size, using the best
        learning rate. """
        #
        print(f"\nStep 4: two hidden layers ({h}, {h}), SGD lr={lr}")
        runs, s, _ = self._run("step4", f"sgd_2x{h}_lr{lr}", (h, h), "sgd", lr)
        self._save_runs("step4", [runs])
        hidden = (h, h) if s["test_acc_mean"] > one_layer_acc else (h,)
        print(f"  -> selected architecture: {hidden}")
        return hidden

    def step5_optimiser(self, hidden, lr):
        #Adam against SGD on the chosen architecture.

        print(f"\nStep 5: Adam vs SGD, hidden layers {hidden}")
        tag = "x".join(map(str, hidden))
        layers = f"{len(hidden)}x{hidden[0]}"
        settings = [("sgd", lr), ("adam", ADAM_LR)]
        if lr != ADAM_LR:   # skip the duplicate when the two rates are the same
            settings.append(("adam", lr))
        frames, results = [], {}
        for solver, rate in settings:
            name = f"{solver}_{layers}_lr{rate}"
            runs, s, _ = self._run("step5", name, hidden, solver, rate)
            frames.append(runs)
            results[(solver, rate)] = s["test_acc_mean"]
        self._save_runs("step5", frames)
        solver, rate = max(results, key=results.get)
        print(f"  -> selected optimiser: {solver} (lr={rate}), layers {tag}")
        return solver, rate

    def step6_final_model(self, hidden, solver, lr):
        #final evaluation
        print(f"\nStep 6: final model {hidden}, {solver}, lr={lr}")
        X_train, X_test, y_train, y_test = self.data
        layers = f"{len(hidden)}x{hidden[0]}"
        runs, s, models = self._run("step6", f"final_{solver}_{layers}_lr{lr}",
                                    hidden, solver, lr)
        self._save_runs("step6", [runs])

        # One-vs-rest AUC for every class and every run against the other three using that class's predicted probability.
        aucs = []
        for mdl in models:
            proba = mdl.predict_proba(X_test)
            aucs.append([auc(*roc_curve(y_test == c, proba[:, c])[:2])
                         for c in range(4)])
        aucs = np.array(aucs)

        # Representative run for the confusion matrix and ROC curves.
        rep = int(np.argmin(np.abs(runs["test_acc"] - s["test_acc_mean"])))
        model = models[rep]
        proba = model.predict_proba(X_test)
        pred = proba.argmax(1)
        roc = {}
        for c in range(4):
            fpr, tpr, _ = roc_curve(y_test == c, proba[:, c])
            roc[c] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(),
                      "auc": float(auc(fpr, tpr))}
        result = {
            "hidden": list(hidden), "solver": solver, "lr": lr,
            "representative_run": rep,
            "test_accuracy": float(accuracy_score(y_test, pred)),
            "train_accuracy": float(accuracy_score(y_train, model.predict(X_train))),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
            "classification_report": classification_report(
                y_test, pred, target_names=CLASS_NAMES, output_dict=True,
                zero_division=0),
            "roc": roc,
            "auc_mean": [mean_ci(aucs[:, c])[0] for c in range(4)],
            "auc_ci": [mean_ci(aucs[:, c])[1] for c in range(4)],
        }
        with open(os.path.join(self.out, "final_model.json"), "w") as f:
            json.dump(result, f, indent=2)

        print(f"  test accuracy (mean of runs): {s['test_acc_mean']:.4f} "
              f"+/- {s['test_acc_ci']:.4f}")
        print("  confusion matrix (representative run):")
        print(np.array(result["confusion_matrix"]))
        for c in range(4):
            print(f"  AUC {CLASS_NAMES[c]:<26} {result['auc_mean'][c]:.4f} "
                  f"+/- {result['auc_ci'][c]:.4f}")
        print(classification_report(y_test, pred, target_names=CLASS_NAMES,
                                    zero_division=0, digits=4))
        return result

    def save_summary(self):
        pd.DataFrame(self.summaries).to_csv(
            os.path.join(self.out, "summary.csv"), index=False)


def convergence_trial(data, out_dir, epochs=1000, n_runs=3):
    #Decide how many epochs to train for, test for overfitting 

    print(f"\nConvergence trial ({epochs} epochs, {n_runs} runs each)")
    settings = [("trial_sgd_1x10_lr0.01", (10,), "sgd", 0.01),
                ("trial_sgd_1x10_lr0.1", (10,), "sgd", 0.1),
                ("trial_adam_1x10_lr0.001", (10,), "adam", 0.001)]
    for name, hidden, solver, lr in settings:
        hist = np.stack([train_model(build_model(hidden, solver, lr, seed=r),
                                     data, epochs)[1] for r in range(n_runs)])
        np.save(os.path.join(out_dir, f"history_{name}.npy"), hist)
        mean = hist.mean(0)
        best = int(mean[:, 1].argmin())
        print(f"  {name:<26} lowest test loss at epoch {best + 1} "
              f"(test acc {mean[best, 3]:.4f}); epoch {epochs}: "
              f"test acc {mean[-1, 3]:.4f}")


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--runs", type=int, default=N_RUNS,
                        help="runs per configuration (default 1)")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="training epochs per run (default 300)")
    parser.add_argument("--data", default="data/abalone.data", help="path to the abalone data file")
    parser.add_argument("--out", default="results", help="folder for the results")
    parser.add_argument("--convergence", action="store_true",
                        help="run the epoch-budget trial before the study")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # load, clean, label and report the class balance.
    df, report = clean_data(load_data(args.data))
    df = add_age_class(df)
    print(f"Data: {args.data}  {report}")
    counts = df["Class"].value_counts().sort_index()
    for c, n in counts.items():
        print(f"  {CLASS_NAMES[c]:<26} {n:5d} ({n / len(df):.1%})")

    X_train, X_test, y_train, y_test, features = prepare_features(df)
    data = (X_train, X_test, y_train, y_test)
    print(f"Train {X_train.shape}, test {X_test.shape}; features: {features}")

    if args.convergence:
        convergence_trial(data, args.out)

    # Steps 2 to 6 each using the best setting found in the step before.
    study = Study(data, args.runs, args.out, args.epochs)
    h = study.step2_hidden_neurons()
    lr, acc_1layer = study.step3_learning_rate(h)
    hidden = study.step4_hidden_layers(h, lr, acc_1layer)
    solver, rate = study.step5_optimiser(hidden, lr)
    study.step6_final_model(hidden, solver, rate)
    study.save_summary()


if __name__ == "__main__":
    main()
