# Big Data Classification — Titanic Survival Prediction

> **Course project:** Selected Topics in Information Systems · Mansoura University · Faculty of Computers and Information · 2025–2026
>
> **Stack:** Apache Spark · PySpark · Python · Anaconda · Decision Tree Classifier
>
> **Dataset:** Titanic (891 passenger records)

A complete, beginner-friendly Big Data project that trains a **Decision Tree Classifier** on the famous Titanic dataset using **Apache Spark (PySpark)** to predict whether a passenger survived the disaster. The single-file pipeline (`main.py`) walks through every stage of a real-world ML workflow — data loading, exploratory analysis, cleaning, feature engineering, train/test split, training, evaluation, tree visualization, and persisting predictions to disk.

---

## Table of Contents

1. [Project at a Glance](#1-project-at-a-glance)
2. [Team](#2-team)
3. [What This Project Does](#3-what-this-project-does)
4. [Tech Stack & Why](#4-tech-stack--why)
5. [Repository Structure](#5-repository-structure)
6. [Quick Start](#6-quick-start)
7. [Detailed Setup (Windows)](#7-detailed-setup-windows)
8. [Detailed Setup (macOS / Linux)](#8-detailed-setup-macos--linux)
9. [Running the Pipeline](#9-running-the-pipeline)
10. [Pipeline Walkthrough (Step by Step)](#10-pipeline-walkthrough-step-by-step)
11. [The Dataset](#11-the-dataset)
12. [Understanding the Results](#12-understanding-the-results)
13. [Output Files](#13-output-files)
14. [Why Apache Spark?](#14-why-apache-spark)
15. [How a Decision Tree Works](#15-how-a-decision-tree-works)
16. [Configuration & Tunable Parameters](#16-configuration--tunable-parameters)
17. [Troubleshooting](#17-troubleshooting)
18. [References](#18-references)
19. [License & Acknowledgements](#19-license--acknowledgements)

---

## 1. Project at a Glance

| | |
|---|---|
| **Problem type** | Binary classification (survived vs. did not survive) |
| **Algorithm** | Decision Tree Classifier (`pyspark.ml.classification.DecisionTreeClassifier`) |
| **Dataset** | Titanic — 891 rows × 12 columns (`data/titanic.csv`) |
| **Train / Test split** | 80% / 20% (deterministic, `seed=42`) |
| **Tree max depth** | 5 |
| **Metrics produced** | Accuracy, weighted Precision, weighted Recall, F1 score |
| **Typical accuracy** | 78–84% (range expected for a well-tuned tree on this dataset) |
| **Entry point** | [`main.py`](./main.py) |

---

## 2. Team

| Role | Member |
|---|---|
| Team Member 1 | **Omar Ammar** |
| Team Member 2 | **Kareem Reda** |
| Team Member 3 | **Mohammed El Saed** |

- **University:** Mansoura University
- **Faculty:** Faculty of Computers and Information
- **Course:** Selected Topics in Information Systems
- **Academic Year:** 2025–2026

---

## 3. What This Project Does

`main.py` is a single, self-contained PySpark script that:

1. Starts a local Apache Spark session.
2. Reads `data/titanic.csv` into a Spark DataFrame.
3. Performs exploratory data analysis (schema, summary stats, survival counts, missing-value report).
4. Cleans the data (drops non-predictive columns, fills missing `Age` with the median, fills missing `Embarked` with the mode `"S"`).
5. Engineers features by string-indexing categorical columns (`Sex`, `Embarked`) and assembling a feature vector with `VectorAssembler`.
6. Splits the data 80/20 into training and testing sets with a fixed seed for reproducibility.
7. Trains a `DecisionTreeClassifier` (max depth 5) on the training data.
8. Predicts survival for the held-out test set.
9. Evaluates the model with **Accuracy**, **weighted Precision**, **weighted Recall**, and **F1 score**.
10. Prints the learned decision-tree structure (the human-readable rules).
11. Saves predictions as a single CSV under `output/predictions/` and writes a results summary to `output/results_summary.txt`.
12. Stops the Spark session cleanly.

A friendly `try/except` wrapper turns unexpected failures into actionable error messages with hints for the most common setup mistakes.

---

## 4. Tech Stack & Why

| Tool | Why we use it |
|---|---|
| **Apache Spark** | Industry-standard distributed compute engine. Combines batch processing and ML in one framework, and scales from a laptop to a 1000-node cluster without code changes. |
| **PySpark** | The Python API for Spark. Lets us use Spark's power with Python, which the team already knows — no need to learn Scala or Java. |
| **Python 3.10+** | Readable, with thousands of mature data-science libraries. |
| **Anaconda** | Bundles Python plus 1,500+ scientific packages and a clean isolated prompt. Removes most "it works on my machine" issues on Windows. |
| **Decision Tree** | Interpretable (you can literally print the rules), great for teaching, and a strong baseline for tabular data. Unlike a "black box" neural network, you can *see* the questions the model asks. |
| **Titanic dataset** | Small, clean, real-world, and famous — the perfect first classification problem. |
| **Java JDK 17** | Spark runs on the JVM. JDK 17 is the LTS version recommended for Spark 3.5+. |

---

## 5. Repository Structure

```
Big-Data-Project/
├── data/
│   └── titanic.csv          # Input dataset (891 passengers, included)
├── output/                  # Generated at runtime; ignored by git
│   ├── predictions/         # Single CSV with model predictions
│   └── results_summary.txt  # Plain-text summary of metrics
├── main.py                  # The full PySpark pipeline (entry point)
├── README.md                # This file (the comprehensive guide)
├── README.txt               # Original beginner-style guide (kept for reference)
├── SETUP_GUIDE.txt          # Original Windows-only setup walkthrough
├── team_info.txt            # Team & course metadata
└── .gitignore               # Spark temp files, output/, Python caches, OS junk
```

> **Note:** The `output/` folder is git-ignored. It is created automatically on the first run. The included `output/` directory in this repo is empty by design.

---

## 6. Quick Start

If you already have **Java 17**, **Python 3.10+**, and **PySpark 3.5+** installed:

```bash
git clone https://github.com/0xN0RMXL/Big-Data-Project.git
cd Big-Data-Project
pip install pyspark
python main.py
```

After ~10–30 seconds you'll see:

```
========================================
      MODEL EVALUATION RESULTS
========================================
   Accuracy  : 81.xx%
   Precision : 81.xx%
   Recall    : 81.xx%
   F1 Score  : 81.xx%
========================================
```

Predictions and the summary will be in `output/`.

---

## 7. Detailed Setup (Windows)

Tested on Windows 10 / Windows 11 with Anaconda. For a click-by-click walkthrough see also [`SETUP_GUIDE.txt`](./SETUP_GUIDE.txt).

### 7.1 Install Java JDK 17

1. Download the JDK 17 Windows x64 `.msi` from <https://adoptium.net/temurin/releases/?version=17>.
2. Run the installer; click **Next** through every screen.
3. Set the `JAVA_HOME` environment variable:
   - Start → search **environment variables** → **Edit the system environment variables** → **Environment Variables…**
   - Under **User variables** click **New…**
     - Variable name: `JAVA_HOME`
     - Variable value: `C:\Program Files\Eclipse Adoptium\jdk-17.0.x-hotspot` (use the actual path)
4. Add Java to `Path`:
   - Edit the `Path` user variable → **New** → add `C:\Program Files\Eclipse Adoptium\jdk-17.0.x-hotspot\bin`.
5. **Restart** your terminal (or your machine) so the variables take effect.
6. Verify in a new Command Prompt:
   ```cmd
   java -version
   ```
   You should see `openjdk version "17.0.x"`.

### 7.2 Install Anaconda

1. Download from <https://www.anaconda.com/download> (64-bit Graphical Installer).
2. Run the installer:
   - Choose **Just Me (recommended)**.
   - On "Advanced Options" check **Register Anaconda3 as my default Python**, leave **Add Anaconda3 to my PATH** unchecked.
3. Open **Anaconda Prompt** (NOT regular CMD) from the Start menu. The prompt should show `(base)` at the start.
4. Verify Python:
   ```cmd
   python --version
   ```

### 7.3 Install PySpark

In Anaconda Prompt:

```cmd
pip install pyspark
python -c "import pyspark; print(pyspark.__version__)"
```

Expected output: a version number like `3.5.0`.

### 7.4 Run the project

```cmd
cd "C:\path\to\Big-Data-Project"
python main.py
```

---

## 8. Detailed Setup (macOS / Linux)

### macOS (Homebrew)

```bash
brew install --cask temurin@17
brew install python@3.11
export JAVA_HOME=$(/usr/libexec/java_home -v 17)
python3 -m pip install --user pyspark
python3 main.py
```

### Linux (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install -y openjdk-17-jdk python3 python3-pip
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
python3 -m pip install --user pyspark
python3 main.py
```

For a permanent `JAVA_HOME`, add the `export` line to your `~/.bashrc` or `~/.zshrc`.

### Verify

```bash
java -version          # should print "17.x"
python3 --version      # should print "3.10.x" or newer
python3 -c "import pyspark; print(pyspark.__version__)"
```

---

## 9. Running the Pipeline

From the repository root:

```bash
python main.py
```

What happens:

1. A welcome banner with project metadata is printed.
2. Spark starts (a few seconds; you may see harmless `WARN` lines from Hadoop on Windows).
3. The script prints schema, statistics, and missing-value counts.
4. Cleaning and feature engineering messages appear.
5. Training and prediction run.
6. **A formatted box prints Accuracy / Precision / Recall / F1.**
7. The full decision tree is printed in human-readable form (`model.toDebugString`).
8. Files are written to `output/`.
9. Spark stops; you see `ALL DONE!`.

Exit codes:

- `0` — success.
- `1` — an error occurred. The error message and a list of common fixes are printed.

---

## 10. Pipeline Walkthrough (Step by Step)

`main.py` is annotated with beginner-friendly comments. The 12 logical steps are:

| Step | What it does |
|------|---------------|
| 0 | Print project banner (team, course, algorithm, tool). |
| 1 | Create a `SparkSession` named `"Titanic Decision Tree Classification"`. |
| 2 | Read `data/titanic.csv` with `header=True`, `inferSchema=True`, `mode="DROPMALFORMED"`. |
| 3 | EDA: `describe()`, `groupBy("Survived").count()`, per-column null counts. |
| 4 | Clean: drop `Name`, `Ticket`, `Cabin`, `PassengerId`; fill `Age` with median (`approxQuantile`, fallback `29.0`); fill `Embarked` with `"S"`; drop any remaining nulls. |
| 5 | Feature engineering: `StringIndexer` on `Sex` and `Embarked`; `VectorAssembler` over `[Pclass, Sex_indexed, Age, SibSp, Parch, Fare, Embarked_indexed]`; combine with a `Pipeline`. |
| 6 | `randomSplit([0.8, 0.2], seed=42)` → cache both. |
| 7 | Train `DecisionTreeClassifier(labelCol="Survived", featuresCol="features", maxDepth=5, seed=42)`. |
| 8 | Predict on the test set; show 10 sample rows with actual vs. predicted survival. |
| 9 | Evaluate with `MulticlassClassificationEvaluator` — `accuracy`, `weightedPrecision`, `weightedRecall`, `f1`. |
| 10 | Print the learned tree via `model.toDebugString`. |
| 11 | Save predictions (`coalesce(1).write.mode("overwrite").csv(...)`) and write `results_summary.txt`. |
| 12 | `spark.stop()` and print `ALL DONE!`. |

See the inline comments in [`main.py`](./main.py) for the why behind every line.

---

## 11. The Dataset

`data/titanic.csv` contains 891 passenger records and 12 columns:

| Column | Description | Used as |
|--------|-------------|--------|
| `PassengerId` | Unique row id | Dropped |
| `Survived` | `0` = died, `1` = survived | **Label (target)** |
| `Pclass` | Ticket class (1 / 2 / 3) | Feature |
| `Name` | Full name | Dropped (mostly unique) |
| `Sex` | `male` / `female` | Feature (string-indexed) |
| `Age` | Age in years | Feature (median-imputed) |
| `SibSp` | # siblings / spouses aboard | Feature |
| `Parch` | # parents / children aboard | Feature |
| `Ticket` | Ticket number | Dropped |
| `Fare` | Ticket price | Feature |
| `Cabin` | Cabin code | Dropped (mostly missing) |
| `Embarked` | Port: `C` / `Q` / `S` | Feature (string-indexed, mode-imputed) |

> **Label vs. feature** — a *label* is the answer the model must predict (`Survived`); *features* are the inputs the model uses to predict it.

---

## 12. Understanding the Results

After every run, four numbers are printed:

| Metric | Meaning | Plain-English analogy |
|--------|---------|------------------------|
| **Accuracy** | Fraction of all test passengers classified correctly. | "Out of 100 quiz questions, how many you got right." |
| **Precision** | Of those the model *predicted* survived, what fraction *actually* survived. | "I said 10 people would survive — 8 actually did → 80% precision." |
| **Recall** | Of those who *actually* survived, what fraction the model caught. | "20 people really survived; the model caught 16 → 80% recall." |
| **F1 Score** | Harmonic mean of Precision and Recall — high only when both are high. | "GPA-like average of two grades." |

Sanity check ranges for this dataset and configuration:

- **78–84%** → expected, well-trained tree.
- **~50%** → coin flip; something is broken (data not loaded, label leak, etc.).
- **>95%** → very likely overfitting or data leakage.

---

## 13. Output Files

After a successful run you'll find:

```
output/
├── predictions/
│   ├── _SUCCESS                          # Spark's success marker
│   └── part-00000-<uuid>.csv             # Single CSV with predictions (because of .coalesce(1))
└── results_summary.txt                   # Human-readable metrics summary
```

`results_summary.txt` looks roughly like:

```
═══════════════════════════════════════════════════
           PROJECT RESULTS SUMMARY
═══════════════════════════════════════════════════
Date and Time Run    : 2026-05-02 12:34:56
Dataset Used         : Titanic Dataset (891 rows)
Algorithm            : Decision Tree Classifier
Max Depth            : 5
Train/Test Split     : 80% / 20%
Training Rows        : 712
Testing Rows         : 179
───────────────────────────────────────────────────
Accuracy             : 81.56%
Precision            : 81.42%
Recall               : 81.56%
F1 Score             : 81.39%
───────────────────────────────────────────────────
```

---

## 14. Why Apache Spark?

| | Pandas (single-machine) | PySpark (distributed) |
|---|---|---|
| Best for | Datasets that fit in RAM | Datasets too big for RAM, or that need to scale |
| Memory model | Loads the whole file at once | Streams data in partitions, lazy evaluation |
| Scaling | One process, one machine | Cluster with N executors, automatic parallelism |
| Real-world users | Analysts, notebooks | Netflix, Uber, Amazon, Airbnb |

For this assignment the dataset is tiny (891 rows). The point is to learn the **same APIs and patterns** that you would use on a billion-row dataset. The code in `main.py` would run unchanged on a 100-node cluster — only the input path and the cluster URL would change.

---

## 15. How a Decision Tree Works

A Decision Tree is like the game **20 Questions**. The model asks yes/no questions about a passenger's features, and follows a path through the tree until it reaches a leaf with a prediction:

```
                       ┌───────────────────┐
                       │  Is Sex female?   │
                       └───────┬───────────┘
                               │
                ┌──────────────┴───────────────┐
              YES                              NO
                │                               │
   ┌────────────────────────┐       ┌────────────────────────┐
   │ Is Pclass in {1, 2}?   │       │     Is Age < 5?        │
   └──────────┬─────────────┘       └──────────┬─────────────┘
              │                                 │
       ┌──────┴──────┐                   ┌──────┴──────┐
      YES            NO                 YES            NO
       │              │                  │              │
   SURVIVED         …                SURVIVED          …
```

**Training** = the algorithm searches over every possible split and picks the questions that best separate survivors from non-survivors (using *information gain* / *Gini impurity*). **Testing** = we feed unseen passengers down the tree and compare predictions to ground truth.

`maxDepth=5` is a regularization knob: shallower trees generalize better, deeper trees risk memorizing the training set.

To inspect the actual rules learned by the model, look at the output of `model.toDebugString` printed by Step 10.

---

## 16. Configuration & Tunable Parameters

All knobs live near the top of their respective steps in `main.py`:

| Where | What | Default | What to try |
|------|------|---------|-------------|
| Step 4 | Median fallback for `Age` | `29.0` | `mean` instead of median; or drop rows with missing `Age` |
| Step 5 | Feature columns | 7 cols | Add engineered features (`FamilySize = SibSp + Parch + 1`, `IsAlone`, `Title` from `Name`) |
| Step 6 | Train/test split | `[0.8, 0.2]`, `seed=42` | Try a `[0.7, 0.3]` split or k-fold CV |
| Step 7 | `maxDepth` | `5` | Try `3`, `7`, `10`; observe under-/over-fitting |
| Step 7 | Algorithm | `DecisionTreeClassifier` | Swap for `RandomForestClassifier` or `GBTClassifier` |
| Step 11 | Output partitions | `coalesce(1)` | Remove for production runs on big data |

---

## 17. Troubleshooting

| Error | Likely cause | Fix |
|-------|--------------|-----|
| `'java' is not recognized` | `JAVA_HOME`/`PATH` wrong | Reinstall JDK 17, set env vars, **restart terminal**. |
| `No module named 'pyspark'` | Wrong shell or env | Open **Anaconda Prompt** (not CMD): `pip install pyspark`. |
| `Java heap space` / `OutOfMemoryError` | JVM memory too low | Set `PYSPARK_SUBMIT_ARGS="--driver-memory 2g pyspark-shell"`. |
| `FileNotFoundError: titanic.csv` | Wrong working directory | Run from the project root: `cd Big-Data-Project && python main.py`. |
| `Failed to find Spark jars directory` | Broken PySpark install | `pip uninstall pyspark && pip install pyspark`. |
| Many `part-00000-…csv` files | Spark default partitioning | Already mitigated with `.coalesce(1)`; the single part file is the result. |
| Accuracy ~50% | Data not loaded properly, or feature/label mix-up | Verify schema, ensure `Survived` is **not** in the feature vector. |
| Accuracy >95% | Overfitting / leakage | Lower `maxDepth`, double-check no target leak in features. |
| `WARN NativeCodeLoader: Unable to load native-hadoop library` | Cosmetic warning on Windows | Safe to ignore. |

If something else goes wrong, the script's `except` block prints the exact error message followed by a checklist of the four most common fixes.

---

## 18. References

- **Apache Spark** — <https://spark.apache.org/docs/latest/>
- **PySpark API** — <https://spark.apache.org/docs/latest/api/python/>
- **Spark MLlib (Decision Trees)** — <https://spark.apache.org/docs/latest/ml-classification-regression.html#decision-tree-classifier>
- **Adoptium Temurin (Java 17)** — <https://adoptium.net/temurin/releases/?version=17>
- **Anaconda** — <https://www.anaconda.com/download>
- **Titanic Dataset (mirror)** — <https://raw.githubusercontent.com/datasciencedoc/data/master/titanic.csv>

---

## 19. License & Acknowledgements

This repository is an academic submission for **Selected Topics in Information Systems** at **Mansoura University · Faculty of Computers and Information**, academic year **2025–2026**.

The Titanic dataset is publicly available and widely used for teaching. Thanks to Apache Software Foundation for Spark/PySpark and to the open-source community for the supporting tools.

> "Big Data is not about how much data you have — it's about what you can do with it."
