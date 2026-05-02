"""
Titanic Survival Prediction using Apache Spark (PySpark)
Decision Tree Classification
University Project - Mansoura University
Team: Omar Ammar, Kareem Reda, Mohammed El Saed
"""

# ============================================================
# IMPORTS — bringing in the tools we need from PySpark
# ============================================================

# SparkSession is the entry point to use Spark (like turning the key to start the engine)
from pyspark.sql import SparkSession

# col, when, and lit help us work with individual columns and values in a DataFrame
from pyspark.sql.functions import col, when, lit, count

# These are Spark's machine learning tools for preparing text data into numbers
from pyspark.ml.feature import StringIndexer, VectorAssembler

# Pipeline chains multiple data preparation steps together so they run in order automatically
from pyspark.ml import Pipeline

# DecisionTreeClassifier is the machine learning algorithm we chose for this project
from pyspark.ml.classification import DecisionTreeClassifier

# Evaluators measure how good our trained model is at making predictions
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# datetime helps us record the exact date and time we run the script
from datetime import datetime

# os helps us work with file paths in a way that works on any operating system
import os

# sys lets us exit the program cleanly if something goes wrong
import sys


# ============================================================
# MAIN SCRIPT — wrapped in try/except so errors are friendly
# ============================================================

try:

    # --------------------------------------------------------
    # STEP 0 — Print a welcome banner with project title and team info
    # --------------------------------------------------------
    # \n means "new line" so the banner looks clean in the terminal
    print("\n" + "=" * 60)
    print("   BIG DATA CLASSIFICATION PROJECT")
    print("   Titanic Survival Prediction using Apache Spark")
    print("=" * 60)
    print("   University   : Mansoura University")
    print("   Faculty      : Faculty of Computers and Information")
    print("   Course       : Selected Topics in Information Systems")
    print("   Year         : 2025-2026")
    print("   Team         : Omar Ammar, Kareem Reda, Mohammed El Saed")
    print("   Algorithm    : Decision Tree Classifier")
    print("   Tool         : Apache Spark (PySpark)")
    print("=" * 60 + "\n")


    # --------------------------------------------------------
    # STEP 1 — Start Apache Spark
    # --------------------------------------------------------
    # We create a SparkSession. This is the FIRST thing you must do in every PySpark program.
    # builder lets us configure the session. appName gives our job a name that shows in Spark's UI.
    # getOrCreate() starts Spark if it is not already running, or reuses an existing session.
    spark = SparkSession.builder \
        .appName("Titanic Decision Tree Classification") \
        .getOrCreate()

    # Print a confirmation so the user knows Spark started correctly
    print("[STEP 1] Apache Spark started successfully!")

    # spark.version tells us which version of Spark is running (e.g., 3.5.0)
    print("         Spark version:", spark.version)
    print()


    # --------------------------------------------------------
    # STEP 2 — Load the Dataset
    # --------------------------------------------------------
    # Read the titanic.csv file into a Spark DataFrame.
    # A DataFrame is like a table with rows and columns.
    # header=True means the first row of the CSV contains column names.
    # inferSchema=True tells Spark to guess whether each column is a number, text, etc.
    # option("mode", "DROPMALFORMED") silently skips any rows that do not fit the expected shape.
    df = spark.read.csv("data/titanic.csv", header=True, inferSchema=True, mode="DROPMALFORMED")

    # Let the user know the file was loaded
    print("[STEP 2] Dataset loaded successfully from data/titanic.csv")

    # .show(5) prints the first 5 rows to the screen so we can see what the data looks like
    print("         First 5 rows:")
    df.show(5)

    # .count() returns the total number of rows in the dataset as an integer
    total_rows = df.count()
    print("         Total rows in dataset:", total_rows)

    # .printSchema() shows each column name and its data type (e.g., integer, string, double)
    print("         Schema (column names and types):")
    df.printSchema()
    print()


    # --------------------------------------------------------
    # STEP 3 — Explore the Data (EDA = Exploratory Data Analysis)
    # --------------------------------------------------------
    # .describe() calculates basic statistics: count, mean, stddev, min, max for numeric columns
    print("[STEP 3] Exploratory Data Analysis (EDA)")
    print("         Basic statistics:")
    df.describe().show()

    # groupBy("Survived") groups the rows by the Survived column (0 = died, 1 = survived)
    # .count() then counts how many rows are in each group
    print("         Survival counts:")
    df.groupBy("Survived").count().show()

    # Count missing (null) values in each column.
    # We loop through every column name in df.columns.
    # For each column, we filter where the value is null and count those rows.
    print("         Missing values per column:")
    for column_name in df.columns:
        # col(column_name).isNull() creates a condition that is True when a cell is empty
        missing_count = df.filter(col(column_name).isNull()).count()
        print(f"           {column_name}: {missing_count}")
    print()


    # --------------------------------------------------------
    # STEP 4 — Clean the Data
    # --------------------------------------------------------
    print("[STEP 4] Cleaning the data...")

    # Drop columns that are not useful for predicting survival:
    # Name, Ticket, and Cabin are mostly unique text values that do not help the model learn patterns.
    # PassengerId is just an ID number, not a real feature.
    # drop() returns a new DataFrame without those columns; the original df is not changed.
    df_clean = df.drop("Name", "Ticket", "Cabin", "PassengerId")

    # Find the median Age so we can fill missing Age values with a sensible number.
    # approxQuantile is a Spark method that quickly estimates percentiles on large data.
    # 0.5 = the 50th percentile, which is the median.
    # 0.0 = error tolerance (exact median).
    median_age_list = df_clean.approxQuantile("Age", [0.5], 0.0)

    # approxQuantile returns a list, so we take the first element.
    # If the list is empty (unlikely here), we fall back to 29.0 as a safe default.
    median_age = median_age_list[0] if median_age_list else 29.0

    # fillna() replaces null values with the value we provide.
    # We pass a dictionary {"Age": median_age} so only the Age column is filled.
    df_clean = df_clean.fillna({"Age": median_age})

    # For Embarked (port of departure), most passengers boarded at "S" (Southampton).
    # We fill any missing Embarked values with "S".
    df_clean = df_clean.fillna({"Embarked": "S"})

    # Drop any rows that still have null values in any remaining column.
    # This is a safety step so the machine learning pipeline does not crash.
    df_clean = df_clean.na.drop()

    # Count how many rows remain after cleaning
    remaining_rows = df_clean.count()
    print(f"         Data cleaning complete. Remaining rows: {remaining_rows}")
    print()


    # --------------------------------------------------------
    # STEP 5 — Prepare Features for Machine Learning
    # --------------------------------------------------------
    print("[STEP 5] Feature engineering (preparing data for the model)...")

    # StringIndexer turns text categories into numbers.
    # Sex only has two values (male, female), so it becomes 0 and 1.
    # handleInvalid="keep" makes sure any unseen value later is handled safely.
    sex_indexer = StringIndexer(inputCol="Sex", outputCol="Sex_indexed", handleInvalid="keep")

    # Embarked has three values (S, C, Q), so StringIndexer turns them into 0, 1, 2.
    embarked_indexer = StringIndexer(inputCol="Embarked", outputCol="Embarked_indexed", handleInvalid="keep")

    # VectorAssembler combines multiple columns into ONE column called "features".
    # Machine learning models in Spark require all inputs to be packed into a single vector.
    # We choose these columns because they are all numbers (or have been turned into numbers):
    #   Pclass       = passenger class (1st, 2nd, 3rd)
    #   Sex_indexed  = 0 or 1 for gender
    #   Age          = passenger age
    #   SibSp        = number of siblings/spouses aboard
    #   Parch        = number of parents/children aboard
    #   Fare         = ticket price
    #   Embarked_indexed = 0, 1, or 2 for port
    assembler = VectorAssembler(
        inputCols=["Pclass", "Sex_indexed", "Age", "SibSp", "Parch", "Fare", "Embarked_indexed"],
        outputCol="features",
        handleInvalid="skip"
    )

    # Pipeline chains the two indexers and the assembler into one object.
    # When we call fit() and transform(), all three steps run automatically in order.
    pipeline = Pipeline(stages=[sex_indexer, embarked_indexer, assembler])

    # fit() "learns" the mapping from text to numbers (e.g., male->0, female->1) using our data.
    # transform() actually applies those mappings and builds the "features" column.
    df_prepared = pipeline.fit(df_clean).transform(df_clean)

    print("         Feature engineering complete.")
    print("         Sample of prepared features (showing 'features' column only):")
    df_prepared.select("Survived", "features").show(5, truncate=False)
    print()


    # --------------------------------------------------------
    # STEP 6 — Split Data into Training and Testing Sets
    # --------------------------------------------------------
    print("[STEP 6] Splitting data into training and testing sets...")

    # randomSplit divides the DataFrame into two parts using percentages.
    # [0.8, 0.2] means 80% goes to training, 20% goes to testing.
    # seed=42 makes sure that every time we run the script, the SAME rows go to training and testing.
    # This is important so our results are reproducible.
    train_df, test_df = df_prepared.randomSplit([0.8, 0.2], seed=42)

    # Cache the datasets in memory so Spark does not re-read them every time we use them.
    train_df = train_df.cache()
    test_df = test_df.cache()

    # Count the rows in each split
    train_count = train_df.count()
    test_count = test_df.count()

    print(f"         Training set rows : {train_count} ({train_count / (train_count + test_count) * 100:.1f}%)")
    print(f"         Testing set rows  : {test_count} ({test_count / (train_count + test_count) * 100:.1f}%)")
    print()


    # --------------------------------------------------------
    # STEP 7 — Train the Decision Tree Model
    # --------------------------------------------------------
    print("[STEP 7] Training the Decision Tree model...")

    # DecisionTreeClassifier is the algorithm we are using.
    # labelCol="Survived" tells it which column contains the correct answer (0 or 1).
    # featuresCol="features" tells it which column contains the input data for prediction.
    # maxDepth=5 limits how deep the tree can grow. A deeper tree might memorize the training data instead of learning general patterns.
    dt = DecisionTreeClassifier(
        labelCol="Survived",
        featuresCol="features",
        maxDepth=5,
        seed=42
    )

    # fit() is the actual training step. Spark reads the training data and builds the tree.
    model = dt.fit(train_df)

    print("         Model training complete!")
    print()


    # --------------------------------------------------------
    # STEP 8 — Make Predictions on the Test Set
    # --------------------------------------------------------
    print("[STEP 8] Making predictions on the test set...")

    # transform() applies the trained model to the test data.
    # It adds new columns to the DataFrame, including:
    #   prediction  = the model's guess (0 or 1)
    #   probability = the confidence of the guess (e.g., [0.3, 0.7] means 70% confident it is 1)
    predictions = model.transform(test_df)

    # Select columns that are useful to look at, so we can compare actual vs. predicted.
    print("         Sample predictions (first 10 rows):")
    predictions.select(
        "Pclass",
        "Sex",
        "Age",
        "Fare",
        "Survived",      # the real answer
        "prediction",    # what the model guessed
        "probability"  # confidence score
    ).show(10, truncate=False)
    print()


    # --------------------------------------------------------
    # STEP 9 — Evaluate the Model
    # --------------------------------------------------------
    print("[STEP 9] Evaluating model performance...")

    # MulticlassClassificationEvaluator can compute several metrics.
    # We create one evaluator for each metric, changing the metricName parameter.
    # labelCol is the true answer. predictionCol is the model's guess.

    evaluator_accuracy = MulticlassClassificationEvaluator(
        labelCol="Survived", predictionCol="prediction", metricName="accuracy"
    )
    evaluator_precision = MulticlassClassificationEvaluator(
        labelCol="Survived", predictionCol="prediction", metricName="weightedPrecision"
    )
    evaluator_recall = MulticlassClassificationEvaluator(
        labelCol="Survived", predictionCol="prediction", metricName="weightedRecall"
    )
    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol="Survived", predictionCol="prediction", metricName="f1"
    )

    # evaluate() runs the calculation on the predictions DataFrame and returns a number between 0 and 1.
    accuracy = evaluator_accuracy.evaluate(predictions)
    precision = evaluator_precision.evaluate(predictions)
    recall = evaluator_recall.evaluate(predictions)
    f1_score = evaluator_f1.evaluate(predictions)

    # Print the results in a clean, formatted box.
    print()
    print("=" * 40)
    print("      MODEL EVALUATION RESULTS")
    print("=" * 40)
    print(f"   Accuracy  : {accuracy * 100:.2f}%")
    print(f"   Precision : {precision * 100:.2f}%")
    print(f"   Recall    : {recall * 100:.2f}%")
    print(f"   F1 Score  : {f1_score * 100:.2f}%")
    print("=" * 40)
    print()


    # --------------------------------------------------------
    # STEP 10 — Show the Decision Tree Structure
    # --------------------------------------------------------
    print("[STEP 10] Decision Tree structure (rules learned by the model):")
    print()

    # toDebugString prints the tree as text, showing the split conditions at each node.
    # For example: "If feature 1 <= 0.5 (meaning Sex is female) then go left, else go right."
    print(model.toDebugString)

    print()
    print("NOTE: Each line shows a condition (e.g., 'feature 1 <= 0.5').")
    print("      'feature 1' usually refers to the second column in your feature vector.")
    print("      'Predict: 0.0' means the tree predicts the passenger did NOT survive.")
    print("      'Predict: 1.0' means the tree predicts the passenger DID survive.")
    print()


    # --------------------------------------------------------
    # STEP 11 — Save Results to Output Files
    # --------------------------------------------------------
    print("[STEP 11] Saving results to the output/ folder...")

    # Record the current date and time for the summary file.
    run_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Coalesce predictions to a single CSV file (instead of multiple part files).
    # overwrite=True replaces any previous results.
    # header=True includes column names in the CSV.
    predictions.coalesce(1).write.mode("overwrite").csv("output/predictions", header=True)

    # Build a summary text string with all the important information.
    summary_text = f"""═══════════════════════════════════════════════════
           PROJECT RESULTS SUMMARY
═══════════════════════════════════════════════════
Date and Time Run    : {run_datetime}
Dataset Used         : Titanic Dataset ({total_rows} rows)
Algorithm            : Decision Tree Classifier
Max Depth            : 5
Train/Test Split     : 80% / 20%
Training Rows        : {train_count}
Testing Rows         : {test_count}
───────────────────────────────────────────────────
Accuracy             : {accuracy * 100:.2f}%
Precision            : {precision * 100:.2f}%
Recall               : {recall * 100:.2f}%
F1 Score             : {f1_score * 100:.2f}%
───────────────────────────────────────────────────
Note: Accuracy shows overall correctness.
      Precision shows how many predicted survivors actually survived.
      Recall shows how many actual survivors we correctly found.
      F1 Score balances Precision and Recall.
═══════════════════════════════════════════════════
"""

    # Write the summary to a text file in the output folder.
    # 'w' means write mode. encoding="utf-8" ensures special characters work correctly.
    with open("output/results_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    print("         Predictions saved to: output/predictions/")
    print("         Summary saved to   : output/results_summary.txt")
    print()


    # --------------------------------------------------------
    # STEP 12 — Stop Spark and Print Goodbye Message
    # --------------------------------------------------------
    print("[STEP 12] Stopping Apache Spark...")

    # stop() shuts down the Spark session and frees memory.
    spark.stop()

    print()
    print("=" * 60)
    print("   ALL DONE! Project completed successfully.")
    print("   Check the output/ folder for your results.")
    print("=" * 60 + "\n")


# ============================================================
# ERROR HANDLING — catch any unexpected error and explain it
# ============================================================
except Exception as e:
    # If ANY error happens anywhere in the try block above, Python jumps here.
    # We print the error message so the user knows what went wrong instead of seeing a silent crash.
    print()
    print("=" * 60)
    print("   ERROR: Something went wrong!")
    print("=" * 60)
    print(f"   Error message: {e}")
    print()
    print("Common fixes:")
    print("   1. Make sure Java JDK 17 is installed and JAVA_HOME is set.")
    print("   2. Make sure PySpark is installed: pip install pyspark")
    print("   3. Make sure titanic.csv exists inside the data/ folder.")
    print("   4. Run this script from inside the BigDataProject folder.")
    print("=" * 60)
    print()
    # Exit the program with a non-zero code so the operating system knows it failed.
    sys.exit(1)
