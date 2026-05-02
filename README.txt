═══════════════════════════════════════════════════════════════
     BIG DATA CLASSIFICATION PROJECT - README GUIDE
     Titanic Survival Prediction using Apache Spark
═══════════════════════════════════════════════════════════════

Welcome! This guide is written for complete beginners.


═══════════════════════════════════════════════════════════════
SECTION 1: PROJECT OVERVIEW
═══════════════════════════════════════════════════════════════

What is this project about?
--------------------------
This project uses the famous Titanic dataset to PREDICT whether a passenger
survived or not. We feed passenger information into a computer program that
learns patterns from historical data, then guesses survival for new passengers.

What problem are we solving?
--------------------------
We are solving a CLASSIFICATION problem: putting each passenger into one of
two groups - "Survived" or "Did Not Survive".

What is Big Data and why does it matter?
--------------------------
"Big Data" means datasets too large for a regular computer to handle easily.
While our Titanic dataset has 891 rows, real datasets at Netflix or Amazon
have BILLIONS of rows. Big Data tools let us analyze massive information quickly.

What is Apache Spark and why do we use it?
--------------------------
Apache Spark is a powerful engine that processes huge datasets across many
computers at once. Think of 100 people reading a giant book together, each
reading a different chapter, instead of one person reading the whole book alone.
Even on a single laptop, Spark is fast and has built-in machine learning tools.

What is a Decision Tree?
--------------------------
A Decision Tree is like the game "20 Questions." The computer asks yes/no
questions about a passenger:

    "Is the passenger female?" -> YES -> "Is she in 1st class?" -> YES -> SURVIVED
                                            |
                                            -> NO -> "Is she a child?" ...
    |
    -> NO (male) -> "Did he pay a high fare?" -> ...

At the end of each path, the tree makes a prediction.


═══════════════════════════════════════════════════════════════
SECTION 2: TEAM INFORMATION
═══════════════════════════════════════════════════════════════

Team Member 1        : Omar Ammar
Team Member 2        : Kareem Reda
Team Member 3        : Mohammed El Saed

University           : Mansoura University
Faculty              : Faculty of Computers and Information
Course               : Selected Topics in Information Systems
Academic Year        : 2025-2026


═══════════════════════════════════════════════════════════════
SECTION 3: TOOLS & TECHNOLOGIES USED
═══════════════════════════════════════════════════════════════

Apache Spark
- An open-source engine for processing large-scale data.
- We chose Spark because it handles data processing and machine learning
  in one framework, and is faster than older tools like Hadoop.

PySpark
- The Python interface for Apache Spark.
- Since our team knows Python, PySpark lets us use Spark's power without
  learning Java or Scala.

Python
- One of the most popular languages in data science.
- Easy to read, with thousands of free libraries.

Anaconda
- A "Python distribution" that installs Python plus 1,500+ useful packages.
- Gives us the Anaconda Prompt where all Python tools are ready to use.

Decision Tree Algorithm
- Easy to understand and great for beginners.
- Unlike a "black box" neural network, you can SEE the questions the tree asks.
- Perfect for explaining how the model works in a university presentation.

The Titanic Dataset
- Contains 891 real passenger records from the Titanic ship.
- Each record has 12 columns. Because the data is small, clean, and meaningful,
  it is the ideal starting point for learning classification.


═══════════════════════════════════════════════════════════════
SECTION 4: PROJECT FOLDER STRUCTURE
═══════════════════════════════════════════════════════════════

    BigDataProject/
    |-- data/
    |   |-- titanic.csv          <-- The dataset (download this)
    |
    |-- output/
    |   |-- predictions/         <-- CSV with model predictions
    |   |-- results_summary.txt  <-- Text summary of metrics
    |
    |-- main.py                  <-- The main PySpark script
    |-- README.txt               <-- This guide
    |-- team_info.txt            <-- Team and project details
    |-- SETUP_GUIDE.txt          <-- Installation instructions


═══════════════════════════════════════════════════════════════
SECTION 5: HOW TO INSTALL AND RUN
═══════════════════════════════════════════════════════════════

Step 1 - Install Java JDK 17
--------------------------
Open: https://adoptium.net/temurin/releases/?version=17
Download the Windows x64 .msi installer for JDK 17.
Run it and click Next until it finishes.

Set JAVA_HOME:
- Search "environment variables" in Windows Start.
- Click "Environment Variables".
- Under "User variables", click "New".
- Name: JAVA_HOME
- Value: C:\Program Files\Eclipse Adoptium\jdk-17.xxx
- Click OK three times.

Add to PATH:
- Find "Path" under User variables -> Edit -> New.
- Add: C:\Program Files\Eclipse Adoptium\jdk-17.xxx\bin
- Click OK.

Verify: Open Command Prompt, type: java -version
Expected: openjdk version "17.0.x"

Step 2 - Install Anaconda
--------------------------
Open: https://www.anaconda.com/download
Download the 64-bit Windows installer.
Run it, choose "Just Me", leave defaults checked.

Verify: Open Anaconda Prompt, type: python --version

Step 3 - Install PySpark
--------------------------
Inside Anaconda Prompt, type: pip install pyspark

Verify: python -c "import pyspark; print(pyspark.__version__)"

Step 4 - Download the Dataset
--------------------------
Open browser to:
    https://raw.githubusercontent.com/datasciencedoc/data/master/titanic.csv
Right-click -> Save As -> name it "titanic.csv"
Move it into the data/ subfolder of your project.

Step 5 - Run the Project
--------------------------
Open Anaconda Prompt.
Navigate to project folder:
    cd "d:\College Tasks\Big Data Project"
Run:
    python main.py

Step 6 - Find Your Results
--------------------------
Open the output/ folder.
You will see predictions/ and results_summary.txt.


═══════════════════════════════════════════════════════════════
SECTION 6: UNDERSTANDING THE DATASET
═══════════════════════════════════════════════════════════════

The Titanic dataset columns:

PassengerId    : A number for each passenger (not useful for prediction)
Survived       : 0 = died, 1 = survived. THIS IS OUR TARGET (what we predict).
Pclass         : Ticket class. 1 = 1st class, 2 = 2nd class, 3 = 3rd class.
Name           : Full name (not useful for math-based prediction)
Sex            : male or female. Very important for survival prediction.
Age            : Passenger age. Children had higher survival rates.
SibSp          : Number of Siblings or Spouses aboard.
Parch          : Number of Parents or Children aboard.
Ticket         : Ticket number (mostly unique, not useful)
Fare           : How much the passenger paid for their ticket.
Cabin          : Cabin number (many missing values, not very useful)
Embarked       : Port where the passenger boarded. C = Cherbourg, Q = Queenstown, S = Southampton.

What is a "feature" and what is a "label"?
--------------------------
- A LABEL is the answer you are trying to predict. Here, the label is "Survived".
- A FEATURE is any piece of information you use to make the prediction.
  Our features are: Pclass, Sex, Age, SibSp, Parch, Fare, and Embarked.

Think of it like a math test: the FEATURES are the questions, and the LABEL
is the correct answer in the answer key.


═══════════════════════════════════════════════════════════════
SECTION 7: HOW THE DECISION TREE WORKS
═══════════════════════════════════════════════════════════════

Simple Example
--------------------------
Here is how the Decision Tree might think about a passenger:

    START
      |
      +-- Is Sex = female? ---> YES
      |                         |
      |                         +-- Is Pclass = 1 or 2? ---> YES  ---> SURVIVED
      |                         |                           |
      |                         |                           +-- NO ---> DIED
      |                         |
      |                         +-- Is Age < 10? ---> YES ---> SURVIVED
      |                                                     |
      |                                                     +-- NO ---> DIED
      |
      +-- Is Sex = male? ---> YES
                                |
                                +-- Is Age < 5? ---> YES ---> SURVIVED
                                |                           |
                                |                           +-- NO ---> DIED
                                |
                                +-- Is Fare > 100? ---> YES ---> SURVIVED
                                                            |
                                                            +-- NO ---> DIED

Training vs. Testing
--------------------------
- TRAINING: We show the tree 80% of the passengers (with their real answers)
  and let it figure out the best questions to ask.
- TESTING: We hide the real answers for the remaining 20% and ask the tree
  to guess. We compare its guesses to the real answers.

80/20 Split
--------------------------
80% of the data is used for learning (training), and 20% is kept secret for
testing. This is like studying 80% of a textbook, then taking an exam on
the other 20% to see if you really learned the material.


═══════════════════════════════════════════════════════════════
SECTION 8: UNDERSTANDING THE RESULTS
═══════════════════════════════════════════════════════════════

After running main.py, you will see four numbers. Here is what they mean:

Accuracy
--------------------------
"Out of every 100 passengers we tested, how many did we guess correctly?"

Example: If accuracy is 82%, the model correctly predicted survival for
82 out of every 100 passengers.
Analogy: A student who gets 82 out of 100 questions right on a quiz.

Precision
--------------------------
"Of all passengers the model PREDICTED would survive, how many actually did?"

Analogy: You tell your friend "I think these 10 people will survive."
If 8 actually survived, your precision is 80%.
High precision means the model is careful about predicting survival.

Recall
--------------------------
"Of all passengers who ACTUALLY survived, how many did the model find?"

Analogy: 20 people actually survived. The model said "I found 16 survivors."
Recall is 16/20 = 80%.
High recall means the model rarely misses a real survivor.

F1 Score
--------------------------
F1 Score balances Precision and Recall. If you only cared about one, you
could cheat (e.g., predict everyone survives to get perfect recall).
F1 Score forces both to be good.

Analogy: F1 Score is like a GPA that averages your grades in two subjects.
You cannot get a high GPA by being perfect in one and failing the other.

What is a "good" accuracy?
--------------------------
For the Titanic dataset, a good Decision Tree gets between 78% and 84%.
- 78-84% = good, working model
- 50%   = no better than a coin flip, something is wrong
- 95%+  = tree might be memorizing instead of learning


═══════════════════════════════════════════════════════════════
SECTION 9: WHY APACHE SPARK FOR THIS PROJECT?
═══════════════════════════════════════════════════════════════

Traditional Python (pandas) vs. PySpark
--------------------------
Pandas is a Python library for working with tables. It is easy to use, but
runs on ONE computer only. If your table has 100 million rows, pandas might
freeze your laptop or run out of memory.

PySpark does the SAME work, but can split the job across many computers.
On a single laptop, Spark is still faster than pandas for very large files
because it processes data in small chunks instead of loading everything at once.

Why would Spark matter for 100 million rows?
--------------------------
Imagine reading 100 million rows in Excel. Excel would crash.
With Spark, those rows can be split across 100 computers. Each computer reads
1 million rows, and results are combined at the end.

This is how Netflix recommends movies to 230 million users, or how Uber
processes millions of ride requests per day.

Real companies using Spark
--------------------------
- Netflix: Analyzes what 230 million users watch and recommends shows.
- Uber: Calculates prices and finds the nearest driver in real time.
- Amazon: Predicts what products customers will buy next.


═══════════════════════════════════════════════════════════════
SECTION 10: COMMON ERRORS & SOLUTIONS
═══════════════════════════════════════════════════════════════

ERROR 1: "'java' is not recognized..."
CAUSE: Java not installed, or JAVA_HOME/PATH wrong.
FIX: Reinstall Java JDK 17, set JAVA_HOME and Path, RESTART computer.

ERROR 2: "No module named 'pyspark'"
CAUSE: PySpark not installed, or wrong terminal.
FIX: Open Anaconda Prompt (not regular CMD), run: pip install pyspark

ERROR 3: "Java heap space" or OutOfMemoryError
CAUSE: Java ran out of memory.
FIX: Close other programs. If it keeps happening, set environment variable:
     PYSPARK_SUBMIT_ARGS="--driver-memory 2g pyspark-shell"

ERROR 4: "FileNotFoundError: titanic.csv"
CAUSE: Dataset missing or in wrong folder.
FIX: Make sure titanic.csv is inside the data/ folder.

ERROR 5: "Failed to find Spark jars directory"
CAUSE: PySpark installation broken.
FIX: Run: pip uninstall pyspark  then  pip install pyspark

ERROR 6: "Python was not found"
CAUSE: You are in regular Command Prompt instead of Anaconda Prompt.
FIX: Close CMD. Open Anaconda Prompt from the Start menu.

ERROR 7: Predictions CSV has many part files
CAUSE: This is normal! Spark saves output in multiple parts for speed.
FIX: In main.py we use .coalesce(1) to force ONE file. If you still see parts,
     look for part-00000-xxxx.csv inside output/predictions/.

ERROR 8: Accuracy very low (< 60%) or very high (> 95%)
CAUSE: Data is wrong (low), or tree is overfitting (high).
FIX: Make sure dataset was not modified. Check you did not accidentally
     include "Survived" inside the feature vector.


═══════════════════════════════════════════════════════════════
SECTION 11: REFERENCES
═══════════════════════════════════════════════════════════════

- Apache Spark Official Documentation
  https://spark.apache.org/docs/latest/

- PySpark Documentation (Python API)
  https://spark.apache.org/docs/latest/api/python/

- Titanic Dataset Source
  https://raw.githubusercontent.com/datasciencedoc/data/master/titanic.csv

- Mansoura University
  Faculty of Computers and Information
  Course: Selected Topics in Information Systems
  Academic Year: 2025-2026

═══════════════════════════════════════════════════════════════
              GOOD LUCK WITH YOUR PROJECT!
═══════════════════════════════════════════════════════════════
