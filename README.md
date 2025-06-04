# Heart-Failure-Prediction-with-Machine-Learning-Models
Predicting heart failure using various machine learning classification algorithms, including Gaussian Naive Bayes, Random Forest, KNN, Gradient Boosting, Voting, and Stacking, with data exploration and visualization.


## Dataset

The primary dataset used in this project is the "Heart Failure Prediction" dataset from Kaggle. The code uses the `opendatasets` library to download this dataset directly.

The code also attempts to load a dataset named "nanofluid\_heat\_transfer\_dataset.csv" from a specific Google Drive path. **Please note:** If you do not have this file at the specified path in your Google Drive mounted in Colab, the initial data loading will fail. However, the core analysis and modeling sections focus on the heart failure dataset downloaded from Kaggle.

## Workflow

The Jupyter Notebook follows these steps:

1.  **Import Libraries:** Essential libraries like `pandas`, `numpy`, `sklearn`, `matplotlib`, and `seaborn` are imported.
2.  **Load Data:**
    *   Attempts to load a dataset from Google Drive (may require Google Colab and mounted Drive).
    *   Downloads the Heart Failure Prediction dataset from Kaggle using `opendatasets`.
    *   Loads the downloaded "heart.csv" file into a pandas DataFrame.
3.  **Data Exploration and Preprocessing:**
    *   Displays the first few rows of the DataFrame (`df.head()`).
    *   Provides a concise summary of the DataFrame, including data types and non-null values (`df.info()`).
    *   Generates descriptive statistics of the numerical columns (`df.describe()`).
    *   Checks for missing values (`df.isnull().shape`, `df.isnull().sum()`). The dataset appears to have no missing values based on `df.isnull().sum()`.
    *   Identifies the number of unique values in each column (`df.nunique()`) which can hint at categorical columns.
    *   Performs one-hot encoding on the categorical variables using `pd.get_dummies()` to convert them into a numerical format suitable for machine learning models.
4.  **Data Splitting:**
    *   Separates the features (input variables, `X`) from the target variable (the column indicating 'HeartDisease', `y`).
    *   Splits the data into training and testing sets using `train_test_split` with a test size of 20% and a fixed `random_state` for reproducibility.
5.  **Model Training and Evaluation:**
    *   **Gaussian Naive Bayes:** Trains a `GaussianNB` model and evaluates its performance using a `classification_report`.
    *   **Individual Model Evaluation Metrics:** Calculates and prints individual evaluation metrics (Confusion Matrix, Accuracy, Precision, Recall, F1-score) using a variable `y_pred`. **Note:** The variable `y_pred` is first assigned the predictions from a Voting Classifier later in the notebook. This section might appear out of sequence for evaluating GaussianNB specifically.
    *   **Random Forest:** Trains a `RandomForestClassifier` and evaluates its performance using a `classification_report`.
    *   **K-Nearest Neighbors (KNN):** Trains a `KNeighborsClassifier` and evaluates its performance using a `classification_report`.
    *   **Gradient Boosting:** Trains a `GradientBoostingClassifier` and evaluates its performance using a `classification_report`.
6.  **Ensemble Modeling (Voting Classifier):**
    *   Trains three different `VotingClassifier` models, combining various base estimators (`RandomForestClassifier`, `GradientBoostingClassifier`, `KNeighborsClassifier`) using a 'soft' voting strategy (based on predicted probabilities).
    *   Evaluates the performance of each voting classifier using a `classification_report`. The `y_pred` variable is updated with the predictions from these voting classifiers.
7.  **Visualization of Results:**
    *   Visualizes a single confusion matrix using `seaborn.heatmap`.
    *   Visualizes the confusion matrices for Gaussian Naive Bayes, Random Forest, and KNN in a single figure using subplots.
    *   Visualizes the distribution of the target variable ('HeartDisease') using a count plot.
    *   Explores the relationship between 'Cholesterol' and 'HeartDisease' using a box plot.
    *   Visualizes the distribution of 'Age' by 'HeartDisease' using a histogram.
    *   Visualizes the relationship between 'Sex' and 'HeartDisease' using a count plot.
    *   Visualizes the confusion matrices for Random Forest, Gradient Boosting, Gaussian Naive Bayes, and KNN in a single figure using subplots for comparison.
8.  **Feature Importance (Random Forest):**
    *   Trains a `RandomForestClassifier` on the entire dataset (`X` and `y`) to determine feature importances.
    *   Visualizes the feature importances using a horizontal bar plot.
9.  **Ensemble Modeling (Stacking Classifier):**
    *   Trains a `StackingClassifier` using Random Forest and Gradient Boosting as base estimators and Logistic Regression as the final estimator.
    *   Evaluates the performance of the stacking classifier using a `classification_report`.

## Key Libraries Used

-   **pandas:** Data manipulation and analysis.
-   **numpy:** Numerical operations.
-   **scikit-learn:** Machine learning models, data splitting, and evaluation metrics.
-   **matplotlib.pyplot & seaborn:** Data visualization.
-   **spacy:** Natural language processing (used for loading a model, but not extensively in the core analysis).
-   **opendatasets:** Downloading datasets from Kaggle.

## Running the Code

Open the Jupyter Notebook file in your Jupyter environment and run the cells sequentially. Ensure you have the necessary libraries installed and, if using Google Colab, that your Google Drive is mounted if you intend to load the nanofluid dataset.

## Notes

-   The initial loading of the nanofluid dataset might fail if the file is not present at the specified Google Drive path. The subsequent analysis focuses on the heart failure dataset.
-   The evaluation of individual metrics after the Gaussian Naive Bayes training uses the `y_pred` variable, which is populated later by the Voting Classifiers. This means the confusion matrix, accuracy, precision, recall, and F1-score printed in that section will correspond to the performance of the last Voting Classifier evaluated, not the Gaussian Naive Bayes model trained just before it.
-   The code trains and evaluates several models independently and then explores ensemble methods (Voting and Stacking) to see if combining models improves performance.
