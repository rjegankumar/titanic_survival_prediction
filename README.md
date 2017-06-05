# Titanic Survival Prediction - Kaggle Competition

My first few attempts at [Kaggle's Titanic Survival Prediction](https://www.kaggle.com/c/titanic).

## Files

| File name | Description |
| :--- | :--- |
| [environment.yml](environment.yml) | Conda environment required to run this code |
| [feature_engineering.py](feature_engineering.py) | Python script to cleanup existing features, create new features and predict using a RandomForestClassifier |
| [gradient_boost_ensembling.py](gradient_boost_ensembling.py) | Python script to make predictions using an ensemble of GradientBoostingClassifier and LogisticRegression |
| [linear_regression.py](linear_regression.py) | Python script to make predictions using LinearRegression |
| [logistic_regression.py](logistic_regression.py) | Python script to make predictions using LogisticRegression |
| [NextSteps.txt](NextSteps.txt) | Text file with ideas on how to improve the features and algorithm |
| [random_forest.py](random_forest.py) | Python script to make predictions using RandomForestClassifier |
| [submission_1.csv](submission_1.csv) | First Kaggle submission |
| [submission_2.csv](submission_2.csv) | Second Kaggle submission |
| [test.csv](test.csv) | Test data |
| [titanic_survival_prediction_submission_1.py](titanic_survival_prediction_submission_1.py) | Python script for submission 1 that used LogisticRegression to make predictions |
| [titanic_survival_prediction_submission_2.py](titanic_survival_prediction_submission_2.py) | Python script for submission 2 that used feature engineering and an ensemble of GradientBoostingClassifier and LogisticRegression to make predictions |
| [train.csv](train.csv) | Training data |

## Setup

- You must have [Anaconda](https://www.continuum.io/downloads) installed to run this code.
- Create a conda environment using [environment.yml](environment.yml) YAML file. More help on this can be found [here](https://conda.io/docs/using/envs.html#use-environment-from-file).

## License

The contents of this repository are covered under the [MIT License](LICENSE).
