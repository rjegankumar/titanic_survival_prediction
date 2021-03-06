import pandas as pd
import numpy as np
import re
import operator
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingClassifier

titanic = pd.read_csv('train.csv')

titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
titanic['Embarked'] = titanic['Embarked'].fillna('S')
titanic.loc[titanic['Embarked'] == 'S','Embarked'] = 0
titanic.loc[titanic['Embarked'] == 'C','Embarked'] = 1
titanic.loc[titanic['Embarked'] == 'Q','Embarked'] = 2

titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]
titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))

def get_title(name):
	title_search = re.search(' ([A-Za-z]+)\.', name)
	if title_search:
		return title_search.group(1)
	return ""

titles = titanic["Name"].apply(get_title)
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7,\
 "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
for k,v in title_mapping.items():
    titles[titles == k] = v
titanic["Title"] = titles

family_id_mapping = {}
def get_family_id(row):
	last_name = row["Name"].split(",")[0]
	family_id = "{0}{1}".format(last_name, row["FamilySize"])
	if family_id not in family_id_mapping:
		if len(family_id_mapping) == 0:
			current_id = 1
		else:
			current_id = (max(family_id_mapping.items(), key=operator.itemgetter(1))[1] + 1)
		family_id_mapping[family_id] = current_id
	return family_id_mapping[family_id]

family_ids = titanic.apply(get_family_id, axis=1)
family_ids[titanic["FamilySize"] < 3] = -1
titanic["FamilyId"] = family_ids

algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), ["Pclass", "Sex", "Age", "Fare", "Embarked",\
     "FamilySize", "Title", "FamilyId", "NameLength"]],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]
]
kf = model_selection.KFold(n_splits=3, random_state=1)

predictions = []
for train, test in kf.split(titanic):
    train_target = titanic["Survived"].iloc[train]
    full_test_predictions = []
    for alg, predictors in algorithms:
    	alg.fit(titanic[predictors].iloc[train,:], train_target)
    	test_predictions = alg.predict_proba(titanic[predictors].iloc[test,:].astype(float))[:,1]
    	full_test_predictions.append(test_predictions)
    test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2
    test_predictions[test_predictions <= .5] = 0
    test_predictions[test_predictions > .5] = 1
    predictions.append(test_predictions)

predictions = np.concatenate(predictions, axis=0)

accuracy = sum(predictions == titanic["Survived"]) / len(predictions)
print(accuracy)