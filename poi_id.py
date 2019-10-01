#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

import numpy as np
np.random.seed(42)

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# The feature selection is detailed in "Data Exploration and Feature Selection" file
features_list = ['poi', 'salary', 'bonus', 'total_stock_value',
                 'expenses', 'exercised_stock_options', 'other',
                 'restricted_stock', 'shared_receipt_with_poi',
                 'fraction_to_poi', 'fraction_from_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
# Removing the keys incorrectly inserted in the dataset:
del data_dict["TOTAL"]
del data_dict["THE TRAVEL AGENCY IN THE PARK"]

### Task 3: Create new feature(s)
# fraction of messages TO poi
for key in data_dict:
    person = data_dict[key]
    try:
        person["fraction_to_poi"] = int(person["from_this_person_to_poi"])/int(person["from_messages"])
    except:
        person["fraction_to_poi"] = 0
# fraction of messages FROM poi
for key in data_dict:
    person = data_dict[key]
    try:
        person["fraction_from_poi"] = int(person["from_poi_to_this_person"])/int(person["to_messages"])
    except:
        person["fraction_from_poi"] = 0
# When fields are "NaN" we are considering the fields to have value zero.
# This may not be the optimal strategy.

### Store to my_dataset for easy export below.
my_dataset = data_dict



### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Scaling features to zero mean and unit variance
#from sklearn.preprocessing import scale
#features = scale(features)

### Task 4: Try a variety of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# SELECT CLASSIFIER: "randomforest" OR "svm"
select_classifier = "randomforest" # either "randomforest" or "svm"

if select_classifier == "randomforest":
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced')
if select_classifier == "svm":
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    steps = [("rescale", StandardScaler()), ("classifier", SVC(class_weight="balanced"))]
    clf = Pipeline(steps)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

from sklearn.model_selection import GridSearchCV
if select_classifier == "randomforest":
    param = {"min_samples_split": np.arange(2,21)}
if select_classifier == "svm":
    C_values = [0.001, 0.01, 0.1, 1, 100, 1000, 10000]
    k_values = ["linear", "poly", "rbf"]
    param = {"classifier__C": C_values, "classifier__kernel": k_values}

clf = GridSearchCV(clf, param, cv=5, scoring=["f1","precision","recall"],
                    refit="f1")

clf.fit(features, labels)

# Showing the "best estimator"
print(clf.best_estimator_)
# Showing the cross-validated scores for the best estimator
print("CV-Precision:", clf.cv_results_["mean_test_precision"][clf.best_index_])
print("CV-Recall:", clf.cv_results_["mean_test_recall"][clf.best_index_])
print("CV-F1:", clf.cv_results_["mean_test_f1"][clf.best_index_])
# Setting the best_estimator for export
clf = clf.best_estimator_

# Example starting point. Try investigating other evaluation techniques!
#from sklearn.cross_validation import train_test_split
#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)