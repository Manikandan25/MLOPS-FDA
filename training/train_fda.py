import pickle
import os
import numpy as np
import pandas as pd
import json
import subprocess

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, balanced_accuracy_score, confusion_matrix,average_precision_score, precision_recall_fscore_support,precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from joblib import dump
from typing import Tuple, List

import matplotlib.pyplot as plt
import seaborn as sns

from azureml.core import Workspace, Datastore, Dataset
from azureml.core.run import Run
from azureml.core.model import Model

run = Run.get_context()
exp = run.experiment
ws = run.experiment.workspace

print("Loading training data...")
# https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html
datastore = ws.get_default_datastore()
datastore_paths = [(datastore, 'fda/seafood_imports.csv')]
traindata = Dataset.Tabular.from_delimited_files(path=datastore_paths)
fda_data = traindata.to_pandas_dataframe()
print("Columns:", fda_data.columns) 
print("Diabetes data set dimensions : {}".format(fda_data.shape))

X = fda_data.drop('IsSafe', axis = 1)
X = X.apply(LabelEncoder().fit_transform)
fda_data['IsSafe'] = fda_data['IsSafe'].fillna('Yes')
fda_data['IsSafe'][fda_data['IsSafe'].isin(['No'])]=0
fda_data['IsSafe'][fda_data['IsSafe'].isin(['Yes'])]=1
y = fda_data.pop('IsSafe')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
data = {"train": {"X": X_train, "y": y_train}, "test": {"X": X_test, "y": y_test}}

print("Training the model...")
# Randomly pic alpha
n_estimator = np.arange(100, 500, 100)
# alpha = alphas[np.random.choice(alphas.shape[0], 1, replace=False)][0]
print("estimator:", n_estimator)
run.log("estimator", n_estimator)
model_rf = RandomForestClassifier(n_estimator=n_estimator)
model_rf.fit(fda_data["train"]["X"], fda_data["train"]["y"])
# run.log_list("coefficients", model_rf.coef_)

print("Evaluate the model...")
preds = model_rf.predict(data["test"]["X"])
f1_score = f1_score(preds, data["test"]["y"])
print("F1 Score:", f1_score)
run.log("F1 Score", f1_score)

# Save model as part of the run history
print("Exporting the model as pickle file...")
outputs_folder = './model'
os.makedirs(outputs_folder, exist_ok=True)

model_filename = "sklearn_fda_model.pkl"
model_path = os.path.join(outputs_folder, model_filename)
dump(model_rf, model_path)

# upload the model file explicitly into artifacts
print("Uploading the model into run artifacts...")
run.upload_file(name="./outputs/models/" + model_filename, path_or_stream=model_path)
print("Uploaded the model {} to experiment {}".format(model_filename, run.experiment.name))
dirpath = os.getcwd()
print(dirpath)
print("Following files are uploaded ")
print(run.get_file_names())

run.complete()