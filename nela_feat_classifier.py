import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_curve, auc
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import classification_report, accuracy_score
from nela_features.nela_features import NELAFeatureExtractor
from xgboost import XGBClassifier


import string
import json

llms = ['Human_story', 'gemma-2-9b', 'mistral-7B', 'qwen-2-72B', 'llama-8B', 'Yi-Large', 'GPT_4-o']
label_mapping = {llm: idx for idx, llm in enumerate(llms)}
reverse_mapping = {idx: llm for llm, idx in label_mapping.items()}

df_train_nela = pd.read_pickle("train_nela.pickle")

df_val_nela = pd.read_pickle("val_nela.pickle")

df_test_nela = pd.read_pickle("test_nela.pickle")

# Prepare training and testing data

X_train = df_train_nela[df_train_nela.columns[2:].values] # First two columns are non-features
y_train = df_train_nela['label'].map(label_mapping)

X_valid = df_val_nela[df_val_nela.columns[2:].values]
y_valid = df_val_nela['label'].map(label_mapping)

X_test = df_test_nela[df_test_nela.columns[2:].values]

def save_answer_csv(pred_list, suffix):
    # pred_map = {}
    # for i, l in enumerate(llms[1:]):
    #     pred_map[i] = l

    pd_test = pd.read_csv('updated_test_data.csv')
    data_tuple = []
    for i, pair in enumerate(zip(pred_list, pd_test.iterrows())):
        pred = pair[0]
        text = pair[1][1]['Text']
        t_dict = {
            "index": int(i),
            "Text": text,
            "Label_A": 0 if pred=='Human_story' else 1,
            "Label_B": pred
        }
        data_tuple.append(t_dict)
    with open(f'output_test_{suffix}.json', 'w') as json_file:
        json.dump(data_tuple, json_file, indent=4)

# Define base models
# clf = RandomForestClassifier()
# clf = SVC()
clf = xgb.XGBClassifier()
clf.fit(X_train, y_train)

y_valid_pred = clf.predict(X_valid)


print("Accuracy Binary:", accuracy_score(np.array([0 if e ==0 else 1 for e in y_valid]), np.array([0 if e ==0 else 1 for e in y_valid_pred])))
print(classification_report(np.array([0 if e ==0 else 1 for e in y_valid]), np.array([0 if e ==0 else 1 for e in y_valid_pred])))

print("Accuracy Muticlass:", accuracy_score(y_valid, y_valid_pred))
print(classification_report(y_valid, y_valid_pred))

y_test_pred = clf.predict(X_test)
decoded_labels = [reverse_mapping[num] for num in y_test_pred]
save_answer_csv(decoded_labels,'nela')
