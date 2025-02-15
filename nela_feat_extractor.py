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

# Extract train data
def extract_train_pairs(df):
    col = df.columns
    txt_lbl_pairs = []
    for l in col[1:]:
        txt_lbl_pairs = txt_lbl_pairs + [(t,l) for t in df[l].values]

    # Convert to a DataFrame and return
    return pd.DataFrame(txt_lbl_pairs, columns=["text", "label"])

nela = NELAFeatureExtractor()

train_df = pd.read_csv(f'train.csv').rename(columns={'accounts/yi-01-ai/models/yi-large': 'Yi-Large'})
train_df.fillna("This is an empty string.", inplace=True)

val_df = pd.read_csv(f'val.csv').rename(columns={'Text': 'text', 'LABEL_B': 'label'})
val_df.fillna("This is an empty string.", inplace=True)

test_df = pd.read_csv(f'updated_test_data.csv').rename(columns={'Text': 'Human_story'})
test_df.fillna("This is an empty string.", inplace=True)

df_train = extract_train_pairs(train_df)
df_train_features = pd.DataFrame([nela.extract_all(text)[0] for text in tqdm(df_train["text"])])
df_train = pd.concat([df_train, df_train_features], axis=1)
df_train.to_pickle("train_nela.pickle")



df_test = extract_train_pairs(test_df)
df_test_features = pd.DataFrame([nela.extract_all(text)[0] for text in tqdm(df_test["text"])])
df_test = pd.concat([df_test, df_test_features], axis=1)
df_test.to_pickle("test_nela.pickle")


df_val = val_df[['text','label']].reset_index(drop=True)
df_val_features = pd.DataFrame([nela.extract_all(text)[0] for text in tqdm(df_val["text"])])
df_val = pd.concat([df_val.reset_index(drop=True), df_val_features], axis=1)
df_val.to_pickle("val_nela_non_empty.pickle")

