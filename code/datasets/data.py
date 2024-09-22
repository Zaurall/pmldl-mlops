import os
import zenml
import pandas as pd
import numpy as np
import dvc.api
from zenml.client import Client
from sklearn.model_selection import train_test_split

BASE_PATH = os.path.expandvars("$PROJECTPATH")


def extract_data(version=None) -> tuple[pd.DataFrame, str]:
    data_path = "data/raw/data.csv"
    data_store = "myremote"
    if version == None:
        version = "v1"

    path = dvc.api.get_url(rev=version, path=data_path, remote=data_store, repo=BASE_PATH)

    df = pd.read_csv(path)

    return df, version

def transform_data(df):
    # Renaming column
    df = df.rename(columns={'n_hos_beds': 'n_hot_beds'})
    
    # Impute missing values
    df['n_hot_beds'] = df['n_hot_beds'].fillna(df['n_hot_beds'].mean())
    
    # Outliers processing
    lower_threshold = np.percentile(df["n_hot_rooms"], [1])[0]
    higher_threshold = np.percentile(df["n_hot_rooms"], [99])[0]
    df.loc[df["n_hot_rooms"] < 0.3 * lower_threshold, "n_hot_rooms"] = 0.3 * lower_threshold
    df.loc[df["n_hot_rooms"] > 3 * higher_threshold, "n_hot_rooms"] = 3 * higher_threshold

    X = df.drop(['price'],axis = 1)
    y = df['price'].to_frame()

    return X, y

def split_save_data(X: pd.DataFrame, y: pd.DataFrame, version: str):
    data_path = f"{BASE_PATH}/data/processed/"
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    train_df.to_csv(f"{data_path}train.csv")
    test_df.to_csv(f"{data_path}test.csv")
    
    zenml.save_artifact(data=train_df, name="train_df", tags=[version])
    zenml.save_artifact(data=test_df, name="test_df", tags=[version])
    
def extract_preprocessed_data(name, version, size=1):
    client = Client()
    l = client.list_artifact_versions(name = name, tag = version, sort_by="version").items
    latest_artifact = sorted(l, key=lambda x: x.created)[-1]
    df = latest_artifact.load()
    df = df.sample(frac = size, random_state = 88)

    print("size of df is ", df.shape)
    print("df columns: ", df.columns)

    return df