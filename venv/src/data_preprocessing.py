import pandas as pd
import numpy as np
from typing import Tuple
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "data.csv"
def load_data() -> pd.DataFrame:
    '''
    Download dataset

    :return: pd.DataFrame:
             dataframe with data
    '''

    df=pd.read_csv(DATA_PATH)
    return df





def prepare_data(test_size = 0.2, random_state = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray,np.ndarray]:
    '''
    Prepare data for training

    :param test_size (float) : part of the data for the training set
    :param random_state: grain of randomness
    :return:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            X_train, X_test, y_train, y_test – scaled feature matrices
            and label vectors
    '''
    df = load_data()
    # binary label
    df.diagnosis = [1 if each == "M" else 0 for each in df.diagnosis]

    # drop data and x/y data split
    X = df.drop(columns=["id","diagnosis","Unnamed: 32"])
    Y = df["diagnosis"]

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=test_size, random_state=random_state, stratify = Y)

    # scaling

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # save scaler

    models_dir = Path(__file__).resolve().parents[1] / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(scaler, models_dir / "scaler.pkl")

    return X_train, X_test, Y_train.values, Y_test.values


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te = prepare_data()
    print("train:", X_tr.shape,  "test:", X_te.shape,
          " malignant in train:", y_tr.sum())







