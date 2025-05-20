import pandas as pd
from catboost import CatBoostClassifier

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import optuna

# from giskard import Dataset, Model, scan, testing
import pickle

from minio import Minio
from minio.error import S3Error
import warnings

import psycopg2

import mlflow 
# import mlflow.catboost
import mlflow.sklearn
import mlflow.data
from mlflow.models import infer_signature

from functools import reduce
from datetime import datetime

# IMPORT ENV VARIABLES -------------------------------------------------------
from dotenv import load_dotenv
import os

# def main():
load_dotenv()

DB_HOST = os.getenv('DB_HOST')
POSTGRESQL_PORT = os.getenv('POSTGRESQL_PORT')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_NAME = os.getenv('DB_NAME')

ACCESS_KEY=os.getenv('MINIO_ROOT_USER')
SECRET_KEY=os.getenv('MINIO_ROOT_PASSWORD')


# SET COLUMNS --------------------------------------------------------------------
COLUMN_TYPES = {
    'age': 'category',
    'sex': 'category',
    'job': 'category',
    'housing': 'category',
    'credit_amount': 'numeric',
    'duration': 'numeric'
}

TARGET_COLUMN_NAME = 'default'
FEATURE_COLUMNS = [i for i in COLUMN_TYPES.keys()]
FEATURE_TYPES = {i: COLUMN_TYPES[i] for i in COLUMN_TYPES if i != TARGET_COLUMN_NAME}

COLUMNS_TO_SCALE = [key for key in COLUMN_TYPES.keys() if COLUMN_TYPES[key] == "numeric"]
COLUMNS_TO_ENCODE = [key for key in COLUMN_TYPES.keys() if COLUMN_TYPES[key] == "category"]

# GET DATA --------------------------------------------------------------------------
job_list = {
    0: 'unskilled and non-resident', 
    1: 'unskilled and resident', 
    2: 'skilled', 
    3: 'highly skilled'
}

conn = psycopg2.connect(dbname='credit',
                                user=DB_USER,
                                password=DB_PASSWORD,
                                host=DB_HOST,
                                port=POSTGRESQL_PORT)
cur = conn.cursor()
cur.execute(f"SELECT {reduce(lambda a,b: a + ', ' + b, FEATURE_COLUMNS)} FROM credit;")
X = (
    pd
    .DataFrame(cur.fetchall(), columns=FEATURE_COLUMNS)
    .assign(job = lambda x: x['job'].apply(lambda x: job_list[x]))
)
conn.commit()
conn.close()

conn = psycopg2.connect(dbname='credit',
                                user=DB_USER,
                                password=DB_PASSWORD,
                                host=DB_HOST,
                                port=POSTGRESQL_PORT)
cur = conn.cursor()
cur.execute(f'SELECT cr."{TARGET_COLUMN_NAME}" FROM credit cr;')
y = [x[0] for x in cur.fetchall()]
conn.commit()
conn.close()


# GET DATA

df = X.join(pd.DataFrame(y, columns = [TARGET_COLUMN_NAME]))
mlflow_dataset = mlflow.data.from_pandas(df, name='german_credit', targets='default')


# MLFLOW CONNECTION ------------------------------------------------------------------
warnings.filterwarnings('ignore')
mlflow.set_tracking_uri('http://mlflow:5000/')
# print("URI", mlflow.get_tracking_uri())
# -----------------------------------------------------------------------------------

# CREATE PIPELINE AND X_PREPROCESSED -------------------------------------------------
numeric_transformer = Pipeline(steps = [
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline(steps = [
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])
# 
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, COLUMNS_TO_SCALE),
        ("cat", categorical_transformer, COLUMNS_TO_ENCODE)
    ]
)
X_preproccessed = preprocessor.fit_transform(X)
#-------------------------------------------------------------------------------

# TRAIN TEST SPLIT -------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size = 0.25
)
# ---------------------------------------------------------------------------------

# OPTIMIZE HYPERPARAMETERS AND WRITE IT TO MLFLOW  -------------------------------

def objective(trial):
    global mlflow
    global X_preproccessed    
    # CatBoostClassifier hyperparams
    params = {
        "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
        "depth": trial.suggest_int("depth", 1, 12),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        ),
        "used_ram_limit": "3gb",
    }
    with mlflow.start_run(run_name = f"run_{datetime.now().strftime('%d.%m.%Y_%H:%M:%S')}", nested=True):
        
        # model
        mlflow.log_params(params)
        estimator = CatBoostClassifier(**params, verbose=False)
        
        accuracy = cross_val_score(estimator, X_preproccessed, y, cv=3, scoring= 'accuracy').mean()
        mlflow.log_metric('Accuracy', accuracy) 
        return accuracy

experiment_name = f"credit_pred_{datetime.now().strftime('01.%m.%Y')}"
try:
    mlflow.create_experiment(experiment_name)
except:
    pass
mlflow.set_experiment(experiment_name)

with mlflow.start_run(run_name = f"params_opt_{datetime.now().strftime('%d.%m.%Y_%H:%M')}") as run:
    study = optuna.create_study(direction="maximize", study_name=f"params_opt_{datetime.now().strftime('%Y%m%d-%H%M%S')}")

    # Hyperparams searching
    study.optimize(objective, n_trials=2)
    
    # best result is
    params = study.best_params

    estimator = CatBoostClassifier(**params, verbose=False)  
    catboostclassifier = Pipeline(steps = [
        ("preprocessor", preprocessor),
        ("classifier", estimator)
    ])
    # catboostclassifier = CatBoostClassifier(**params, verbose=False)
    catboostclassifier.fit(X_train, y_train)
    
    pred_test = catboostclassifier.predict(X_test)
    signature = infer_signature(X_test, pred_test)

    
    metrics = {'accuracy': accuracy_score(pred_test, y_test),
                'precision': precision_score(pred_test, y_test),
                'recall': recall_score(pred_test, y_test),
                'f1': f1_score(pred_test, y_test),
                'roc_auc': roc_auc_score(y_test, catboostclassifier.predict_proba(X_test)[:,1])
            }

    
    input_example = X_train.iloc[[0], :]
    mlflow.models.infer_signature(input_example, 0, params)
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)

    mlflow.log_input(mlflow_dataset)

    mlflow.sklearn.log_model(sk_model=catboostclassifier, 
                            artifact_path='catboostclassifier', 
                            signature=signature,
                            input_example=input_example
                            )

    # save model.pkl
with open('model.pkl', 'wb') as f:
    pickle.dump(catboostclassifier, f)

# minio save model.pkl
    # Create a client with the MinIO server playground, its access key
    # and secret key.
client = Minio("s3:9099",
    access_key=ACCESS_KEY,
    secret_key=SECRET_KEY,
    secure=False
)

# The file to upload, change this path if needed
source_file = "model.pkl"

# The destination bucket and filename on the MinIO server
bucket_name = "credit-model"
destination_file = "model.pkl"

# Make the bucket if it doesn't exist.
found = client.bucket_exists(bucket_name)
if not found:
    client.make_bucket(bucket_name)
    print("Created bucket", bucket_name)
else:
    print("Bucket", bucket_name, "already exists")

# Upload the file, renaming it in the process
client.fput_object(
    bucket_name, destination_file, source_file,
)
print(
    source_file, "successfully uploaded as object",
    destination_file, "to bucket", bucket_name,
)

# if __name__ == "main":
#     main
