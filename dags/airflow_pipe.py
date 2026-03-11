import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer # т.н. преобразователь колонок
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import root_mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import requests
from pathlib import Path
import os
from datetime import timedelta
from train_model import train

def download_data():
    df = pd.read_csv('https://raw.githubusercontent.com/UsumiMin/lab_ML_1/refs/heads/main/StudentsPerformance.csv', delimiter = ',')
    df.to_csv("students.csv", index = False)
    print("df: ", df.shape)
    return df

def clear_data():
    df = pd.read_csv("students.csv")
    
    cat_columns = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
    num_columns = ['math score', 'reading score', 'writing score']
    
    question_score = df[(df['math score'] < 0) | (df['math score'] > 100)]
    df = df.drop(question_score.index)
    question_score = df[(df['reading score'] < 0) | (df['reading score'] > 100)]
    df = df.drop(question_score.index)
    question_score = df[(df['writing score'] < 0) | (df['writing score'] > 100)]
    df = df.drop(question_score.index)

    question_gender = df[~((df['gender'] == 'male') | (df['gender'] == 'female'))]
    df = df.drop(question_gender.index)
    
    df = df.reset_index(drop=True)  
    ordinal = OrdinalEncoder()
    ordinal.fit(df[cat_columns]);
    Ordinal_encoded = ordinal.transform(df[cat_columns])
    df_ordinal = pd.DataFrame(Ordinal_encoded, columns=cat_columns)
    df[cat_columns] = df_ordinal[cat_columns]
    df.to_csv('df_clear.csv')
    return True

dag_students = DAG(
    dag_id="train_pipe_students",
    start_date=datetime(2026, 3, 11),
    max_active_tasks=4,
    schedule=timedelta(minutes=5),
#    schedule="@hourly",
    max_active_runs=1,
    catchup=False,
)
download_task = PythonOperator(python_callable=download_data, task_id = "download_students", dag = dag_students)
clear_task = PythonOperator(python_callable=clear_data, task_id = "clear_students", dag = dag_students)
train_task = PythonOperator(python_callable=train, task_id = "train_students", dag = dag_students)
download_task >> clear_task >> train_task
