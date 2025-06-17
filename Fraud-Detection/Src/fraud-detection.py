import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler


df = pd.read_csv('./Data/creditcard.csv')


X = df.drop('Class', axis=1)
y = df['Class']


amount_scaler = MinMaxScaler()
time_scaler = MinMaxScaler()
v_scaler = StandardScaler()


if 'Time' in X.columns:
    X['Time'] = time_scaler.fit_transform(X[['Time']])
    
if 'Amount' in X.columns:
    X['Amount'] = amount_scaler.fit_transform(X[['Amount']])
    

v_columns = [col for col in X.columns if col.startswith('V')]
if v_columns:
    X[v_columns] = v_scaler.fit_transform(X[v_columns])
    

X = X.values
y = y.values


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)