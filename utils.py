import json
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score)
import matplotlib.patches as mpatches

def load_data():
    # Carregamento de dados
    train_df = pd.read_csv('data/hibrid_dataset_GOOSE_train.csv', sep=',')
    test_df = pd.read_csv('data/hibrid_dataset_GOOSE_test.csv', sep=',')
    print(train_df.head())
    print(test_df.head())
    print("Classes únicas no conjunto de teste:", test_df['class'].unique())
    print("Classes únicas no conjunto de treinamento:", train_df['class'].unique())
    # Remover o ataque específico do conjunto de treinamento
    # train_df = train_df[train_df['class'] != 'random_replay']
    # train_df = train_df[train_df['class'] != 'inverse_replay']
    # train_df = train_df[train_df['class'] != 'masquerade_fake_fault']
    # train_df = train_df[train_df['class'] != 'masquerade_fake_normal']
    # train_df = train_df[train_df['class'] != 'injection']
    # train_df = train_df[train_df['class'] != 'high_StNum']
    # train_df = train_df[train_df['class'] != 'poi soned_high_rate']

    train_df = train_df.reset_index(drop=True)

    print("Classes únicas no conjunto de treinamento:", train_df['class'].unique())


    # Colunas enriquecidas para remover
    columns_to_remove = ['stDiff', 'sqDiff', 'gooseLenghtDiff', 'cbStatusDiff', 'apduSizeDiff',
                         'frameLengthDiff', 'timestampDiff', 'tDiff', 'timeFromLastChange',
                         'delay', 'isbARms', 'isbBRms', 'isbCRms', 'ismARms', 'ismBRms', 'ismCRms',
                         'ismARmsValue', 'ismBRmsValue', 'ismCRmsValue', 'csbArms', 'csvBRms',
                         'csbCRms', 'vsmARms', 'vsmBRms', 'vsmCRms', 'isbARmsValue', 'isbBRmsValue',
                         'isbCRmsValue', 'vsbARmsValue', 'vsbBRmsValue', 'vsbCRmsValue',
                         'vsmARmsValue', 'vsmBRmsValue', 'vsmCRmsValue', 'isbATrapAreaSum',
                         'isbBTrapAreaSum', 'isbCTrapAreaSum', 'ismATrapAreaSum', 'ismBTrapAreaSum',
                         'ismCTrapAreaSum', 'csvATrapAreaSum', 'csvBTrapAreaSum', 'vsbATrapAreaSum',
                         'vsbBTrapAreaSum', 'vsbCTrapAreaSum', 'vsmATrapAreaSum', 'vsmBTrapAreaSum',
                         'vsmCTrapAreaSum', 'gooseLengthDiff']

    # Remoção de colunas enriquecidas e com NaN
    # train_df = train_df.dropna(axis=1)  # .drop(columns=columns_to_remove, errors='ignore')
    # test_df = test_df.dropna(axis=1)  # .drop(columns=columns_to_remove, errors='ignore')

    # Separação de features e labels
    X_train = train_df.drop(columns=['class'])
    y_train = train_df['class']
    X_test = test_df.drop(columns=['class'])
    y_test = test_df['class']
    return X_train, y_train, X_test, y_test

def preprocess_data(X_train, y_train, X_test, y_test):
    # Identificar colunas numéricas
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    cat_cols = X_train.select_dtypes(include=['object']).columns

    # Utilizar StandardScaler para normalizar os dados numéricos
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # Utilizar OneHotEncoder para colunas categóricas
    if len(cat_cols) > 0:
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        cat_encoded = encoder.fit_transform(X_train[cat_cols])
        cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(cat_cols))

        # Drop original categorical columns and concat encoded columns
        X_train = pd.concat([X_train[num_cols], cat_encoded_df], axis=1)
        cat_encoded_test = encoder.transform(X_test[cat_cols])
        cat_encoded_test_df = pd.DataFrame(cat_encoded_test, columns=encoder.get_feature_names_out(cat_cols))
        X_test = pd.concat([X_test[num_cols], cat_encoded_test_df], axis=1)

    # Inicializar o LabelEncoder para os rótulos
    le = LabelEncoder()

    le.fit(y_train)

    y_train = le.transform(y_train)


    # Transformar y_train e y_test para numérico
    if y_train.dtype == 'object' or isinstance(y_train, pd.Series):
        y_train = le.transform(y_train)
    if y_test.dtype == 'object' or isinstance(y_test, pd.Series):
        y_test = transform_test_labels(y_test, le)
    return y_train, y_test, X_train, X_test, le

def transform_test_labels(y_test, label_encoder):
    # Encontrar o maior valor numérico atual do LabelEncoder
    max_label_value = max(label_encoder.transform(label_encoder.classes_))

    # Substituir rótulos desconhecidos por um novo valor numérico
    y_test_transformed = []
    for label in y_test:
        if label in label_encoder.classes_:
            # Transformar rótulos conhecidos
            label_encoded = label_encoder.transform([label])[0]
            y_test_transformed.append(label_encoded)
        else:
            # Adicionar um novo valor numérico para rótulos desconhecidos
            y_test_transformed.append(max_label_value + 1)

    return np.array(y_test_transformed)

def calculate_metrics(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    VN = conf_matrix[0, 0]
    FN = conf_matrix[0, 1:].sum()
    FP = conf_matrix[1:, 0].sum() + (conf_matrix[:, 1:].sum(axis=0) - np.diag(conf_matrix)[1:]).sum()
    VP = np.diag(conf_matrix)[1:].sum()

    accuracy = (VP + VN) / (VP + VN + FP + FN)
    precision = VP / (VP + FP) if (VP + FP) > 0 else 0
    recall = VP / (VP + FN) if (VP + FN) > 0 else 0
    # print(f"VN: {VN}, FN: {FN}, VP: {VP}, FP: {FP}")

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1, conf_matrix

def evaluate_model(classifier, X_train, y_train, X_test, y_test):
    # Treinar o classificador
    classifier.fit(X_train, y_train)

    # Prever os rótulos para o conjunto de teste
    y_pred = classifier.predict(X_test)

    # Calcular métricas
    accuracy, precision, recall, f1, conf_matrix = calculate_metrics(y_test, y_pred)

    # Imprimir a matriz de confusão e as métricas
    # print(conf_matrix)
    # print(f"Accuracy: {accuracy * 100:.2f}%, Precision: {precision * 100:.2f}%, Recall: {recall * 100:.2f}%, F1-Score: {f1 * 100:.2f}%")

    return f1
