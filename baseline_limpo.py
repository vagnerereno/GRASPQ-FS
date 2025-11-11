import pandas as pd
import time
import os
import logging
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Imports de ML
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, f1_score

# Imports dos Modelos
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
import xgboost as xgb
import lightgbm as lgb

# Importa a função de carregar dados do utils
import utils


def setup_logger(log_filename):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    if logger.hasHandlers(): logger.handlers.clear()
    file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def get_baseline_model(algorithm_name):
    if algorithm_name == 'knn':
        return KNeighborsClassifier()
    elif algorithm_name == 'dt':
        return DecisionTreeClassifier(random_state=42)
    elif algorithm_name == 'nb':
        return GaussianNB()
    elif algorithm_name == 'svm':
        return SVC(random_state=42)
    elif algorithm_name == 'rf':
        return RandomForestClassifier(random_state=42)
    elif algorithm_name == 'xgboost':
        return xgb.XGBClassifier(eval_metric='mlogloss', random_state=42, tree_method="hist", n_estimators=50,
                                 max_depth=4)
    elif algorithm_name == 'linear_svc':
        return LinearSVC(max_iter=2000, random_state=42, dual=False)
    elif algorithm_name == 'sgd':
        return SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
    elif algorithm_name == 'lightgbm':
        return lgb.LGBMClassifier(random_state=42, verbosity=-1)
    else:
        raise ValueError(f"Baseline: Unsupported algorithm {algorithm_name}")


def save_confusion_matrix_pdf(y_true, y_pred, class_names, output_filename, dataset_name, algorithm_name):
    try:
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm,
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)

        plt.title(
            f'Matriz de Confusão (Contagem Bruta) (Baseline)\nDataset: {dataset_name} | Algoritmo: {algorithm_name.upper()}',
            fontsize=14)
        plt.ylabel('Classe Verdadeira', fontsize=12)
        plt.xlabel('Classe Prevista', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        plt.savefig(output_filename, format='pdf')
        plt.close()

    except Exception as e:
        logger = logging.getLogger()
        logger.error(f"Falha ao gerar matriz de confusão para {output_filename}: {e}")

def main():
    datasets_to_test = ['ereninho', 'batadal', 'wadi', 'wustl', 'drone', 'ransomset']
    algorithms_to_test = ['nb', 'dt', 'knn', 'linear_svc']
    k_folds = 5

    if not os.path.exists('results'): os.makedirs('results')
    log_filename = "results/baseline_evaluation_log.txt"
    logger = setup_logger(log_filename)

    start_time_total = time.time()
    all_results = []

    logger.info("--- Starting Baseline Evaluation Script (All Features Post-EDA) ---")

    for dataset_name in datasets_to_test:
        logger.info(f"========== PROCESSING DATASET: {dataset_name.upper()} ==========")

        try:
            X_cleaned, y_cleaned, feature_names_cleaned = utils.load_unified_dataset(dataset_name=dataset_name)

            class_names = []
            if y_cleaned.dtype == 'object':
                le = LabelEncoder()
                y_cleaned_encoded = le.fit_transform(y_cleaned)
                class_names = le.classes_
            else:
                y_cleaned_encoded = y_cleaned.astype(int)
                class_names = sorted(y_cleaned.unique())

        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}. Skipping. Error: {e}")
            continue

        numeric_features = X_cleaned.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X_cleaned.select_dtypes(exclude=np.number).columns.tolist()
        logger.info(f"Found {len(numeric_features)} numeric and {len(categorical_features)} categorical features.")

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ],
            remainder='passthrough'
        )

        cv_strategy = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

        for algorithm_name in algorithms_to_test:
            logger.info(f"--- Testing Algorithm: {algorithm_name} on {dataset_name} ---")

            try:
                model = get_baseline_model(algorithm_name)
                pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

                cv_results = cross_validate(pipeline, X_cleaned, y_cleaned_encoded,
                                            cv=cv_strategy,
                                            scoring='f1_weighted',
                                            n_jobs=-1,
                                            return_train_score=False)

                mean_f1 = np.mean(cv_results['test_score'])
                std_f1 = np.std(cv_results['test_score'])
                mean_time = np.mean(cv_results['fit_time']) + np.mean(cv_results['score_time'])

                logger.info(f"Result: F1-Score = {mean_f1:.4f} (+/- {std_f1:.4f}), Avg. Time = {mean_time:.2f}s")

                all_results.append({
                    'dataset': dataset_name,
                    'algorithm': algorithm_name,
                    'mean_f1_score': mean_f1,
                    'std_f1_score': std_f1,
                    'mean_exec_time_s': mean_time
                })

                logger.info(f"Generating predictions for Confusion Matrix ({algorithm_name} on {dataset_name})...")

                y_pred = cross_val_predict(pipeline, X_cleaned, y_cleaned_encoded,
                                           cv=cv_strategy, n_jobs=-1)

                cm_filename = f"results/CM_Baseline_{dataset_name}_{algorithm_name}.pdf"

                save_confusion_matrix_pdf(y_cleaned_encoded, y_pred, class_names,
                                          cm_filename, dataset_name, algorithm_name)

                logger.info(f"Confusion Matrix PDF saved to {cm_filename}")

            except Exception as e:
                logger.exception(f"Failed to evaluate {algorithm_name} on {dataset_name}. Error: {e}")
                all_results.append({
                    'dataset': dataset_name, 'algorithm': algorithm_name,
                    'mean_f1_score': np.nan, 'std_f1_score': np.nan, 'mean_exec_time_s': np.nan,
                    'error': str(e)
                })

    logger.info("--- All evaluations complete. Saving results. ---")
    results_df = pd.DataFrame(all_results)
    output_filename = "results/baseline_summary.csv"
    results_df.to_csv(output_filename, index=False, float_format='%.4f')

    total_time = time.time() - start_time_total
    logger.info(f"Total script execution time: {total_time / 60:.2f} minutes.")
    logger.info(f"Baseline results saved to '{output_filename}'")


if __name__ == '__main__':
    main()