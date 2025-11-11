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
from sklearn.svm import LinearSVC


# Não precisamos importar outros modelos ou o utils.py

def setup_logger(log_filename):
    """Configura o logger para salvar em um arquivo e exibir no console."""
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
    """Instancia um modelo de ML (focado nos 4 solicitados)."""
    if algorithm_name == 'knn':
        return KNeighborsClassifier()
    elif algorithm_name == 'dt':
        return DecisionTreeClassifier(random_state=42)
    elif algorithm_name == 'nb':
        return GaussianNB()
    elif algorithm_name == 'linear_svc':
        return LinearSVC(max_iter=2000, random_state=42, dual=False)
    else:
        # Mantém os outros caso você mude a lista 'algorithms_to_test'
        logger = logging.getLogger()
        logger.warning(f"Algoritmo {algorithm_name} não otimizado, usando fallback se disponível.")
        if algorithm_name == 'svm':
            return SVC(random_state=42)
        elif algorithm_name == 'rf':
            return RandomForestClassifier(random_state=42)
        elif algorithm_name == 'xgboost':
            return xgb.XGBClassifier(eval_metric='mlogloss', random_state=42, tree_method="hist", n_estimators=50,
                                     max_depth=4)
        elif algorithm_name == 'sgd':
            return SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
        elif algorithm_name == 'lightgbm':
            return lgb.LGBMClassifier(random_state=42, verbosity=-1)
        else:
            raise ValueError(f"Baseline: Unsupported algorithm {algorithm_name}")


def save_confusion_matrix_pdf(y_true, y_pred, class_names, output_filename, dataset_name, algorithm_name):
    """Calcula, plota e salva a matriz de confusão com contagens brutas (raw counts) em um arquivo PDF."""
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
            f'Matriz de Confusão (Contagem Bruta) (Baseline "Sem Limpeza")\nDataset: {dataset_name} | Algoritmo: {algorithm_name.upper()}',
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
    # Lista de datasets para testar
    datasets_to_test = ['ereninho', 'batadal', 'wadi', 'wustl', 'drone', 'ransomset']
    # Lista de algoritmos conforme solicitado
    algorithms_to_test = ['nb', 'dt', 'knn', 'linear_svc']
    k_folds = 5

    if not os.path.exists('results'): os.makedirs('results')
    # Nome de log e CSV específicos para este script
    log_filename = "results/baseline_sem_limpeza_log.txt"
    output_filename_csv = "results/baseline_sem_limpeza_summary.csv"
    logger = setup_logger(log_filename)

    start_time_total = time.time()
    all_results = []

    logger.info("--- Starting Baseline Evaluation Script (Original Uncleaned Features) ---")

    for dataset_name in datasets_to_test:
        logger.info(f"========== PROCESSING DATASET: {dataset_name.upper()} ==========")

        # --- Bloco de Carregamento de Dados Originais (sem limpeza do utils.py) ---
        try:
            if dataset_name == 'ereninho':
                df_orig = pd.read_csv('data/ERENINHO_10k.csv', sep=',', skipinitialspace=True)
                target_col_orig = 'class'
            elif dataset_name == 'batadal':
                df1 = pd.read_csv('data/BATADAL_dataset03.csv', sep=',', skipinitialspace=True)
                df2 = pd.read_csv('data/BATADAL_dataset04.csv', sep=',', skipinitialspace=True)
                df1.columns = df1.columns.str.strip();
                df2.columns = df2.columns.str.strip()
                df2['ATT_FLAG'] = df2['ATT_FLAG'].replace(-999, 0)
                df_orig = pd.concat([df1, df2], ignore_index=True)
                target_col_orig = 'ATT_FLAG'
                if 'DATETIME' in df_orig.columns: df_orig = df_orig.drop(columns=['DATETIME'])
            elif dataset_name == 'wadi':
                filepath_orig = 'data/WADI.csv'
                original_target_col = 'Attack LABLE (1:No Attack, -1:Attack)'
                target_col_orig = 'Attack_Label'
                df_orig = pd.read_csv(filepath_orig, sep=',', skipinitialspace=True)
                df_orig.columns = df_orig.columns.str.strip()
                if original_target_col in df_orig.columns:
                    df_orig.rename(columns={original_target_col: target_col_orig}, inplace=True)
                else:
                    raise KeyError(f"Original target column '{original_target_col}' not found for WADI.")
                cols_to_drop_null = ['2_LS_001_AL', '2_LS_002_AL', '2_P_001_STATUS', '2_P_002_STATUS']
                cols_to_drop_null_exist = [col for col in cols_to_drop_null if col in df_orig.columns]
                if cols_to_drop_null_exist: df_orig = df_orig.drop(columns=cols_to_drop_null_exist)
                cols_to_drop_meta = ['Row', 'Date', 'Time']
                cols_to_drop_meta_exist = [col for col in cols_to_drop_meta if col in df_orig.columns]
                if cols_to_drop_meta_exist: df_orig = df_orig.drop(columns=cols_to_drop_meta_exist)
                df_orig[target_col_orig] = df_orig[target_col_orig].map({1: 0, -1: 1})
                df_orig.dropna(subset=[target_col_orig], inplace=True)
                df_orig.dropna(inplace=True)
            elif dataset_name == 'ransomset':
                filepath_orig = 'data/ransomset-multiclass-dataset.csv'
                target_col_orig = 'classe'
                df_orig = pd.read_csv(filepath_orig, sep=',', skipinitialspace=True)
                df_orig.columns = df_orig.columns.str.strip()
                if 'score_binary' in df_orig.columns:  # Remove apenas o score, não as constantes
                    df_orig = df_orig.drop(columns=['score_binary'])
            elif dataset_name == 'wustl':
                filepath_orig = 'data/wustl-ehms-2020.csv'
                target_col_orig = 'Attack Category'  # Usando multiclasse
                df_orig = pd.read_csv(filepath_orig, sep=',', skipinitialspace=True)
                df_orig.columns = df_orig.columns.str.strip()
                if 'Label' in df_orig.columns:  # Remove a coluna binária
                    df_orig = df_orig.drop(columns=['Label'])
            elif dataset_name == 'drone':
                filepath_orig = 'data/drone.csv'
                target_col_orig = 'label'
                df_orig = pd.read_csv(filepath_orig, sep=',', skipinitialspace=True)
                df_orig.columns = df_orig.columns.str.strip()
            else:
                raise ValueError(f"Unknown dataset name: {dataset_name}")

            df_orig.columns = df_orig.columns.str.strip()
            X = df_orig.drop(columns=[target_col_orig])
            y = df_orig[target_col_orig]

            # Prepara rótulos e nomes de classes
            class_names = []
            if y.dtype == 'object':
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
                class_names = le.classes_
            else:
                y_encoded = y.astype(int)
                class_names = sorted(y.unique())

        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}. Skipping. Error: {e}")
            continue

        # --- Fim do Bloco de Carregamento ---

        # Pré-processamento genérico (necessário para features 'object' e scaling)
        numeric_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
        logger.info(
            f"Using {len(numeric_features)} numeric and {len(categorical_features)} categorical features (including constants/low-variance).")

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

                cv_results = cross_validate(pipeline, X, y_encoded,
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

                y_pred = cross_val_predict(pipeline, X, y_encoded,
                                           cv=cv_strategy, n_jobs=-1)

                # Nome de arquivo específico para este script
                cm_filename = f"results/CM_Baseline_SemLimpeza_{dataset_name}_{algorithm_name}.pdf"

                save_confusion_matrix_pdf(y_encoded, y_pred, class_names,
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

    results_df.to_csv(output_filename_csv, index=False, float_format='%.4f')

    total_time = time.time() - start_time_total
    logger.info(f"Total script execution time: {total_time / 60:.2f} minutes.")
    logger.info(f"Baseline (sem limpeza) results saved to '{output_filename_csv}'")
    logger.info(f"Detailed logs saved to '{log_filename}'")


if __name__ == '__main__':
    main()