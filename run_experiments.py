import pandas as pd
from argparse import Namespace
import time
import os
import logging
from datetime import datetime
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
import xgboost as xgb
import utils
from main import run_single_experiment


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
    # Função auxiliar para instanciar o modelo do baseline (sem n_jobs)
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
        return xgb.XGBClassifier(eval_metric='mlogloss', random_state=42)
    elif algorithm_name == 'linear_svc':
        return LinearSVC(max_iter=2000, random_state=42, dual=False)
    elif algorithm_name == 'sgd':
        return SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
    else:
        raise ValueError(f"Baseline: Unsupported algorithm {algorithm_name}")


def main():
    parameter_to_vary = "priority_queue"
    values_to_test = range(1, 5)  # Teste rápido

    fixed_args = {
        "dataset": "batadal",
        "algorithm": "dt",
        "rcl_size": 17,
        "initial_solution": 5,
        "local_iterations": 100,
        "constructive_iterations": 100,
        "k_folds": 5
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    algorithm_name = fixed_args.get('algorithm', 'unknown_alg')
    dataset_name = fixed_args.get('dataset', 'unknown_data')
    base_filename = f"{dataset_name}_{algorithm_name}_varying_{parameter_to_vary}_{timestamp}"
    if not os.path.exists('results'): os.makedirs('results')
    log_filename = f"results/log_{base_filename}.txt"
    output_filename = f"results/summary_{base_filename}.csv"
    logger = setup_logger(log_filename)

    start_time = time.time()
    baseline_results = {}

    logger.info("--- Starting Automated Experiments Run ---")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Algorithm for Baselines & GRASP: {algorithm_name}")
    logger.info(f"Parameter to vary: '{parameter_to_vary}'")
    logger.info(f"Values to test: {list(values_to_test)}")
    logger.info(
        f"Other fixed parameters: { {k: v for k, v in fixed_args.items() if k not in ['dataset', 'algorithm']} }")
    logger.info(f"Detailed logs will be saved to: {log_filename}")
    logger.info(f"Summary results will be saved to: {output_filename}")

    logger.info("--- Loading Data ---")
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
            filepath_orig = 'data/WADI_attackdataLABLE.csv'
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
        else:
            raise ValueError("Unknown dataset name.")

        df_orig.columns = df_orig.columns.str.strip()
        X_orig = df_orig.drop(columns=[target_col_orig])
        y_orig = df_orig[target_col_orig]
        if y_orig.dtype == 'object':
            y_orig_encoded = LabelEncoder().fit_transform(y_orig)
        else:
            y_orig_encoded = y_orig.astype(int)

        X_cleaned, y_cleaned, feature_names_cleaned = utils.load_unified_dataset(dataset_name=dataset_name)
        if y_cleaned.dtype == 'object':
            y_cleaned_encoded = LabelEncoder().fit_transform(y_cleaned)
        else:
            y_cleaned_encoded = y_cleaned.astype(int)

    except Exception as e:
        logger.exception(f"CRITICAL ERROR: Failed to load data. Aborting experiment run. Error: {e}")
        return

    cv_strategy = StratifiedKFold(n_splits=fixed_args['k_folds'], shuffle=True, random_state=42)
    model_base = get_baseline_model(algorithm_name)

    logger.info("--- Calculating Baseline 1 (All Original Features) ---")
    try:
        numeric_features_orig = X_orig.select_dtypes(include=np.number).columns.tolist()
        categorical_features_orig = X_orig.select_dtypes(exclude=np.number).columns.tolist()
        preprocessor_orig = ColumnTransformer(
            transformers=[('num', StandardScaler(), numeric_features_orig),
                          ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
                           categorical_features_orig)],
            remainder='passthrough')
        pipeline_orig = Pipeline(steps=[('preprocessor', preprocessor_orig), ('classifier', model_base)])
        baseline1_scores = cross_val_score(pipeline_orig, X_orig, y_orig_encoded,
                                           cv=cv_strategy, scoring='f1_weighted', n_jobs=-1)
        baseline_results["baseline1_mean_f1"] = np.mean(baseline1_scores)
        baseline_results["baseline1_std_f1"] = np.std(baseline1_scores)
        logger.info(
            f"Baseline 1 Results: Mean F1 = {baseline_results['baseline1_mean_f1']:.4f} (+/- {baseline_results['baseline1_std_f1']:.4f})")
    except Exception as e:
        logger.exception("Failed to evaluate Baseline 1.")
        baseline_results["baseline1_mean_f1"] = -1;
        baseline_results["baseline1_std_f1"] = -1

    logger.info("--- Calculating Baseline 2 (Post-EDA Features) ---")
    try:
        pipeline_cleaned = Pipeline(steps=[('scaler', StandardScaler()), ('classifier', model_base)])
        baseline2_scores = cross_val_score(pipeline_cleaned, X_cleaned, y_cleaned_encoded,
                                           cv=cv_strategy, scoring='f1_weighted', n_jobs=-1)
        baseline_results["baseline2_mean_f1"] = np.mean(baseline2_scores)
        baseline_results["baseline2_std_f1"] = np.std(baseline2_scores)
        logger.info(
            f"Baseline 2 Results: Mean F1 = {baseline_results['baseline2_mean_f1']:.4f} (+/- {baseline_results['baseline2_std_f1']:.4f})")
    except Exception as e:
        logger.exception("Failed to evaluate Baseline 2.")
        baseline_results["baseline2_mean_f1"] = -1;
        baseline_results["baseline2_std_f1"] = -1

    # --- Loop de Experimentos GRASP ---
    all_results_list = []
    logger.info("--- Starting GRASPQ-FS Parameter Variation Loop ---")
    for i, value in enumerate(values_to_test):
        logger.info("=" * 80)
        logger.info(f"EXECUTING GRASP EXPERIMENT {i + 1}/{len(list(values_to_test))}: '{parameter_to_vary}' = {value}")
        logger.info("=" * 80)

        current_args_dict = fixed_args.copy()
        current_args_dict[parameter_to_vary] = value
        args = Namespace(**current_args_dict)

        try:
            grasp_summary = run_single_experiment(args, X_cleaned, y_cleaned_encoded, feature_names_cleaned)

            experiment_row = {**vars(args), **grasp_summary, **baseline_results}
            all_results_list.append(experiment_row)
        except Exception as e:
            logger.exception(
                f"!!! GRASP Experiment {i + 1} failed for {parameter_to_vary}={value} due to an error: {e}")
            break

    if all_results_list:
        results_df = pd.DataFrame(all_results_list)
        rounding_map = {
            'baseline1_mean_f1': 4, 'baseline1_std_f1': 4,
            'baseline2_mean_f1': 4, 'baseline2_std_f1': 4,
            'grasp_mean_f1_score': 4, 'grasp_std_f1_score': 4,
            'grasp_best_f1_overall': 4,
            'grasp_mean_construction_time': 2, 'grasp_std_construction_time': 2,
            'grasp_mean_ls_time': 2, 'grasp_std_ls_time': 2
        }
        for col, decimals in rounding_map.items():
            if col in results_df.columns:
                try:
                    results_df[col] = results_df[col].round(decimals)
                except TypeError:
                    logger.warning(f"Could not round column '{col}'.")

        results_df.to_csv(output_filename, index=False)
        logger.info(f"Aggregated results saved to '{output_filename}'")
    else:
        logger.warning("No GRASP experiments completed successfully. CSV summary file was not generated.")

    total_time = time.time() - start_time
    logger.info("=" * 80)
    logger.info("--- Experiment Run Finished ---")
    logger.info(f"Total execution time: {total_time / 60:.2f} minutes.")
    logger.info(f"Detailed logs saved to: {log_filename}")


if __name__ == '__main__':
    main()