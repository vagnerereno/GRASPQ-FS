# main.py

import time
import random
import logging
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.feature_selection import mutual_info_classif

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
import xgboost as xgb
import lightgbm as lgb
import utils
import shap
from priority_queue import MaxPriorityQueue

def _shap_to_ncf(values, n_samples, n_features):
    arr = values.values if hasattr(values, "values") else values
    if isinstance(arr, (list, tuple)) and len(arr) > 0 and hasattr(arr[0], "shape"):
        arr = np.stack(arr, axis=1)
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[:, None, :]
    elif arr.ndim == 3:
        if arr.shape[0] == n_samples and arr.shape[1] == n_features:
            arr = np.transpose(arr, (0, 2, 1))
        elif arr.shape[0] == n_samples and arr.shape[2] == n_features:
            pass
        elif arr.shape[1] == n_samples and arr.shape[2] == n_features:
            arr = np.transpose(arr, (1, 0, 2))
        elif arr.shape[0] == n_samples and arr.shape[1] != n_features and arr.shape[2] != n_features:
            arr = np.transpose(arr, (0, 2, 1))
    return arr


def compute_shap_mean_abs(model_xgb, X_scaled, shap_n=4000):
    n = X_scaled.shape[0]
    m = min(int(shap_n), n)
    rng = np.random.RandomState(42)
    idx = rng.choice(n, size=m, replace=False)

    if isinstance(X_scaled, pd.DataFrame):
        Xs = X_scaled.iloc[idx].reset_index(drop=True)
    else:
        Xs = X_scaled[idx]

    explainer = shap.TreeExplainer(model_xgb)
    sv = explainer.shap_values(Xs)
    sv3 = _shap_to_ncf(sv, n_samples=Xs.shape[0], n_features=Xs.shape[1])
    shap_abs = np.abs(sv3).mean(axis=(0, 1))

    if isinstance(X_scaled, pd.DataFrame):
        return pd.Series(shap_abs, index=Xs.columns, name="SHAP_XGB_mean_abs")
    else:
        return shap_abs

def evaluate_algorithm(features_idx, algorithm, X_train_fold, y_train_fold, X_val_fold, y_val_fold, feature_names):
    """Evaluates a subset of features using a specified ML algorithm for a given data fold."""
    features = [feature_names[i] for i in features_idx]

    if algorithm == 'knn':
        model = KNeighborsClassifier(n_jobs=-1)
    elif algorithm == 'dt':
        model = DecisionTreeClassifier(random_state=42)
    elif algorithm == 'nb':
        model = GaussianNB()
    elif algorithm == 'svm': # Warning: This is EXTREMELY SLOW on large datasets
        model = SVC(random_state=42)
    elif algorithm == 'rf':
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
    elif algorithm == 'xgboost':
        model = xgb.XGBClassifier(eval_metric='mlogloss', random_state=42, n_jobs=-1, tree_method="hist", n_estimators=50, max_depth=4, subsample=0.8, colsample_bytree=0.8)
    elif algorithm == 'linear_svc':
        model = LinearSVC(max_iter=2000, random_state=42, dual=False)
    elif algorithm == 'sgd':
        model = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42, n_jobs=-1)
    elif algorithm == 'lightgbm':
        model = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbosity=-1)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    return utils.evaluate_model(model, X_train_fold[features], y_train_fold, X_val_fold[features], y_val_fold)


def local_search(initial_solution, algorithm, args, X_train_fold, y_train_fold, X_val_fold, y_val_fold, feature_names,
                 sorted_features):
    """Performs local search to improve an initial solution, with detailed logging at DEBUG level."""
    logger = logging.getLogger()
    max_f1_score = evaluate_algorithm(initial_solution, algorithm, X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                                      feature_names)
    best_solution = initial_solution.copy()
    seen_solutions = {frozenset(initial_solution)}

    logger.debug(f"    [Local Search] Starting for solution {best_solution} | Initial F1-Score: {max_f1_score:.4f}")

    for iteration in range(args.local_iterations):
        current_solution = best_solution.copy()
        replace_index = random.randrange(len(current_solution))
        rcl_indices = [feature_names.index(feat) for feat, _ in sorted_features[:args.rcl_size]]
        candidate_features = [idx for idx in rcl_indices if idx not in current_solution]
        if not candidate_features:
            logger.debug(f"      [LS Iteration {iteration + 1}] RCL is empty, cannot find new features.")
            break
        new_feature = random.choice(candidate_features)
        neighbor_solution = current_solution[:]
        neighbor_solution[replace_index] = new_feature
        neighbor_solution.sort()
        if frozenset(neighbor_solution) in seen_solutions:
            continue
        seen_solutions.add(frozenset(neighbor_solution))
        f1_score = evaluate_algorithm(neighbor_solution, algorithm, X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                                      feature_names)
        logger.debug(
            f"      [LS Iteration {iteration + 1}/{args.local_iterations}] Neighbor {neighbor_solution} -> F1-Score: {f1_score:.4f}")
        if f1_score > max_f1_score:
            max_f1_score = f1_score
            best_solution = neighbor_solution
            logger.debug(
                f"        >>> Improvement found! New best F1: {max_f1_score:.4f} with solution {best_solution}")

    return max_f1_score, best_solution


def construction(args, X_train_fold, y_train_fold, X_val_fold, y_val_fold, feature_names, sorted_features, fold_number):
    """GRASP construction and local search phases for a single fold, with detailed logging."""
    logger = logging.getLogger()
    RCL_FEATURES = [feature for feature, _ in sorted_features[:args.rcl_size]]
    priority_queue = MaxPriorityQueue()
    max_f1_score = -1
    best_overall_solution = []

    logger.info(f"  [Constructive Phase] Starting...")
    start_time_construction = time.perf_counter()

    for i in range(args.constructive_iterations):
        selected_features_names = random.sample(RCL_FEATURES, k=args.initial_solution)
        solution_indices = sorted([feature_names.index(name) for name in selected_features_names])
        f1_score = evaluate_algorithm(solution_indices, args.algorithm, X_train_fold, y_train_fold, X_val_fold,
                                      y_val_fold, feature_names)
        logger.debug(
            f"    [Constructive Iteration {i + 1}/{args.constructive_iterations}] Solution {solution_indices} -> F1-Score: {f1_score:.4f}")
        priority_queue.insert((f1_score, solution_indices))

    construction_time = time.perf_counter() - start_time_construction
    logger.info(f"  [Constructive Phase] Finished in {construction_time:.2f}s.")

    best_solutions_for_ls = []
    while not priority_queue.is_empty() and len(best_solutions_for_ls) < args.priority_queue:
        best_solutions_for_ls.append(priority_queue.extract_max())

    logger.info(f"  Top {len(best_solutions_for_ls)} solutions from Priority Queue selected for Local Search:")
    for f1, sol in best_solutions_for_ls:
        logger.debug(f"    - Solution: {sol}, F1-Score: {f1:.4f}")

    start_time_ls = time.perf_counter()
    logger.info(f"  [Local Search Phase] Starting...")

    for initial_f1, solution in best_solutions_for_ls:
        improved_f1, improved_solution = local_search(
            solution, args.algorithm, args,
            X_train_fold, y_train_fold, X_val_fold, y_val_fold,
            feature_names, sorted_features
        )
        if improved_f1 > max_f1_score:
            max_f1_score = improved_f1
            best_overall_solution = improved_solution

    local_search_time = time.perf_counter() - start_time_ls
    logger.info(f"  [Local Search Phase] Finished in {local_search_time:.2f}s.")

    best_feature_names = sorted([feature_names[i] for i in best_overall_solution])
    logger.info(f"Best solution in fold {fold_number + 1}: F1={max_f1_score:.4f}, Features={best_feature_names}")

    return max_f1_score, best_overall_solution, construction_time, local_search_time


def print_feature_scores(sorted_features):
    logger = logging.getLogger()
    logger.info("Feature ranking complete. Detailed scores saved to the log file.")

    logger.debug("--- Feature Ranking Scores ---")
    for feature, score in sorted_features:
        logger.debug(f"  Feature: {feature:<40} Score = {score:.4f}")
    logger.debug("------------------------------")


def run_single_experiment(args, X_cleaned, y_cleaned_encoded, feature_names_cleaned):
    logger = logging.getLogger()
    logger.info(f"--- Starting GRASPQ-FS Run for params: {vars(args)} ---")
    logger.info(f"Using ranking method: {args.ranker}")

    try:
        if not isinstance(X_cleaned, pd.DataFrame):
            X_cleaned = pd.DataFrame(X_cleaned, columns=feature_names_cleaned)

        if args.ranker == 'mi':
            logger.info("Ranking cleaned features using Mutual Information...")
            X_for_mi = pd.get_dummies(X_cleaned, columns=X_cleaned.select_dtypes(exclude=np.number).columns)
            feature_names_ranked = X_for_mi.columns.tolist()
            ig_scores = mutual_info_classif(X_for_mi, y_cleaned_encoded, random_state=42)
            scores_series = pd.Series(ig_scores, index=feature_names_ranked)

        elif args.ranker == 'shap':
            logger.info("Ranking cleaned features using SHAP with XGBoost...")
            logger.info("Preparing data for SHAP model training...")
            numeric_features = X_cleaned.select_dtypes(include=np.number).columns.tolist()
            categorical_features = X_cleaned.select_dtypes(exclude=np.number).columns.tolist()

            preprocessor_shap = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numeric_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
                ],
                remainder='passthrough'
            )

            X_processed_shap = preprocessor_shap.fit_transform(X_cleaned)

            feature_names_ranked = numeric_features[:]
            if categorical_features:
                try:
                    ohe_feature_names = preprocessor_shap.named_transformers_['cat'].get_feature_names_out(
                        categorical_features).tolist()
                except AttributeError:
                    ohe_feature_names = preprocessor_shap.named_transformers_['cat'].get_feature_names_out().tolist()
                feature_names_ranked.extend(ohe_feature_names)

            X_scaled_shap_df = pd.DataFrame(X_processed_shap, columns=feature_names_ranked)

            logger.info("Training temporary XGBoost model for SHAP ranking...")
            model_xgb_shap = xgb.XGBClassifier(
                random_state=42, n_estimators=500, learning_rate=0.05,
                max_depth=6, subsample=0.8, colsample_bytree=0.8,
                eval_metric='mlogloss', tree_method="hist", n_jobs=-1
            ).fit(X_scaled_shap_df, y_cleaned_encoded)

            logger.info("Calculating SHAP values...")
            scores_series = compute_shap_mean_abs(model_xgb_shap, X_scaled_shap_df, shap_n=4000)

        else:
            raise ValueError(f"Unknown ranker: {args.ranker}")

        sorted_features = sorted(scores_series.items(), key=lambda x: x[1], reverse=True)
        print_feature_scores(sorted_features)

    except Exception as e:
        logger.exception(f"Error during feature ranking ({args.ranker}): {e}")
        return {"error": f"{args.ranker} ranking failed"}

    cv_strategy = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=42)
    fold_f1_scores, fold_construction_times, fold_ls_times = [], [], []
    best_overall_f1 = -1
    best_overall_solution_info = {}

    for fold, (train_index, val_index) in enumerate(cv_strategy.split(X_cleaned, y_cleaned_encoded)):
        logger.info(f"========== EXECUTING GRASP FOLD {fold + 1}/{args.k_folds} ==========")
        if len(train_index) == 0 or len(val_index) == 0:
            logger.error(f"Empty train/validation set in fold {fold + 1}. Skipping.");
            continue

        try:
            X_train_fold, X_val_fold = X_cleaned.iloc[train_index], X_cleaned.iloc[val_index]
            y_train_fold, y_val_fold = y_cleaned_encoded[train_index], y_cleaned_encoded[val_index]
        except Exception as e:
            logger.exception(f"Error during data splitting in fold {fold + 1}: {e}. Skipping.");
            continue

        try:
            numeric_features_fold = X_train_fold.select_dtypes(include=np.number).columns.tolist()
            categorical_features_fold = X_train_fold.select_dtypes(exclude=np.number).columns.tolist()

            preprocessor_fold = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numeric_features_fold),
                    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features_fold)
                ],
                remainder='passthrough'
            )

            X_train_processed_np = preprocessor_fold.fit_transform(X_train_fold)
            X_val_processed_np = preprocessor_fold.transform(X_val_fold)

            feature_names_fold = numeric_features_fold[:]
            if categorical_features_fold:
                try:
                    ohe_feature_names = preprocessor_fold.named_transformers_['cat'].get_feature_names_out(
                        categorical_features_fold).tolist()
                except AttributeError:
                    ohe_feature_names = preprocessor_fold.named_transformers_['cat'].get_feature_names_out().tolist()
                feature_names_fold.extend(ohe_feature_names)

            X_train_processed = pd.DataFrame(X_train_processed_np, columns=feature_names_fold, index=X_train_fold.index)
            X_val_processed = pd.DataFrame(X_val_processed_np, columns=feature_names_fold, index=X_val_fold.index)

        except Exception as e:
            logger.exception(f"Error during preprocessing (ColumnTransformer) in fold {fold + 1}: {e}. Skipping.");
            continue

        try:
            ig_scores_fold = mutual_info_classif(X_train_processed, y_train_fold, random_state=42)
            sorted_features_fold = sorted(zip(feature_names_fold, ig_scores_fold), key=lambda x: x[1], reverse=True)
        except Exception as e:
            logger.error(f"Error during MI calculation in fold {fold + 1}: {e}. Skipping.");
            continue

        try:
            best_f1, best_solution_indices, const_time, ls_time = construction(
                args, X_train_processed, y_train_fold, X_val_processed, y_val_fold,
                feature_names_fold, sorted_features_fold, fold
            )
        except Exception as e:
            logger.exception(f"Error during GRASP execution in fold {fold + 1}: {e}");
            continue

        fold_f1_scores.append(best_f1)
        fold_construction_times.append(const_time)
        fold_ls_times.append(ls_time)
        if best_f1 > best_overall_f1:
            best_overall_f1 = best_f1
            best_overall_solution_info = {
                'fold': fold + 1, 'f1_score': best_f1,
                'features': [feature_names_fold[i] for i in best_solution_indices]
            }

    grasp_results = {
        "grasp_mean_f1_score": np.mean(fold_f1_scores) if fold_f1_scores else -1,
        "grasp_std_f1_score": np.std(fold_f1_scores) if fold_f1_scores else -1,
        "grasp_mean_construction_time": np.mean(fold_construction_times) if fold_construction_times else -1,
        "grasp_std_construction_time": np.std(fold_construction_times) if fold_construction_times else -1,
        "grasp_mean_ls_time": np.mean(fold_ls_times) if fold_ls_times else -1,
        "grasp_std_ls_time": np.std(fold_ls_times) if fold_ls_times else -1,
        "grasp_best_f1_overall": best_overall_f1,
        "grasp_best_solution_features": best_overall_solution_info.get('features', [])
    }

    logger.info("--- GRASPQ-FS Run Finished ---")
    logger.info("--- GRASP Results for this parameter set ---")
    for key, value in grasp_results.items():
        if isinstance(value, float) and value != -1:
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")

    return grasp_results