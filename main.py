# main.py

import time
import random
import logging
from collections import Counter

import numpy as np
import pandas as pd

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

import utils
from priority_queue import MaxPriorityQueue


def evaluate_algorithm(features_idx, algorithm, X_train_fold, y_train_fold, X_val_fold, y_val_fold, feature_names):
    """Evaluates a subset of features using a specified ML algorithm for a given data fold."""
    features = [feature_names[i] for i in features_idx]

    if algorithm == 'knn':
        model = KNeighborsClassifier()
    elif algorithm == 'dt':
        model = DecisionTreeClassifier(random_state=42)
    elif algorithm == 'nb':
        model = GaussianNB()
    elif algorithm == 'svm':
        model = SVC(random_state=42)
    elif algorithm == 'rf':
        model = RandomForestClassifier(random_state=42)
    elif algorithm == 'xgboost':
        model = xgb.XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, random_state=42)
    elif algorithm == 'linear_svc':
        model = LinearSVC(max_iter=2000, random_state=42, dual=True)
    elif algorithm == 'sgd':
        model = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
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

    logger.debug("--- Mutual Information Feature Scores ---")
    for feature, score in sorted_features:
        logger.debug(f"  Feature: {feature:<40} MI = {score:.4f}")
    logger.debug("---------------------------------------")


def run_single_experiment(args):
    """
    Executes a complete cross-validation experiment for a given set of parameters.
    Uses the logger that was configured by the calling script.
    Returns a dictionary with the consolidated results.
    """
    logger = logging.getLogger()
    logger.info(f"Starting experiment with parameters: {vars(args)}")

    X, y = utils.load_unified_dataset()
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X_for_ranking = pd.get_dummies(X)
    feature_names_ranked = X_for_ranking.columns.tolist()
    logger.info("Ranking all features using Mutual Information...")
    ig_scores = mutual_info_classif(X_for_ranking, y_encoded, random_state=42)
    sorted_features = sorted(zip(feature_names_ranked, ig_scores), key=lambda x: x[1], reverse=True)

    print_feature_scores(sorted_features)

    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=42)
    fold_f1_scores, fold_construction_times, fold_ls_times = [], [], []
    best_overall_f1 = -1
    best_overall_solution_info = {}

    for fold, (train_index, val_index) in enumerate(skf.split(X, y_encoded)):
        logger.info(f"========== EXECUTING FOLD {fold + 1}/{args.k_folds} ==========")
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y_encoded[train_index], y_encoded[val_index]

        num_cols = X_train_fold.select_dtypes(include=np.number).columns.tolist()
        cat_cols = X_train_fold.select_dtypes(exclude=np.number).columns.tolist()
        scaler = StandardScaler()
        X_train_num_scaled = pd.DataFrame(scaler.fit_transform(X_train_fold[num_cols]), columns=num_cols,
                                          index=X_train_fold.index)
        X_val_num_scaled = pd.DataFrame(scaler.transform(X_val_fold[num_cols]), columns=num_cols,
                                        index=X_val_fold.index)
        X_train_cat_encoded = X_train_fold[cat_cols]
        X_val_cat_encoded = X_val_fold[cat_cols]
        if cat_cols:
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            X_train_cat_encoded = pd.DataFrame(encoder.fit_transform(X_train_fold[cat_cols]),
                                               columns=encoder.get_feature_names_out(cat_cols),
                                               index=X_train_fold.index)
            X_val_cat_encoded = pd.DataFrame(encoder.transform(X_val_fold[cat_cols]),
                                             columns=encoder.get_feature_names_out(cat_cols), index=X_val_fold.index)
        X_train_processed = pd.concat([X_train_num_scaled, X_train_cat_encoded], axis=1)
        X_val_processed = pd.concat([X_val_num_scaled, X_val_cat_encoded], axis=1)
        feature_names_fold = X_train_processed.columns.tolist()
        ig_scores_fold = mutual_info_classif(X_train_processed, y_train_fold, random_state=42)
        sorted_features_fold = sorted(zip(feature_names_fold, ig_scores_fold), key=lambda x: x[1], reverse=True)

        best_f1, best_solution_indices, const_time, ls_time = construction(
            args, X_train_processed, y_train_fold, X_val_processed, y_val_fold,
            feature_names_fold, sorted_features_fold, fold
        )

        fold_f1_scores.append(best_f1)
        fold_construction_times.append(const_time)
        fold_ls_times.append(ls_time)

        if best_f1 > best_overall_f1:
            best_overall_f1 = best_f1
            best_overall_solution_info = {
                'fold': fold + 1, 'f1_score': best_f1,
                'features': [feature_names_fold[i] for i in best_solution_indices]
            }

    results = {
        "mean_f1_score": np.mean(fold_f1_scores),
        "std_f1_score": np.std(fold_f1_scores),
        "mean_construction_time": np.mean(fold_construction_times),
        "std_construction_time": np.std(fold_construction_times),
        "mean_ls_time": np.mean(fold_ls_times),
        "std_ls_time": np.std(fold_ls_times),
        "best_f1_overall": best_overall_f1,
        "best_solution_features": best_overall_solution_info.get('features')
    }

    logger.info("========== EXPERIMENT RESULTS ==========")
    for key, value in results.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")

    return results

if __name__ == '__main__':
    args = utils.parse_args()
    run_single_experiment(args)