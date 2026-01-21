import pandas as pd
from argparse import Namespace
import time
import os
import logging
from datetime import datetime
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
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


def main():
    parameter_to_vary = "initial_solution"
    values_step_1 = list(range(1, 11))
    values_step_10 = list(range(20, 101, 10))
    # values_to_test = values_step_1 + values_step_10
    values_to_test = [5, 10, 20]
    # values_to_test = range(1, 47) # Para testes rápidos

    fixed_args = {
        "dataset": "cic-iot",
        "algorithm": "knn",
        "ranker": "mi",
        "rcl_size": 47,
        "initial_solution": values_to_test,
        "priority_queue": 10,
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
    all_results_list = []

    logger.info("--- Starting Automated Experiments Run ---")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Algorithm: {algorithm_name}")
    logger.info(f"Parameter to vary: '{parameter_to_vary}'")
    logger.info(f"Values to test: {list(values_to_test)}")
    logger.info(
        f"Other fixed parameters: { {k: v for k, v in fixed_args.items() if k not in ['dataset', 'algorithm']} }")
    logger.info(f"Detailed logs will be saved to: {log_filename}")
    logger.info(f"Summary results will be saved to: {output_filename}")

    logger.info("--- Loading and Cleaning Data (Once) ---")
    try:
        X_cleaned, y_cleaned, feature_names_cleaned = utils.load_unified_dataset(dataset_name=dataset_name)
        if y_cleaned.dtype == 'object':
            y_cleaned_encoded = LabelEncoder().fit_transform(y_cleaned)
        else:
            y_cleaned_encoded = y_cleaned.astype(int)
    except Exception as e:
        logger.exception(f"CRITICAL ERROR: Failed to load data. Aborting experiment run. Error: {e}")
        return

    rounding_map = {
        'grasp_mean_f1_score': 4, 'grasp_std_f1_score': 4,
        'grasp_best_f1_overall': 4,
        'grasp_mean_construction_time': 2, 'grasp_std_construction_time': 2,
        'grasp_mean_ls_time': 2, 'grasp_std_ls_time': 2
    }

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

            experiment_row = {**vars(args), **grasp_summary}
            all_results_list.append(experiment_row)

            logger.info(f"Saving intermediate results for experiment {i + 1}...")
            results_df_intermediate = pd.DataFrame(all_results_list)

            for col, decimals in rounding_map.items():
                if col in results_df_intermediate.columns:
                    try:
                        results_df_intermediate[col] = results_df_intermediate[col].round(decimals)
                    except TypeError:
                        logger.warning(f"Could not round column '{col}'.")

            results_df_intermediate.to_csv(output_filename, index=False)
            logger.info(f"Intermediate results successfully saved to '{output_filename}'")

        except Exception as e:
            logger.exception(
                f"!!! GRASP Experiment {i + 1} failed for {parameter_to_vary}={value} due to an error: {e}")
            break

    if not all_results_list:
        logger.warning("No GRASP experiments completed successfully. CSV summary file was not generated.")

    total_time = time.time() - start_time
    logger.info("=" * 80)
    logger.info("--- Experiment Run Finished ---")
    logger.info(f"Total execution time: {total_time / 60:.2f} minutes.")
    logger.info(f"Final aggregated results saved to '{output_filename}'")
    logger.info(f"Detailed logs saved to: {log_filename}")


if __name__ == '__main__':
    main()