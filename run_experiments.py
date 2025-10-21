import pandas as pd
from argparse import Namespace
import time
import os
import logging
from datetime import datetime

from main import run_single_experiment


def setup_logger(log_filename):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    if logger.hasHandlers():
        logger.handlers.clear()

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
    parameter_to_vary = "priority_queue"
    # values_to_test = [10, 20, 30, 50]
    values_to_test = range(1,2)

    fixed_args = {
        "algorithm": "nb",
        "rcl_size": 17,
        "initial_solution": 5,
        # "priority_queue": 10,
        "local_iterations": 100,
        "constructive_iterations": 100,
        "k_folds": 5
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    algorithm_name = fixed_args.get('algorithm', 'unknown_alg')
    base_filename = f"{algorithm_name}_varying_{parameter_to_vary}_{timestamp}"

    if not os.path.exists('results'):
        os.makedirs('results')

    log_filename = f"results/run_log_{base_filename}.txt"
    output_filename = f"results/summary_{base_filename}.csv"

    logger = setup_logger(log_filename)

    start_time = time.time()

    all_results_list = []

    logger.info("--- Starting Automated Experiments Run ---")
    logger.info(f"Parameter to vary: '{parameter_to_vary}'")
    logger.info(f"Values to test: {values_to_test}")
    logger.info(f"Fixed parameters: {fixed_args}")

    for i, value in enumerate(values_to_test):
        logger.info("=" * 80)
        logger.info(f"EXECUTING EXPERIMENT {i + 1}/{len(values_to_test)}: '{parameter_to_vary}' = {value}")
        logger.info("=" * 80)

        current_args_dict = fixed_args.copy()
        current_args_dict[parameter_to_vary] = value
        args = Namespace(**current_args_dict)

        try:
            results_summary = run_single_experiment(args)

            experiment_row = {**vars(args), **results_summary}
            all_results_list.append(experiment_row)
        except Exception as e:
            logger.exception(f"!!! Experiment {i + 1} failed for {parameter_to_vary}={value} due to an error: {e}")
            break

    results_df = pd.DataFrame(all_results_list)

    rounding_map = {
        'mean_f1_score': 4,
        'std_f1_score': 4,
        'best_f1_overall': 4,
        'mean_construction_time': 2,
        'std_construction_time': 2,
        'mean_ls_time': 2,
        'std_ls_time': 2
    }

    for col, decimals in rounding_map.items():
        if col in results_df.columns:
            try:
                results_df[col] = results_df[col].round(decimals)
            except TypeError:
                logger.warning(f"Could not round column '{col}' as it may not be numeric.")

    results_df.to_csv(output_filename, index=False)

    total_time = time.time() - start_time

    logger.info("=" * 80)
    logger.info("--- Experiments Finished ---")
    logger.info(f"Total execution time: {total_time / 60:.2f} minutes.")
    logger.info(f"Aggregated results saved to '{output_filename}'")
    logger.info(f"Detailed logs for the entire run saved to '{log_filename}'")


if __name__ == '__main__':
    main()