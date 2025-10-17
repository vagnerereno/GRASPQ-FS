import argparse
import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score)
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

def load_unified_dataset(filepath='data/wustl-ehms-2020.csv'):
    """
    Loads a single, unified dataset from a CSV file.

    Args:
        filepath (str): The path to the unified CSV file.

    Returns:
        pd.DataFrame: Features (X).
        pd.Series: Target labels (y).
    """
    try:
        df = pd.read_csv(filepath, sep=',')
        logger.info(f"Unified dataset loaded successfully from {filepath}. Shape: {df.shape}")
    except FileNotFoundError:
        logger.error(f"Dataset file not found at {filepath}. Please create a unified dataset file.")
        raise

    X = df.drop(columns=['Attack Category', 'Label', 'Dir', 'Flgs', 'Packet_num', 'SrcAddr', 'DstAddr', 'Sport', 'Dport', 'SrcMac', 'DstMac'])
    y = df['Attack Category']

    return X, y

def calculate_metrics(y_true, y_pred):
    """
    Calculates the F1-score for the given true and predicted labels.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        float: The weighted F1-score.
    """
    return f1_score(y_true, y_pred, average='weighted')

def evaluate_model(classifier, X_train, y_train, X_test, y_test):
    """
    Trains a classifier and evaluates its F1-score on the test set.

    Args:
        classifier: The scikit-learn classifier instance.
        X_train: Training features.
        y_train: Training labels.
        X_test: Testing features.
        y_test: Testing labels.

    Returns:
        float: The calculated F1-score.
    """
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    f1 = calculate_metrics(y_test, y_pred)
    return f1

def plot_solutions_with_priority(all_solutions, priority_queue):
    # Convert the priority queue into a set for fast lookup
    priority_set = set([tuple(sol) for _, sol in priority_queue.heap])

    # Extracting iteration indices and F1-Scores
    iterations = [iteration for _, iteration, _ in all_solutions]
    f1_scores = [f1 for f1, _, _ in all_solutions]

    # Checking which solutions are in the top 10
    priority_colors = ['red' if tuple(sol) in priority_set else 'blue' for _, _, sol in all_solutions]

    plt.scatter(f1_scores, iterations, color=priority_colors)
    plt.ylabel('F1-Score')
    plt.xlabel('Índice da Solução')
    plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label='Top 10', markersize=10, markerfacecolor='red'),
                       plt.Line2D([0], [0], marker='o', color='w', label='Outras Soluções', markersize=10, markerfacecolor='blue')],
               loc='lower right')
    plt.savefig(f"priority_plot.png")


def plot_solutions(all_solutions, priority_queue_snapshot, local_search_improvements):
    plt.figure(figsize=(9, 4))

    # Convert the priority queue into a set for fast lookup
    priority_set = set([tuple(sol) for _, sol in priority_queue_snapshot])

    # Extracting iteration indices and F1-Scores
    iterations = [iteration for iteration, _, _ in all_solutions]
    f1_scores = [f1 for _, f1, _ in all_solutions]
    solutions = [sol for _, _, sol in all_solutions]

    # Draw all blue bars first
    plt.bar(iterations, f1_scores, color='blue', label='Soluções Iniciais (SI)')

    # Draw bars for priority queue solutions in red
    for i, sol in enumerate(solutions):
        if tuple(sol) in priority_set:
            plt.bar(iterations[i], f1_scores[i], color='red', label='SI Incluídas na Fila de Prioridades')

    # Overpaint improvements in green where applicable
    for i, sol in enumerate(solutions):
        improvement = local_search_improvements.get(tuple(sol), 0)
        if improvement > 0:
            plt.bar(iterations[i], improvement, bottom=f1_scores[i], color='green', label='SI Melhoradas na Busca Local')

    plt.xlabel('Índice da Solução', fontsize=12, fontweight='bold')
    plt.ylabel('F1-Score', fontsize=12, fontweight='bold')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # by_label = {label: handle for label, handle in by_label.items() if label in ['Soluções Iniciais', 'Top 10', 'Melhoradas na Busca Local']}
    plt.legend(by_label.values(), by_label.keys(), loc='lower right', prop={'weight': 'bold'})
    plt.xlim(min(iterations) - 1, max(iterations) + 1)
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)
    plt.tight_layout()

    plt.savefig(f"all_bestsolution.png")
    plt.savefig(f"all_bestsolution.pdf")

def parse_args():
    parser = argparse.ArgumentParser(
        description="GRASPQ-FS for Feature Selection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-a", "--algorithm", "--alg",
        type=str, choices=['knn', 'dt', 'nb', 'svm', 'rf', 'xgboost', 'linear_svc', 'sgd'],
        default='nb',
        help="Algorithm to be used for evaluation (knn, dt, nb, rf, svm, xgboost, linear_svc, sgd)."
    )
    parser.add_argument(
        "-rcl", "--rcl_size", "--rcl",
        type=int, default=10,
        help="Size of the Restricted Candidate List (RCL)."
    )
    parser.add_argument(
        "-is", "--initial_solution", "--init_sol",
        type=int, default=5,
        help="Size of the initial solution generated."
    )
    parser.add_argument(
        "-pq", "--priority_queue", "--pq_size",
        type=int, default=10,
        help="Maximum size of the priority queue."
    )
    parser.add_argument(
        "-lc", "--local_iterations", "--ls",
        type=int, default=50,
        help="Number of iterations in the local search phase."
    )
    parser.add_argument(
        "-cc", "--constructive_iterations", "--const",
        type=int, default=100,
        help="Number of iterations in the constructive phase."
    )
    parser.add_argument(
        "-k", "--k_folds",
        type=int, default=5,
        help="Number of folds for cross-validation."
    )

    return parser.parse_args()

