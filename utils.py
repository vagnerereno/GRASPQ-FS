import argparse
import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score)
import logging
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

def load_data():
    # Data loading
    train_df = pd.read_csv('data/hibrid_dataset_GOOSE_train.csv', sep=',')
    test_df = pd.read_csv('data/hibrid_dataset_GOOSE_test.csv', sep=',')
    logging.info(f"Original dataset (Train): \n{train_df.head().to_string()}")
    logging.info(f"Original dataset (Test): \n{test_df.head().to_string()}")
    logging.info(f"Unique classes in the test dataset: {test_df['class'].unique()}")
    logging.info(f"Unique classes in the training dataset: {train_df['class'].unique()}")

    # Remove specific attacks from the training set
    # train_df = train_df[train_df['class'] != 'random_replay']
    # train_df = train_df[train_df['class'] != 'inverse_replay']
    # train_df = train_df[train_df['class'] != 'masquerade_fake_fault']
    # train_df = train_df[train_df['class'] != 'masquerade_fake_normal']
    # train_df = train_df[train_df['class'] != 'injection']
    # train_df = train_df[train_df['class'] != 'high_StNum']
    # train_df = train_df[train_df['class'] != 'poisoned_high_rate']
    logging.info(f"Remaining unique classes in the training dataset: {train_df['class'].unique()}")
    logging.info(f"Size of the training dataset after filtering: {len(train_df)}")

    # Remove specific attacks from the test set
    # test_df = test_df[test_df['class'] != 'random_replay']
    # test_df = test_df[test_df['class'] != 'inverse_replay']
    # test_df = test_df[test_df['class'] != 'masquerade_fake_fault']
    # test_df = test_df[test_df['class'] != 'masquerade_fake_normal']
    # test_df = test_df[test_df['class'] != 'injection']
    # test_df = test_df[test_df['class'] != 'high_StNum']
    # test_df = test_df[test_df['class'] != 'poisoned_high_rate']
    logging.info(f"Remaining unique classes in the test dataset: {test_df['class'].unique()}")
    logging.info(f"Size of the test dataset after filtering: {len(test_df)}")

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Enriched columns from the ERENO dataset to be removed, if necessary
    columns_to_remove = [
        "isbARmsValue", "isbBRmsValue", "iisbCRmsValue", "ismARmsValue", "ismBRmsValue", "ismCRmsValue",
        "vsbARmsue", "vsbBRmsValue", "vsbCRmsValue", "vsmARmsValue", "vsmBRmsValue", "vsmCRmsValue",
        "isbATrapAreaSum", "isbBTrapAreaSum", "isbCTrapAreaSum", "ismATrapAreaSuValm", "ismBTrapAreaSum",
        "ismCTrapAreaSum", "vsbATrapAreaSum", "vsbBTrapAreaSum", "vsbCTrapAreaSum", "vsmATrapAreaSum",
        "vsmBTrapAreaSum", "vsmCTrapAreaSum", "stDiff", "sqDiff", "gooseLengthDiff", "cbStatusDiff",
        "apduSizeDiff", "frameLengthDiff", "timestampDiff", "tDiff", "timeFromLastChange", "delay"
    ]

    initial_features_train = train_df.shape[1] - 1  # Subtracting 1 to exclude the 'class' column
    initial_features_test = test_df.shape[1] - 1  # Subtracting 1 to exclude the 'class' column

    logging.info(f"Initial number of features in the training dataset: {initial_features_train}")
    logging.info(f"Initial number of features in the test dataset: {initial_features_test}")
    # Removing enriched and NaN columns
    train_df = train_df.drop(columns=columns_to_remove, errors='ignore')
    test_df = test_df.drop(columns=columns_to_remove, errors='ignore')

    remove_features_train = train_df.shape[1] - 1  # Subtracting 1 to exclude the 'class' column
    remove_features_test = test_df.shape[1] - 1  # Subtracting 1 to exclude the 'class' column
    logging.info(f"Number of features in the training dataset after removing enriched ones: {remove_features_train}")
    logging.info(f"Number of features in the test dataset after removing enriched ones:: {remove_features_test}")

    # Splitting features and labels
    X_train = train_df.drop(columns=['class'])
    y_train = train_df['class']
    X_test = test_df.drop(columns=['class'])
    y_test = test_df['class']

    return X_train, y_train, X_test, y_test

def preprocess_data(X_train, y_train, X_test, y_test):
    # Identify numerical columns
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    cat_cols = X_train.select_dtypes(include=['object']).columns

    # Use StandardScaler to normalize numerical data
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # Use OneHotEncoder for categorical columns
    if len(cat_cols) > 0:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        cat_encoded = encoder.fit_transform(X_train[cat_cols])
        cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(cat_cols))

        # Drop original categorical columns and concat encoded columns
        X_train = pd.concat([X_train[num_cols], cat_encoded_df], axis=1)
        cat_encoded_test = encoder.transform(X_test[cat_cols])
        cat_encoded_test_df = pd.DataFrame(cat_encoded_test, columns=encoder.get_feature_names_out(cat_cols))
        X_test = pd.concat([X_test[num_cols], cat_encoded_test_df], axis=1)

    # Initialize LabelEncoder for labels/classes
    le = LabelEncoder()

    le.fit(y_train)

    y_train = le.transform(y_train)
    y_test = le.transform(y_test) # Direct transformation of test labels

    return y_train, y_test, X_train, X_test, le

def calculate_metrics(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    VN = conf_matrix[0, 0]
    FN = conf_matrix[0, 1:].sum()
    FP = conf_matrix[1:, 0].sum() + (conf_matrix[:, 1:].sum(axis=0) - np.diag(conf_matrix)[1:]).sum()
    VP = np.diag(conf_matrix)[1:].sum()

    accuracy = (VP + VN) / (VP + VN + FP + FN)
    precision = VP / (VP + FP) if (VP + FP) > 0 else 0
    recall = VP / (VP + FN) if (VP + FN) > 0 else 0
    # logging.info(f"VN: {VN}, FN: {FN}, VP: {VP}, FP: {FP}")

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1, conf_matrix

def evaluate_model(classifier, X_train, y_train, X_test, y_test):
    # Train the classifier
    classifier.fit(X_train, y_train)

    # Predict labels for the test dataset
    y_pred = classifier.predict(X_test)

    # Calculate metrics
    accuracy, precision, recall, f1, conf_matrix = calculate_metrics(y_test, y_pred)

    # Print the confusion matrix and metrics
    # logging.info(conf_matrix)
    # logging.info(f"Accuracy: {accuracy * 100:.2f}%, Precision: {precision * 100:.2f}%, Recall: {recall * 100:.2f}%, F1-Score: {f1 * 100:.2f}%")

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
    plt.savefig(f"priority_plot_{timestamp}.png")


def plot_solutions(all_solutions, priority_queue, local_search_improvements):
    plt.figure(figsize=(9, 4))

    # Convert the priority queue into a set for fast lookup
    priority_set = set([tuple(sol) for _, sol in priority_queue.heap])

    # Extracting iteration indices and F1-Scores
    iterations = [iteration for iteration, _, _ in all_solutions]
    f1_scores = [f1 for _, f1, _ in all_solutions]
    solutions = [sol for _, _, sol in all_solutions]

    # Draw all blue bars first
    plt.bar(iterations, f1_scores, color='blue', label='Soluções Iniciais (SI)')

    # Draw bars for priority queue solutions in red
    for i, sol in enumerate(solutions):
        if tuple(sol) in priority_set:
            plt.bar(iterations[i], f1_scores[i], color='red', label='Top 10')

    # Check for improvements and paint the bar red
    for i, sol in enumerate(solutions):
        improvement = local_search_improvements.get(tuple(sol), 0)
        if improvement > 0:
            # Se houve melhoria, pinta a barra inteira de vermelho
            plt.bar(iterations[i], f1_scores[i] + improvement, color='red', label='SI Incluídas na Fila de Prioridades')

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
    plt.tick_params(axis='x', labelsize=12)  # Aumenta o tamanho da fonte das marcações do eixo x
    plt.tick_params(axis='y', labelsize=12)  # Aumenta o tamanho da fonte das marcações do eixo y
    plt.tight_layout()

    plt.savefig(f"all_bestsolution_{timestamp}.png")
    plt.savefig(f"all_bestsolution_{timestamp}.pdf")

# Função para parsear argumentos da linha de comando
def parse_args():
    parser = argparse.ArgumentParser(
        description="GRASPQ-FS for Feature Selection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-a", "--algorithm", "--alg",
        type=str, choices=['knn', 'dt', 'nb', 'svm', 'rf', 'xgboost'],
        default='knn',
        help="Algorithm to be used for evaluation (knn, dt, nb, rf, svm, xgboost)."
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

    return parser.parse_args()

