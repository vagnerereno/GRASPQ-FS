import datetime

from sklearn.tree import DecisionTreeClassifier
import random
import time
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.feature_selection import mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier
import utils
from priority_queue import MaxPriorityQueue
import csv
import logging
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
log_filename = f"log_{timestamp}.txt"
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Nível de log (INFO, DEBUG, ERROR, etc.)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
file_handler.setFormatter(formatter)
console_handler = logging.StreamHandler()  # Exibe no terminal
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

RCL_25_FEATURES = ['t', 'StNum', 'stDiff', 'timestampDiff', 'timeFromLastChange', 'sqDiff', 'SqNum', 'tDiff', 'delay',
                   'Time', 'GooseTimestamp', 'isbCRmsValue', 'vsbB', 'isbARmsValue', 'isbA', 'isbC', 'isbATrapAreaSum',
                   'isbCTrapAreaSum', 'vsbC', 'vsbARmsValue', 'vsbBRmsValue', 'isbBRmsValue', 'vsbA', 'isbBTrapAreaSum',
                   'vsbCRmsValue']
RCL_25_INDICES = [19, 22, 30, 36, 38, 31, 21, 37, 39, 0, 20, 9, 5, 7, 1, 3, 13, 15, 6, 10, 11, 8, 4, 14, 12]

def evaluate_algorithm(features_idx, algorithm):
    features = [feature_names[i] for i in features_idx]
    if algorithm == 'knn':
        model = KNeighborsClassifier()
    elif algorithm == 'dt':
        model = DecisionTreeClassifier()
    elif algorithm == 'nb':
        model = GaussianNB()
    elif algorithm == 'svm':
        model = SVC()
    elif algorithm == 'rf':
        model = RandomForestClassifier()
    elif algorithm == 'xgboost':
        model = xgb.XGBClassifier(eval_metric='mlogloss')
    else:
        raise ValueError("Unsupported algorithm")

    return utils.evaluate_model(model, X_train[features], y_train, X_test[features], y_test)

def evaluate_baseline(feature_names, X_train, y_train, X_test, y_test):
    logging.info("\nBaseline Evaluation with all algorithms using all features:")
    algorithms = ['knn', 'dt', 'nb', 'svm', 'rf', 'xgboost']
    baseline_results = {}
    for algo in algorithms:
        f1 = evaluate_algorithm(list(range(len(feature_names))), algo)
        baseline_results[algo] = f1
        logging.info(f"F1-Score {algo.upper()} using all features: {f1:.4f}")
    logging.info("-" * 50)
    return baseline_results

def log_to_csv(phase, initial_solution, initial_f1, improved_solution, improved_f1, best_f1, phase_time):
    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([phase, initial_solution, initial_f1, improved_solution, improved_f1, best_f1, phase_time])

def load_and_preprocess():
    X_train, y_train, X_test, y_test = utils.load_data()
    y_train, y_test, X_train, X_test, le = utils.preprocess_data(X_train, y_train, X_test, y_test)
    logging.info("Preprocessing completed successfully.")
    feature_names = X_train.columns.tolist()

    logging.info("Ranking Features using Mutual Information for composing RCL.")
    # Mutual Information (MI) measures the mutual dependence between two random variables.
    # In the context of feature selection, it evaluates how much information about the label
    # is provided by a particular feature.
    # ig_scores = mutual_info_classif(X_train, y_train, random_state=42)
    # logging.info("Feature ranking completed.")
    #
    # sorted_features = sorted(zip(feature_names, ig_scores), key=lambda x: x[1], reverse=True)
    logging.info("Using precomputed RCL features and indices.")
    sorted_features = [(feature, feature_names.index(feature)) for feature in RCL_25_FEATURES]

    return X_train, y_train, X_test, y_test, feature_names, sorted_features, le

def print_feature_scores(sorted_features):
    logging.info("\nMutua Information for Features:")
    for feature, score in sorted_features:
        logging.info(f"Feature {feature}: MI = {score:.4f}")

def local_search(initial_solution, repeated_solutions_count, algorithm, rcl_size):
    max_f1_score = evaluate_algorithm(initial_solution, algorithm)
    best_solution = initial_solution.copy()
    seen_solutions = {frozenset(initial_solution)}

    for _ in range(args.local_iterations):
        new_solution = best_solution.copy()

        for replace_index in range(len(new_solution)):
            RCL = [feature_names.index(feature) for feature, score in sorted_features[:rcl_size]
                   if feature_names.index(feature) not in new_solution]
            if not RCL:
                break

            new_feature = random.choice(RCL)
            new_solution[replace_index] = new_feature

        new_solution_set = frozenset(new_solution)
        if new_solution_set in seen_solutions:
            repeated_solutions_count += 1  # Incrementa o contador
            logging.info(f"Duplicate feature combination found: {new_solution_set}, generating a new solution...")
            continue  # Ignora esta solução e continua a busca

        f1_score = evaluate_algorithm(new_solution, algorithm)

        if f1_score > max_f1_score and new_solution_set != frozenset(best_solution):
            max_f1_score = f1_score
            best_solution = new_solution
            seen_solutions.add(new_solution_set)
        elif new_solution_set == frozenset(best_solution):
            logging.info(f"No real improvement in the solution: {new_solution}")

    return max_f1_score, best_solution, repeated_solutions_count

def construction(args):
    # 'sorted_features' is a list of tuples (feature, IG) sorted by IG. Picking the top X to compose the RCL.
    # RCL = [feature for feature, _ in sorted_features[:args.rcl_size]]
    # RCL_indices = [feature_names.index(feature) for feature in RCL]
    RCL = RCL_25_FEATURES
    RCL_indices = RCL_25_INDICES

    logging.info(f"RCL Features: {RCL}")
    logging.info(f"RCL Feature Indices: {RCL_indices}")

    all_solutions = []
    local_search_improvements = {}  # Dictionary to store results of local search

    priority_queue = MaxPriorityQueue()
    max_f1_score = -1
    best_solution = []

    seen_initial_solutions = set()
    repeated_solutions_count = 0  # Initialize the counter for repeated solutions
    repeated_solutions_count_local_search = 0  # Initialize the counter for repeated solutions during local search

    start_time = time.perf_counter()

    if args.rcl_size > len(feature_names):
        raise ValueError("The RCL size cannot exceed the number of available features.")
    if args.initial_solution > args.rcl_size:
        raise ValueError("The initial solution size cannot exceed the RCL size.")

    for iteration in range(args.constructive_iterations):
        # Ensure the initial solution is unique
        while True:
            # Randomly select k features from RCL to generate initial solutions
            selected_features = random.sample(RCL, k=args.initial_solution)
            # Convert feature names into indices
            solution = [feature_names.index(feature_name) for feature_name in selected_features]
            solution_set = frozenset(selected_features)

            if solution_set not in seen_initial_solutions:
                seen_initial_solutions.add(solution_set)
                break
            else:
                repeated_solutions_count += 1  # Incrementa o contador
                logging.info(f"Repeated initial solution found: {solution}, generating a new solution...")

        f1_score = evaluate_algorithm(solution, args.algorithm)
        logging.info(f"F1-Score: {f1_score} for solution: {solution}")
        all_solutions.append((iteration, f1_score, solution))

        if f1_score > 0.0:
            # If the priority queue is not full, simply insert the new F1-Score.
            if len(priority_queue.heap) < args.priority_queue:
                priority_queue.insert((f1_score, solution))
            else:
                # If the priority queue is full, find the lowest F1-Score in the queue.
                lowest_f1 = min(priority_queue.heap, key=lambda x: x[0])[0]
                if f1_score > lowest_f1:
                    # Remove the item with the lowest F1-Score before inserting the new item.
                    priority_queue.heap.remove((lowest_f1, [item[1] for item in priority_queue.heap if item[0] == lowest_f1][0]))
                    priority_queue.insert((f1_score, solution))
        local_search_improvements[tuple(solution)] = 0

        log_to_csv("Constructive Phase", solution, f1_score, None, None, max_f1_score, time.perf_counter() - start_time)
        # visualize_heap(priority_queue.heap)
    total_elapsed_time = time.perf_counter() - start_time
    logging.info(f"Total repeated initial solutions: {repeated_solutions_count}")
    logging.info(f"Total execution time for Constructive Phase: {total_elapsed_time} seconds")
    print_priority_queue(priority_queue)
    utils.plot_solutions_with_priority(all_solutions, priority_queue)

    start_time = time.perf_counter()  # Local Search Phase
    total_iterations = len(priority_queue.heap) * args.local_iterations  # Total predicted iterations
    current_iteration = 0

    while not priority_queue.is_empty():
        _, current_solution = priority_queue.extract_max()

        original_f1_score = evaluate_algorithm(current_solution, args.algorithm)  # Evaluate the current solution once
        improved_f1_score, improved_solution, repeated_solutions_count_local_search = local_search(
        current_solution, repeated_solutions_count_local_search, args.algorithm, args.rcl_size)

        log_to_csv("Local Search", current_solution, original_f1_score, improved_solution, improved_f1_score,
                   max_f1_score, time.perf_counter() - start_time)

        # Increment iteration count
        current_iteration += args.local_iterations

        # Progress log every 50 iterations
        elapsed_time = time.perf_counter() - start_time
        estimated_total_time = (elapsed_time / current_iteration) * total_iterations
        logging.info(
            f"[{current_iteration}/{total_iterations}] Best solution: F1-Score {improved_f1_score:.4f} |"
            f" Estimated remaining time: {estimated_total_time - elapsed_time:.2f}s")

        # Check if there was an improvement compared to the original F1-Score of the specific solution
        if improved_f1_score > original_f1_score:
            local_search_improvements[tuple(current_solution)] = improved_f1_score - original_f1_score
            logging.info(f"Improvement in Local Search! F1-Score: {improved_f1_score} for solution: {current_solution}. New solution: {improved_solution}")

        # Check if the improved solution is the global best solution
        if improved_f1_score > max_f1_score:
            max_f1_score = improved_f1_score
            best_solution = improved_solution
            logging.info(f"New Global Best Solution! F1-Score: {max_f1_score} for solution: {best_solution}")

    total_local_search_time = time.perf_counter() - start_time  # Busca Local

    utils.plot_solutions(all_solutions, priority_queue, local_search_improvements)

    logging.info(f"Total repeated solutions in local search: {repeated_solutions_count_local_search}")
    logging.info(f"Initial Solution Size: {selected_features}")
    logging.info(f"RCL Size: {len(RCL)}")
    logging.info(f"Best F1-Score: {max_f1_score}")
    logging.info(f"Best Feature Set (indices): {best_solution}")

    # Map indices to feature names
    best_feature_names = [(feature_names[i], i) for i in best_solution]
    formatted_best_features = ", ".join([f"'{name}' ({index})" for name, index in best_feature_names])

    logging.info(f"Best Feature Set (names): {formatted_best_features}")

    logging.info(f"Total execution time for Constructive Phase: {total_elapsed_time} seconds")
    logging.info(f"Total execution time for Local Search Phase: {total_local_search_time} seconds")
    log_to_csv("Summary Constructive Phase", None, None, None, None, max_f1_score, total_elapsed_time)
    log_to_csv("Summary Local Search", None, None, None, None, max_f1_score, total_local_search_time)

def print_priority_queue(priority_queue):
    logging.info("Priority Queue:")
    for score, solution in priority_queue.heap:
        logging.info(f"F1-Score: {-score}, Solution: {solution}")

if __name__ == '__main__':
    args = utils.parse_args()

    logging.info("Execution parameters:")
    logging.info(f"  Algorithm: {args.algorithm}")
    logging.info(f"  RCL Size: {args.rcl_size}")
    logging.info(f"  Initial Solution Size: {args.initial_solution}")
    logging.info(f"  Priority Queue Size: {args.priority_queue}")
    logging.info(f"  Local Search Iterations: {args.local_iterations}")
    logging.info(f"  Constructive Iterations: {args.constructive_iterations}")
    logging.info("-" * 50)

    log_file = f"graspq_results_rcl{args.rcl_size}_cc{args.constructive_iterations}_bs{args.priority_queue}_is{args.initial_solution}.csv"

    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["phase", "initial_solution", "initial_f1_score", "improved_f1_score", "phase_time"])

    # Load and preprocess the data
    X_train, y_train, X_test, y_test, feature_names, sorted_features, le = load_and_preprocess()

    # Print IG scores
    print_feature_scores(sorted_features)

    # Initial evaluation (baseline)
    # baseline_results = evaluate_baseline(feature_names, X_train, y_train, X_test, y_test)

    # Continue with the selected algorithm for the next steps
    logging.info(f"Selected algorithm for constructive and local search phases: {args.algorithm.upper()}")

    # Execute construction and local search
    construction(args)

