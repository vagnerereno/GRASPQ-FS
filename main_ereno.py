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
        raise ValueError("Algoritmo não suportado")

    return utils.evaluate_model(model, X_train[features], y_train, X_test[features], y_test)

def evaluate_baseline(feature_names, X_train, y_train, X_test, y_test):
    print("\nAvaliação Baseline com todos os algoritmos usando todas as features:")
    algorithms = ['knn', 'dt', 'nb', 'svm', 'rf', 'xgboost']
    baseline_results = {}
    for algo in algorithms:
        f1 = evaluate_algorithm(list(range(len(feature_names))), algo)
        baseline_results[algo] = f1
        print(f"F1-Score {algo.upper()} usando todas as características: {f1:.4f}")
    print("-" * 50)
    return baseline_results

def load_and_preprocess():
    X_train, y_train, X_test, y_test = utils.load_data()
    y_train, y_test, X_train, X_test, le = utils.preprocess_data(X_train, y_train, X_test, y_test)
    feature_names = X_train.columns.tolist()
    ig_scores = mutual_info_classif(X_train, y_train)
    sorted_features = sorted(zip(feature_names, ig_scores), key=lambda x: x[1], reverse=True)

    # Criar DataFrames combinados e salvar
    y_train_original = le.inverse_transform(y_train)
    y_test_original = le.inverse_transform(y_test)
    train_df_combined = X_train.copy()
    train_df_combined['class'] = y_train_original
    test_df_combined = X_test.copy()
    test_df_combined['class'] = y_test_original
    train_df_combined.to_csv('train_dataset.csv', index=False, encoding='utf-8')
    test_df_combined.to_csv('test_dataset.csv', index=False, encoding='utf-8')

    return X_train, y_train, X_test, y_test, feature_names, sorted_features, le

def print_feature_scores(sorted_features):
    print("\nScores de Information Gain para as features:")
    for feature, score in sorted_features:
        print(f"Feature {feature}: IG = {score:.4f}")

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
            print(f"Mesma combinação de características encontrada: {new_solution_set}, gerando nova solução...")
            continue  # Ignora esta solução e continua a busca

        f1_score = evaluate_algorithm(new_solution, algorithm)

        if f1_score > max_f1_score and new_solution_set != frozenset(best_solution):
            max_f1_score = f1_score
            best_solution = new_solution
            seen_solutions.add(new_solution_set)
        elif new_solution_set == frozenset(best_solution):
            print(f"Nenhuma melhoria real na solução: {new_solution}")

    return max_f1_score, best_solution, repeated_solutions_count

def construction(args):
    # 'sorted_features' é uma lista de tuplas (feature, IG) ordenada pelo IG. estou pegando as top X para compor a RCL.
    RCL = [feature for feature, _ in sorted_features[:args.rcl_size]]

    RCL_indices = [feature_names.index(feature) for feature in RCL]

    print(f"RCL Features: {RCL}")
    print(f"RCL Features Índices: {RCL_indices}")

    all_solutions = []
    local_search_improvements = {}  # Dicionário para armazenar os resultados da busca local

    priority_queue = MaxPriorityQueue()
    max_f1_score = -1
    best_solution = []

    seen_initial_solutions = set()
    repeated_solutions_count = 0  # Inicializa o contador de soluções repetidas
    repeated_solutions_count_local_search = 0  # Inicializa o contador de soluções repetidas da busca_local

    start_time = time.perf_counter()

    if args.rcl_size > len(feature_names):
        raise ValueError("O tamanho da RCL não pode ser maior que o número de features disponíveis.")
    if args.initial_solution > args.rcl_size:
        raise ValueError("O tamanho da solução inicial não pode ser maior que o tamanho da RCL.")

    for iteration in range(args.constructive_iterations):
        # Garantir que a solução inicial seja única
        while True:
            # Selecionar aleatoriamente k features da RCL para gerar soluções iniciais
            selected_features = random.sample(RCL, k=args.initial_solution)
            # Converter os nomes das características em índices
            solution = [feature_names.index(feature_name) for feature_name in selected_features]
            solution_set = frozenset(selected_features)

            if solution_set not in seen_initial_solutions:
                seen_initial_solutions.add(solution_set)
                break
            else:
                repeated_solutions_count += 1  # Incrementa o contador
                print(f"Solução inicial repetida encontrada: {solution}, gerando nova solução...")

        f1_score = evaluate_algorithm(solution, args.algorithm)
        print(f"F1-Score: {f1_score} for solution: {solution}")
        all_solutions.append((iteration, f1_score, solution))

        if f1_score > 0.0:
            # Se a fila de prioridade não estiver cheia, simplesmente insere o novo F1-Score.
            if len(priority_queue.heap) < args.priority_queue:
                priority_queue.insert((f1_score, solution))
            else:
                # Se a fila de prioridade estiver cheia, encontre o menor F1-Score na fila.
                lowest_f1 = min(priority_queue.heap, key=lambda x: x[0])[0]
                if f1_score > lowest_f1:
                    # Remove o item com o menor F1-Score antes de inserir o novo item.
                    priority_queue.heap.remove((lowest_f1, [item[1] for item in priority_queue.heap if item[0] == lowest_f1][0]))
                    priority_queue.insert((f1_score, solution))
        local_search_improvements[tuple(solution)] = 0
        # visualize_heap(priority_queue.heap)
    total_elapsed_time = time.perf_counter() - start_time
    print(f"Total de soluções iniciais repetidas: {repeated_solutions_count}")
    print(f"Tempo total de execução da Fase Construtiva: {total_elapsed_time} segundos")
    print_priority_queue(priority_queue)
    utils.plot_solutions_with_priority(all_solutions, priority_queue)

    start_time = time.perf_counter()  # Busca Local

    while not priority_queue.is_empty():
        _, current_solution = priority_queue.extract_max()

        original_f1_score = evaluate_algorithm(current_solution, args.algorithm)  # Avaliar a solução atual uma única vez
        improved_f1_score, improved_solution, repeated_solutions_count_local_search = local_search(
        current_solution, repeated_solutions_count_local_search, args.algorithm, args.rcl_size)

        # Verifica se houve melhoria em relação ao F1-Score original da solução específica
        if improved_f1_score > original_f1_score:
            local_search_improvements[tuple(current_solution)] = improved_f1_score - original_f1_score
            print(f"Melhoria na Solução Local! F1-Score: {improved_f1_score} para solução: {current_solution}. Nova solução: {improved_solution}")

        # Verifica se a solução melhorada é a melhor solução global
        if improved_f1_score > max_f1_score:
            max_f1_score = improved_f1_score
            best_solution = improved_solution
            print(f"Nova Melhor Solução Global! F1-Score: {max_f1_score} para solução: {best_solution}")

    total_local_search_time = time.perf_counter() - start_time  # Busca Local

    utils.plot_solutions(all_solutions, priority_queue, local_search_improvements)

    print(f"Total de soluções repetidas na busca local: {repeated_solutions_count_local_search}")
    print("Tamanho Solução Inicial: ", selected_features)
    print("Tamanho RCL: ", len(RCL))


    print("Melhor F1-Score:", max_f1_score)
    print("Melhor conjunto de features:", best_solution)
    print(f"Tempo total de execução da Fase Construtiva: {total_elapsed_time} segundos")
    print(f"Tempo total de execução da Fase de Busca Local: {total_local_search_time} segundos")

def print_priority_queue(priority_queue):
    print("Fila de prioridade:")
    for score, solution in priority_queue.heap:
        print(f"F1-Score: {-score}, Solution: {solution}")

if __name__ == '__main__':
    args = utils.parse_args()

    print("Parâmetros utilizados na execução:")
    print(f"  Algoritmo: {args.algorithm}")
    print(f"  Tamanho da RCL: {args.rcl_size}")
    print(f"  Tamanho da solução inicial: {args.initial_solution}")
    print(f"  Tamanho da fila de prioridade: {args.priority_queue}")
    print(f"  Iterações na fase de busca local: {args.local_iterations}")
    print(f"  Iterações na fase construtiva: {args.constructive_iterations}")
    print("-" * 50)

    # Carregar e preprocessar os dados
    X_train, y_train, X_test, y_test, feature_names, sorted_features, le = load_and_preprocess()

    # Imprimir os scores de IG
    print_feature_scores(sorted_features)

    # Avaliação inicial (baseline)
    baseline_results = evaluate_baseline(feature_names, X_train, y_train, X_test, y_test)

    # Continuar com o algoritmo selecionado para as próximas etapas
    print(f"Algoritmo selecionado para fases construtivas e busca local: {args.algorithm.upper()}")

    # Executar a construção e busca local
    construction(args)

