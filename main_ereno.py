import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import time
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier

import utils

# Carregar os dados
X_train, y_train, X_test, y_test = utils.load_data()

# Aplicar preprocessamento
y_train, y_test, X_train, X_test, le = utils.preprocess_data(X_train, y_train, X_test, y_test)

# Reverter os rótulos numéricos para os nomes das classes
y_train_original = le.inverse_transform(y_train)
y_test_original = le.inverse_transform(y_test)

# Agora você pode adicionar estes rótulos ao DataFrame
train_df_combined = X_train.copy()
train_df_combined['class'] = y_train_original

test_df_combined = X_test.copy()
test_df_combined['class'] = y_test_original

# Salvar os DataFrames combinados em arquivos CSV
train_df_combined.to_csv('train_dataset.csv', index=False, encoding='utf-8')
test_df_combined.to_csv('test_dataset.csv', index=False, encoding='utf-8')

# Atualizar a lista de features após o preprocessamento
feature_names = X_train.columns.tolist()

# Gerar uma lista de todos os índices de características
all_features_indices = list(range(len(feature_names)))

# Calcular o Information Gain de cada feature
ig_scores = mutual_info_classif(X_train, y_train)

# Criar um dicionário para mapear nomes de features com seus respectivos IG scores
ig_scores_dict = dict(zip(feature_names, ig_scores))

# Ordenar as features por IG
sorted_features = sorted(ig_scores_dict.items(), key=lambda x: x[1], reverse=True)

# Imprimir os scores de IG
for feature, score in sorted_features:
    feature_id = feature_names.index(feature)
    print(f"Feature {feature} ({feature_id}):, IG: {score}")

def evaluate_with_decision_tree(features_idx):
    # Mapear índices para nomes das colunas
    features = [feature_names[i] for i in features_idx]

    # Criar e treinar o classificador
    dt = DecisionTreeClassifier()
    dt.fit(X_train[features], y_train)

    f1 = utils.evaluate_model(dt, X_train[features], y_train, X_test[features], y_test)

    # Retornar o F1-Score
    return f1

def evaluate_with_knn(features_idx):
    # Mapear índices para nomes das colunas
    features = [feature_names[i] for i in features_idx]

    # Criar e treinar o classificador
    knn = KNeighborsClassifier()
    knn.fit(X_train[features], y_train)

    f1 = utils.evaluate_model(knn, X_train[features], y_train, X_test[features], y_test)

    # Retornar o F1-Score
    return f1

def evaluate_with_naivebayes(features_idx):
    # Mapear índices para nomes das colunas
    features = [feature_names[i] for i in features_idx]

    # Criar e treinar o classificador
    nb = GaussianNB()
    nb.fit(X_train[features], y_train)
    f1 = utils.evaluate_model(nb, X_train[features], y_train, X_test[features], y_test)

    # Retornar o F1-Score
    return f1

def evaluate_with_svm(features_idx):
    # Mapear índices para nomes das colunas
    features = [feature_names[i] for i in features_idx]

    # Criar e treinar o classificador SVM
    svm = SVC()
    svm.fit(X_train[features], y_train)

    f1 = utils.evaluate_model(svm, X_train[features], y_train, X_test[features], y_test)

    # Retornar o F1-Score
    return f1

def evaluate_with_xgboost(features_idx):
    # Mapear índices para nomes das colunas
    features = [feature_names[i] for i in features_idx]

    # Criar e treinar o classificador XGBoost
    xgboost = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    xgboost.fit(X_train[features], y_train)

    f1 = utils.evaluate_model(xgboost, X_train[features], y_train, X_test[features], y_test)

    # Retornar o F1-Score
    return f1

# Avaliar usando todas as características
f1_score_dt = evaluate_with_decision_tree(all_features_indices)
f1_score_knn = evaluate_with_knn(all_features_indices)
f1_score_nb = evaluate_with_naivebayes(all_features_indices)
f1_score_svm = evaluate_with_svm(all_features_indices)
f1_score_xgboost = evaluate_with_xgboost(all_features_indices)

print("F1 Score Decision Tree usando todas as características:", f1_score_dt)
print("F1 Score KNN usando todas as características:", f1_score_knn)
print("F1 Score Naive Bayes usando todas as características:", f1_score_nb)
print("F1 Score SVM usando todas as características:", f1_score_svm)
print("F1 Score XGBoost usando todas as características:", f1_score_xgboost)
print(all_features_indices)


def local_search(initial_solution, repeated_solutions_count):
    max_f1_score = evaluate_with_naivebayes(initial_solution)
    best_solution = initial_solution.copy()
    seen_solutions = {frozenset(initial_solution)}

    for _ in range(100):
        new_solution = best_solution.copy()

        for replace_index in range(len(new_solution)):
            RCL = [feature_names.index(feature) for feature, score in sorted_features[:15] if feature_names.index(feature) not in new_solution]

            if not RCL:
                break

            new_feature = random.choice(RCL)
            new_solution[replace_index] = new_feature

        new_solution_set = frozenset(new_solution)
        if new_solution_set in seen_solutions:
            repeated_solutions_count += 1  # Incrementa o contador
            print(f"Mesma combinação de características encontrada: {new_solution_set}, gerando nova solução...")
            continue  # Ignora esta solução e continua a busca

        f1_score = evaluate_with_naivebayes(new_solution)

        if f1_score > max_f1_score and new_solution_set != frozenset(initial_solution):
            max_f1_score = f1_score
            best_solution = new_solution
            seen_solutions.add(new_solution_set)

    return max_f1_score, best_solution, repeated_solutions_count


def construction():
    # 'sorted_features' é uma lista de tuplas (feature, IG) ordenada pelo IG. estou pegando as top X para compor a RCL.
    RCL = [feature for feature, score in sorted_features[:15]]

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

    for iteration in range(100):
        # Garantir que a solução inicial seja única
        while True:
            # Selecionar aleatoriamente k features da RCL para gerar soluções iniciais
            selected_features = random.sample(RCL, k=5)
            # Converter os nomes das características em índices
            solution = [feature_names.index(feature_name) for feature_name in selected_features]
            solution_set = frozenset(selected_features)

            if solution_set not in seen_initial_solutions:
                seen_initial_solutions.add(solution_set)
                break
            else:
                repeated_solutions_count += 1  # Incrementa o contador
                print(f"Solução inicial repetida encontrada: {solution}, gerando nova solução...")

        f1_score = evaluate_with_naivebayes(solution)
        print(f"F1-Score: {f1_score} for solution: {solution}")
        all_solutions.append((iteration, f1_score, solution))

        if f1_score > 0.0:
            # Se a fila de prioridade não estiver cheia, simplesmente insere o novo F1-Score.
            if len(priority_queue.heap) < 10:
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
    plot_solutions_with_priority(all_solutions, priority_queue)

    start_time = time.perf_counter()  # Busca Local

    while not priority_queue.is_empty():
        _, current_solution = priority_queue.extract_max()
        original_f1_score = evaluate_with_naivebayes(current_solution)  # Avaliar a solução atual uma única vez
        improved_f1_score, improved_solution, repeated_solutions_count_local_search = local_search(current_solution, repeated_solutions_count_local_search)

        # Verifica se houve melhoria em relação ao F1-Score original da solução específica
        if improved_f1_score > original_f1_score:
            local_search_improvements[tuple(current_solution)] = improved_f1_score - original_f1_score
            # Imprime a melhoria específica para essa solução
            print(f"Melhoria na Solução Local! F1-Score: {improved_f1_score} para solução: {current_solution}. Nova solução: {improved_solution}")

        # Verifica se a solução melhorada é a melhor solução global
        if improved_f1_score > max_f1_score:
            max_f1_score = improved_f1_score
            best_solution = improved_solution
            print(f"Nova Melhor Solução Global! F1-Score: {max_f1_score} para solução: {best_solution}")

    total_local_search_time = time.perf_counter() - start_time  # Busca Local

    plot_solutions(all_solutions, priority_queue, local_search_improvements)

    print(f"Total de soluções repetidas na busca local: {repeated_solutions_count_local_search}")
    print("Tamanho Solução Inicial: ", selected_features)
    print("Tamanho RCL: ", len(RCL))


    print("Melhor F1-Score:", max_f1_score)
    print("Melhor conjunto de features:", best_solution)
    print(f"Tempo total de execução da Fase Construtiva: {total_elapsed_time} segundos")
    print(f"Tempo total de execução da Fase de Busca Local: {total_local_search_time} segundos")

class MaxPriorityQueue:
    def __init__(self):
        self.heap = []

    def insert(self, item):
        self.heap.append(item)
        self._sift_up(len(self.heap) - 1)

    def maximum(self):
        return self.heap[0] if self.heap else None

    def extract_max(self):
        if not self.heap:
            return None

        lastelt = self.heap.pop()
        if self.heap:
            minitem = self.heap[0]
            self.heap[0] = lastelt
            self._sift_down(0)
            return minitem
        return lastelt

    def _sift_up(self, pos):
        startpos = pos
        newitem = self.heap[pos]
        while pos > 0:
            parentpos = (pos - 1) >> 1
            parent = self.heap[parentpos]
            if newitem <= parent:
                break
            self.heap[pos] = parent
            pos = parentpos
        self.heap[pos] = newitem

    def _sift_down(self, pos):
        endpos = len(self.heap)
        startpos = pos
        newitem = self.heap[pos]
        childpos = 2 * pos + 1
        while childpos < endpos:
            rightpos = childpos + 1
            if rightpos < endpos and self.heap[childpos] <= self.heap[rightpos]:
                childpos = rightpos
            self.heap[pos] = self.heap[childpos]
            pos = childpos
            childpos = 2 * pos + 1
        self.heap[pos] = newitem

    def is_empty(self):
        return len(self.heap) == 0

def plot_solutions_with_priority(all_solutions, priority_queue):
    # Convertendo a fila de prioridade em um set para busca rápida
    priority_set = set([tuple(sol) for _, sol in priority_queue.heap])

    # Pegando índices de iteração e F1-Scores
    iterations = [iteration for _, iteration, _ in all_solutions]
    f1_scores = [f1 for f1, _, _ in all_solutions]

    # Verificando quais soluções estão no top 10
    priority_colors = ['red' if tuple(sol) in priority_set else 'blue' for _, _, sol in all_solutions]

    plt.scatter(f1_scores, iterations, color=priority_colors)
    plt.ylabel('F1-Score')
    plt.xlabel('Índice da Solução')
    plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label='Top 10', markersize=10, markerfacecolor='red'),
                       plt.Line2D([0], [0], marker='o', color='w', label='Outras Soluções', markersize=10, markerfacecolor='blue')],
               loc='lower right')
    plt.savefig("priority_plot.png")

def print_priority_queue(priority_queue):
    print("Fila de prioridade:")
    for score, solution in priority_queue.heap:
        print(f"F1-Score: {-score}, Solution: {solution}")  # Note o sinal negativo para converter de volta

def plot_solutions(all_solutions, priority_queue, local_search_improvements):
    # Configura o tamanho da figura
    plt.figure(figsize=(9, 4))

    # Convertendo a fila de prioridade em um set para busca rápida
    priority_set = set([tuple(sol) for _, sol in priority_queue.heap])

    # Pegando índices de iteração e F1-Scores
    iterations = [iteration for iteration, _, _ in all_solutions]
    f1_scores = [f1 for _, f1, _ in all_solutions]
    solutions = [sol for _, _, sol in all_solutions]

    # Desenha todas as barras azuis primeiro
    plt.bar(iterations, f1_scores, color='blue', label='Soluções Iniciais (SI)')

    # Desenha as barras das soluções da fila de prioridade em vermelho
    for i, sol in enumerate(solutions):
        if tuple(sol) in priority_set:
            plt.bar(iterations[i], f1_scores[i], color='red', label='Top 10')

    # Verifica se há melhorias e pinta a barra de vermelho (usado caso queira pintar de vermelho as top 10).
    for i, sol in enumerate(solutions):
        improvement = local_search_improvements.get(tuple(sol), 0)
        if improvement > 0:
            # Se houve melhoria, pinta a barra inteira de vermelho
            plt.bar(iterations[i], f1_scores[i] + improvement, color='red', label='SI Incluídas na Fila de Prioridades')

    # Sobrepinta a melhoria em verde onde aplicável
    for i, sol in enumerate(solutions):
        improvement = local_search_improvements.get(tuple(sol), 0)
        if improvement > 0:
            plt.bar(iterations[i], improvement, bottom=f1_scores[i], color='green', label='SI Melhoradas na Busca Local')

    # Adiciona legendas e rótulos
    plt.xlabel('Índice da Solução', fontsize=12, fontweight='bold')
    plt.ylabel('F1-Score', fontsize=12, fontweight='bold')
    # plt.xticks(iterations, rotation=90)
    # Cria legendas únicas
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # by_label = {label: handle for label, handle in by_label.items() if label in ['Soluções Iniciais', 'Top 10', 'Melhoradas na Busca Local']}
    plt.legend(by_label.values(), by_label.keys(), loc='lower right', prop={'weight': 'bold'})
    plt.xlim(min(iterations) - 1, max(iterations) + 1)
    plt.tick_params(axis='x', labelsize=12)  # Aumenta o tamanho da fonte das marcações do eixo x
    plt.tick_params(axis='y', labelsize=12)  # Aumenta o tamanho da fonte das marcações do eixo y
    plt.tight_layout()

    # Mostra o gráfico
    plt.savefig("all_bestsolution_.png")
    plt.savefig("all_bestsolution_.pdf")

if __name__ == '__main__':
    construction()