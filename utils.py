import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score)

def load_data():
    # Carregamento de dados
    # train_df = pd.read_csv('data/hibrid_dataset_GOOSE_train.csv', sep=',')
    # test_df = pd.read_csv('data/hibrid_dataset_GOOSE_test.csv', sep=',')
    train_df = pd.read_csv('data/train.csv', sep=',')
    test_df = pd.read_csv('data/test.csv', sep=',')
    print(train_df.head())
    print(test_df.head())
    print("Classes únicas no conjunto de teste:", test_df['class'].unique())
    print("Classes únicas no conjunto de treinamento:", train_df['class'].unique())
    # Remover o ataque específico do conjunto de treinamento
    # train_df = train_df[train_df['class'] != 'random_replay']
    # train_df = train_df[train_df['class'] != 'inverse_replay']
    # train_df = train_df[train_df['class'] != 'masquerade_fake_fault']
    # train_df = train_df[train_df['class'] != 'masquerade_fake_normal']
    # train_df = train_df[train_df['class'] != 'injection']
    # train_df = train_df[train_df['class'] != 'high_StNum']
    # train_df = train_df[train_df['class'] != 'poi soned_high_rate']
    # print("Classes únicas no conjunto de treinamento:", train_df['class'].unique())

    train_df = train_df.reset_index(drop=True)

    # Colunas enriquecidas do dataset ERENO para remover, caso necessário.
    columns_to_remove = ['stDiff', 'sqDiff', 'gooseLenghtDiff', 'cbStatusDiff', 'apduSizeDiff',
                         'frameLengthDiff', 'timestampDiff', 'tDiff', 'timeFromLastChange',
                         'delay', 'isbARms', 'isbBRms', 'isbCRms', 'ismARms', 'ismBRms', 'ismCRms',
                         'ismARmsValue', 'ismBRmsValue', 'ismCRmsValue', 'csbArms', 'csvBRms',
                         'csbCRms', 'vsmARms', 'vsmBRms', 'vsmCRms', 'isbARmsValue', 'isbBRmsValue',
                         'isbCRmsValue', 'vsbARmsValue', 'vsbBRmsValue', 'vsbCRmsValue',
                         'vsmARmsValue', 'vsmBRmsValue', 'vsmCRmsValue', 'isbATrapAreaSum',
                         'isbBTrapAreaSum', 'isbCTrapAreaSum', 'ismATrapAreaSum', 'ismBTrapAreaSum',
                         'ismCTrapAreaSum', 'csvATrapAreaSum', 'csvBTrapAreaSum', 'vsbATrapAreaSum',
                         'vsbBTrapAreaSum', 'vsbCTrapAreaSum', 'vsmATrapAreaSum', 'vsmBTrapAreaSum',
                         'vsmCTrapAreaSum', 'gooseLengthDiff']

    # Remoção de colunas enriquecidas e com NaN
    # train_df = train_df.dropna(axis=1)  # .drop(columns=columns_to_remove, errors='ignore')
    # test_df = test_df.dropna(axis=1)  # .drop(columns=columns_to_remove, errors='ignore')

    # Separação de features e labels
    X_train = train_df.drop(columns=['class'])
    y_train = train_df['class']
    X_test = test_df.drop(columns=['class'])
    y_test = test_df['class']
    return X_train, y_train, X_test, y_test

def preprocess_data(X_train, y_train, X_test, y_test):
    # Identificar colunas numéricas
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    cat_cols = X_train.select_dtypes(include=['object']).columns

    # Utilizar StandardScaler para normalizar os dados numéricos
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # Utilizar OneHotEncoder para colunas categóricas
    if len(cat_cols) > 0:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        cat_encoded = encoder.fit_transform(X_train[cat_cols])
        cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(cat_cols))

        # Drop original categorical columns and concat encoded columns
        X_train = pd.concat([X_train[num_cols], cat_encoded_df], axis=1)
        cat_encoded_test = encoder.transform(X_test[cat_cols])
        cat_encoded_test_df = pd.DataFrame(cat_encoded_test, columns=encoder.get_feature_names_out(cat_cols))
        X_test = pd.concat([X_test[num_cols], cat_encoded_test_df], axis=1)

    # Inicializar o LabelEncoder para os labels/classes
    le = LabelEncoder()

    le.fit(y_train)

    y_train = le.transform(y_train)

    # Transformar y_train e y_test para numérico
    if y_train.dtype == 'object' or isinstance(y_train, pd.Series):
        y_train = le.transform(y_train)
    if y_test.dtype == 'object' or isinstance(y_test, pd.Series):
        y_test = transform_test_labels(y_test, le)
    return y_train, y_test, X_train, X_test, le

def transform_test_labels(y_test, label_encoder):
    # Encontrar o maior valor numérico atual do LabelEncoder
    max_label_value = max(label_encoder.transform(label_encoder.classes_))

    # Substituir rótulos desconhecidos por um novo valor numérico
    y_test_transformed = []
    for label in y_test:
        if label in label_encoder.classes_:
            # Transformar rótulos conhecidos
            label_encoded = label_encoder.transform([label])[0]
            y_test_transformed.append(label_encoded)
        else:
            # Adicionar um novo valor numérico para rótulos desconhecidos
            y_test_transformed.append(max_label_value + 1)

    return np.array(y_test_transformed)

def calculate_metrics(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    VN = conf_matrix[0, 0]
    FN = conf_matrix[0, 1:].sum()
    FP = conf_matrix[1:, 0].sum() + (conf_matrix[:, 1:].sum(axis=0) - np.diag(conf_matrix)[1:]).sum()
    VP = np.diag(conf_matrix)[1:].sum()

    accuracy = (VP + VN) / (VP + VN + FP + FN)
    precision = VP / (VP + FP) if (VP + FP) > 0 else 0
    recall = VP / (VP + FN) if (VP + FN) > 0 else 0
    # print(f"VN: {VN}, FN: {FN}, VP: {VP}, FP: {FP}")

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1, conf_matrix

def evaluate_model(classifier, X_train, y_train, X_test, y_test):
    # Treinar o classificador
    classifier.fit(X_train, y_train)

    # Prever os rótulos para o conjunto de teste
    y_pred = classifier.predict(X_test)

    # Calcular métricas
    accuracy, precision, recall, f1, conf_matrix = calculate_metrics(y_test, y_pred)

    # Imprimir a matriz de confusão e as métricas
    # print(conf_matrix)
    # print(f"Accuracy: {accuracy * 100:.2f}%, Precision: {precision * 100:.2f}%, Recall: {recall * 100:.2f}%, F1-Score: {f1 * 100:.2f}%")

    return f1

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


# Função para parsear argumentos da linha de comando
def parse_args():
    parser = argparse.ArgumentParser(
        description="IDS com GRASP para seleção de features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Argumentos com nome completo e abreviação
    parser.add_argument(
        "-a", "--algorithm", "--alg",
        type=str, choices=['knn', 'dt', 'nb', 'svm', 'rf', 'xgboost'],
        default='knn',
        help="Algoritmo a ser usado para a avaliação (knn, dt, nb, rf, svm, xgboost)."
    )
    parser.add_argument(
        "-rcl", "--rcl_size", "--rcl",
        type=int, default=10,
        help="Tamanho da Restricted Candidate List (RCL)."
    )
    parser.add_argument(
        "-is", "--initial_solution", "--init_sol",
        type=int, default=5,
        help="Tamanho da solução inicial gerada."
    )
    parser.add_argument(
        "-pq", "--priority_queue", "--pq_size",
        type=int, default=10,
        help="Tamanho máximo da fila de prioridade."
    )
    parser.add_argument(
        "-lc", "--local_iterations", "--ls",
        type=int, default=50,
        help="Número de iterações na fase de busca local."
    )
    parser.add_argument(
        "-cc", "--constructive_iterations", "--const",
        type=int, default=100,
        help="Número de iterações na fase construtiva."
    )

    return parser.parse_args()

