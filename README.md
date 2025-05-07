<h1 align="center">📌 Welcome to GRASPQ-FS! 📌</h1>

<h4 align="left">
✔️ GRASPQ-FS is a command-line tool that applies the GRASP metaheuristic with a priority queue to perform feature selection in Intrusion Detection Systems (IDS). It was designed to work with enriched datasets, focusing on performance and reproducibility.
</h4>

<h2>📁 Repository Structure</h2>
<pre><code>.
├── data/                     # Folder for training and test datasets
├── main.py                   # Main script to run the GRASPQ-FS algorithm
├── utils.py                  # Data loading, preprocessing, and evaluation helpers
├── priority_queue.py         # Custom max priority queue implementation
├── requirements.txt          # List of required Python packages
├── README.md                 # This documentation file
</code></pre>

<h2>📋 Index</h2>
<ol>
  <li>Test Environment</li>
  <li>Requirements</li>
  <li>Development Environment</li>
  <li>Usage Example</li>
  <li><a href="#portuguese">🇧🇷 Versão em português!</a></li>
</ol>

<h3>🖱️ Test Environment</h3>
<table border="1">
<tr><th>Configuration</th><th>Machine</th></tr>
<tr><td>Operating System</td><td>Windows 11</td></tr>
<tr><td>Processor</td><td>13th Gen Intel(R) Core(TM) i7-13650HX @ 2.60GHz</td></tr>
<tr><td>RAM</td><td>16 GB (15.7 GB usable)</td></tr>
<tr><td>Python Version</td><td>3.10.0</td></tr>
</table>

<h3>⚙️ Development Environment</h3>
<table border="1">
<tr><th>Tool</th><th>Version</th></tr>
<tr><td>Python</td><td>3.10.0</td></tr>
<tr><td>Editor</td><td>VS Code / PyCharm</td></tr>
<tr><td>Terminal</td><td>PowerShell or CMD</td></tr>
</table>

<h3>📝 Requirements</h3>
<p>This Python project uses the following libraries:</p>
<ul>
  <li>numpy ≥ 1.21</li>
  <li>pandas ≥ 1.3</li>
  <li>matplotlib ≥ 3.4</li>
  <li>scikit-learn ≥ 1.0</li>
  <li>xgboost ≥ 1.5</li>
</ul>

<h3>🚀 How to Run</h3>
<p>To get started with this project, follow these steps:</p>
<ol>
  <li>
    <strong>Clone this repository and enter the project folder:</strong>
    <pre><code>git clone https://github.com/this-repository.git
cd this-repository</code></pre>
  </li>


  <li>
    <strong>Create a virtual environment (recommended):</strong>
    <pre><code>python -m venv venv
venv\Scripts\activate   # on Windows
source venv/bin/activate  # on Unix/Mac</code></pre>
  </li>

  <li><strong>Install the dependencies:</strong></li>
  <pre><code>pip install -r requirements.txt</code></pre>

  <li><strong>Run the tool with the desired configuration:</strong></li>
  <pre><code>python main.py -a nb -rcl 10 -is 5 -pq 10 -lc 50 -cc 100</code></pre>
</ol>
<h3>👨‍💻 Other Usage Examples</h3>

<pre><code>
python main.py -a nb -rcl 10 -is 5 -pq 10 -lc 50 -cc 100
or
python main.py --algorithm nb --rcl_size 10 --init_sol 5 --pq_size 10 --ls 50 --const 100
or
python main.py --alg nb --rcl 10 --initial_solution 5 --priority-queue 10 --local_iterations 50 --constructive_iterations 100
</code></pre>

<p><strong>Parameters (with all aliases):</strong></p>
<ul>
  <li><code>-a</code>, <code>--algorithm</code>, <code>--alg</code>: Classifier (<code>nb</code>, <code>dt</code>, <code>knn</code>, <code>rf</code>, <code>svm</code>, <code>linear_svc</code>, <code>sgd</code>, <code>xgboost</code>)</li>
  <li><code>-rcl</code>, <code>--rcl_size</code>, <code>--rcl</code>: Restricted Candidate List size</li>
  <li><code>-is</code>, <code>--init_sol</code>, <code>--initial_solution</code>: Number of features in the initial solution</li>
  <li><code>-pq</code>, <code>--pq_size</code>, <code>--priority-queue</code>: Size of the priority queue</li>
  <li><code>-cc</code>, <code>--const</code>, <code>--constructive_iterations</code>: Number of constructive iterations</li>
  <li><code>-lc</code>, <code>--ls</code>, <code>--local_iterations</code>: Number of local search iterations per solution</li>
</ul>

<p>⚠️ Ensure that the datasets <code>hibrid_dataset_GOOSE_train.csv</code> and <code>hibrid_dataset_GOOSE_test.csv</code> are inside the <code>data/</code> folder.</p>

<hr>

<a name="portuguese"></a>
<h1 align="center">📌 Bem-vindo ao GRASPQ-FS! 📌</h1>

<h4 align="left">
✔️ O GRASPQ-FS é uma ferramenta de linha de comando que aplica a metaheurística GRASP com fila de prioridades para seleção de atributos em sistemas de detecção de intrusos (IDS). Foi projetada para trabalhar com datasets enriquecidos, com foco em desempenho e reprodutibilidade.
</h4>

<h2>📁 Estrutura do Repositório</h2>
<pre><code>.
├── data/                     # Pasta com os conjuntos de dados de treino e teste
├── main.py                   # Script principal para executar o algoritmo GRASPQ-FS
├── utils.py                  # Funções auxiliares para carregamento, pré-processamento e avaliação dos dados
├── priority_queue.py         # Implementação personalizada de fila de prioridade máxima
├── requirements.txt          # Lista de pacotes Python necessários
├── README.md                 # Este arquivo de documentação
</code></pre>

<h2>📋 Índice</h2>
<ol>
  <li>Ambiente de Teste</li>
  <li>Requisitos</li>
  <li>Ambiente de Desenvolvimento</li>
  <li>Exemplo de Uso</li>
  <li><a href="#english-version">English version</a></li>
</ol>

<h3>🖱️ Ambiente de Teste</h3>
<table border="1">
<tr><th>Configuração</th><th>Computador</th></tr>
<tr><td>Sistema Operacional</td><td>Windows 11</td></tr>
<tr><td>Processador</td><td>Intel(R) Core(TM) i7-13650HX @ 2.60GHz</td></tr>
<tr><td>Memória RAM</td><td>16 GB (15,7 GB utilizável)</td></tr>
<tr><td>Versão do Python</td><td>3.10.0</td></tr>
</table>

<h3>⚙️ Ambiente de Desenvolvimento</h3>
<table border="1">
<tr><th>Ferramenta</th><th>Versão</th></tr>
<tr><td>Python</td><td>3.10.0</td></tr>
<tr><td>Editor</td><td>VS Code / PyCharm</td></tr>
<tr><td>Terminal</td><td>PowerShell ou CMD</td></tr>
</table>

<h3>📝 Requisitos</h3>
<p>O projeto utiliza Python 3 e as seguintes bibliotecas:</p>
<ul>
  <li>numpy ≥ 1.21</li>
  <li>pandas ≥ 1.3</li>
  <li>matplotlib ≥ 3.4</li>
  <li>scikit-learn ≥ 1.0</li>
  <li>xgboost ≥ 1.5</li>
</ul>

<h3>🚀 Como Executar</h3>
<p>Para começar a usar este projeto, siga os passos abaixo:</p>
<ol>
  <li>
    <strong>Clone este repositório e entre na pasta do projeto:</strong>
    <pre><code>git clone https://github.com/this-repository.git
cd this-repository</code></pre>
  </li>

  <li>
    <strong>Crie um ambiente virtual (recomendado):</strong>
    <pre><code>python -m venv venv
venv\Scripts\activate   # no Windows
source venv/bin/activate  # no Unix/Mac</code></pre>
  </li>

  <li>
    <strong>Instale as dependências:</strong>
    <pre><code>pip install -r requirements.txt</code></pre>
  </li>

  <li>
    <strong>Execute a ferramenta com a configuração desejada:</strong>
    <pre><code>python main.py -a nb -rcl 10 -is 5 -pq 10 -lc 50 -cc 100</code></pre>
  </li>
</ol>


<h3>👨‍💻 Outros Exemplos de Uso</h3>

<pre><code>
python main.py -a nb -rcl 10 -is 5 -pq 10 -lc 50 -cc 100
ou
python main.py --algorithm nb --rcl_size 10 --init_sol 5 --pq_size 10 --ls 50 --const 100
ou
python main.py --alg nb --rcl 10 --initial_solution 5 --priority-queue 10 --local_iterations 50 --constructive_iterations 100
</code></pre>

<p><strong>Parâmetros (com todos os aliases):</strong></p>
<ul>
  <li><code>-a</code>, <code>--algorithm</code>, <code>--alg</code>: Classificador (<code>nb</code>, <code>dt</code>, <code>knn</code>, <code>rf</code>, <code>svm</code>, <code>linear_svc</code>, <code>sgd</code>, <code>xgboost</code>)</li>
  <li><code>-rcl</code>, <code>--rcl_size</code>, <code>--rcl</code>: Tamanho da Lista Restrita de Candidatos</li>
  <li><code>-is</code>, <code>--init_sol</code>, <code>--initial_solution</code>: Número de atributos na solução inicial</li>
  <li><code>-pq</code>, <code>--pq_size</code>, <code>--priority-queue</code>: Tamanho da fila de prioridade</li>
  <li><code>-cc</code>, <code>--const</code>, <code>--constructive_iterations</code>: Número de iterações da fase construtiva</li>
  <li><code>-lc</code>, <code>--ls</code>, <code>--local_iterations</code>: Número de iterações da busca local por solução</li>
</ul>

<p>⚠️ Certifique-se de que os arquivos <code>hibrid_dataset_GOOSE_train.csv</code> e <code>hibrid_dataset_GOOSE_test.csv</code> estejam na pasta <code>data/</code>.</p>
