<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>GRASPQ-FS README</title>
</head>
<body>

<h1 align="center">📌 Welcome to GRASPQ-FS! 📌</h1>

<h4 align="left">
✔️ GRASPQ-FS is a command-line tool that applies the GRASP metaheuristic with a priority queue to perform feature selection in Intrusion Detection Systems (IDS). It was designed to work with enriched datasets, focusing on performance and reproducibility.
</h4>

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

<h3>📝 Requirements</h3>
<p>This Python project uses scientific computing libraries:</p>
<ul>
  <li>numpy ≥ 1.21</li>
  <li>pandas ≥ 1.3</li>
  <li>matplotlib ≥ 3.4</li>
  <li>scikit-learn ≥ 1.0</li>
  <li>xgboost ≥ 1.5</li>
</ul>

<p>To install all dependencies:</p>
<pre><code>pip install -r requirements.txt</code></pre>

<h3>⚙️ Development Environment</h3>
<table border="1">
<tr><th>Tool</th><th>Version</th></tr>
<tr><td>Python</td><td>3.10.0</td></tr>
<tr><td>Editor</td><td>VS Code / PyCharm</td></tr>
<tr><td>Terminal</td><td>PowerShell or CMD</td></tr>
</table>

<h3>👨‍💻 Usage Example</h3>
<p>Run the tool from the terminal using:</p>
<pre><code>python main_ereno.py -a nb -rcl 10 -is 5 -pq 10 -lc 50 -cc 100</code></pre>

<p><strong>Parameters:</strong></p>
<ul>
  <li><code>-a</code>: Classifier (<code>nb</code>, <code>dt</code>, <code>rf</code>, <code>svm</code>, etc.)</li>
  <li><code>-rcl</code>: Size of the Restricted Candidate List</li>
  <li><code>-is</code>: Number of features in the initial solution</li>
  <li><code>-pq</code>: Priority Queue Size</li>
  <li><code>-lc</code>: Local Search Iterations</li>
  <li><code>-cc</code>: Constructive Iterations</li>
</ul>

<p>⚠️ Ensure that the datasets <code>hibrid_dataset_GOOSE_train.csv</code> and <code>hibrid_dataset_GOOSE_test.csv</code> are inside the <code>data/</code> folder.</p>

<hr>

<a name="portuguese"></a>
<h1 align="center">📌 Bem-vindo ao GRASPQ-FS! 📌</h1>

<h4 align="left">
✔️ O GRASPQ-FS é uma ferramenta de linha de comando que aplica a metaheurística GRASP com fila de prioridades para seleção de atributos em sistemas de detecção de intrusos (IDS). Foi projetada para trabalhar com datasets enriquecidos, com foco em desempenho e reprodutibilidade.
</h4>

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

<h3>📝 Requisitos</h3>
<p>O projeto utiliza Python 3 e bibliotecas científicas:</p>
<ul>
  <li>numpy ≥ 1.21</li>
  <li>pandas ≥ 1.3</li>
  <li>matplotlib ≥ 3.4</li>
  <li>scikit-learn ≥ 1.0</li>
  <li>xgboost ≥ 1.5</li>
</ul>

<p>Para instalar todas as dependências:</p>
<pre><code>pip install -r requirements.txt</code></pre>

<h3>⚙️ Ambiente de Desenvolvimento</h3>
<table border="1">
<tr><th>Ferramenta</th><th>Versão</th></tr>
<tr><td>Python</td><td>3.10.0</td></tr>
<tr><td>Editor</td><td>VS Code / PyCharm</td></tr>
<tr><td>Terminal</td><td>PowerShell ou CMD</td></tr>
</table>

<h3>👨‍💻 Exemplo de Uso</h3>
<p>Execute a ferramenta com o comando:</p>
<pre><code>python main_ereno.py -a nb -rcl 10 -is 5 -pq 10 -lc 50 -cc 100</code></pre>

<p><strong>Parâmetros:</strong></p>
<ul>
  <li><code>-a</code>: Classificador (<code>nb</code>, <code>dt</code>, <code>rf</code>, <code>svm</code>, etc.)</li>
  <li><code>-rcl</code>: Tamanho da Lista de Candidatos Restritos</li>
  <li><code>-is</code>: Tamanho da Solução Inicial</li>
  <li><code>-pq</code>: Tamanho da Fila de Prioridade</li>
  <li><code>-lc</code>: Iterações de Busca Local</li>
  <li><code>-cc</code>: Iterações Construtivas</li>
</ul>

<p>⚠️ Certifique-se de que os arquivos <code>hibrid_dataset_GOOSE_train.csv</code> e <code>hibrid_dataset_GOOSE_test.csv</code> estejam na pasta <code>data/</code>.</p>

</body>
</html>
