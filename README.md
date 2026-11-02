# AG2 - Classificação de canais de vendas

Projeto da atividade de Engenharias de Computação e Software do Inatel.
Classifica clientes em **HoReCa** (hotel, restaurante ou café) ou **Retail** (varejo), usando uma árvore de decisão e sete atributos do conjunto Wholesale customers.

**Integrantes:** Álvaro Careli, Iury Teixeira

## Executar

Requer Python 3.11 ou superior. Validado com Python 3.14, pandas 3.0.3 e scikit-learn 1.8.0.
Clone o repositório (ou extraia o ZIP) e abra o terminal na pasta que contém `main.py`.

```powershell
git clone https://github.com/alvarosamp/Ag02.git
cd Ag02
```

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe main.py
```

No Linux/macOS, use `.venv/bin/python` nos dois últimos comandos.
O programa treina, mostra as métricas e solicita sete inteiros para classificar um novo cliente. Não use separadores de milhares: escreva `10000`.
Ao final, digite `s` para outra previsão; qualquer outra resposta encerra. Entradas inválidas podem ser corrigidas. Ctrl+C encerra a entrada interativa.

Somente treinamento e avaliação:

```powershell
.venv\Scripts\python.exe main.py --sem-interacao
```

Comparar os quatro algoritmos permitidos pelo enunciado na mesma divisão treino/teste:

```powershell
.venv\Scripts\python.exe main.py --sem-interacao --comparar
```

Previsão pela linha de comando (valores ilustrativos, sem classe verdadeira conhecida):

```powershell
.venv\Scripts\python.exe main.py --prever 2 10000 5000 8000 1000 3000 1000
```

Ordem: Region, Fresh, Milk, Grocery, Frozen, Detergents Paper, Delicatessen.
Regiões: 0 = Lisbon, 1 = Oporto, 2 = Other. Os seis gastos são anuais em unidades monetárias.
O resultado é uma estimativa do modelo, não uma confirmação do canal real.

## Correspondência com as nove etapas

1. **Download:** `dados/wholesale.csv` é uma cópia do CSV pré-processado vinculado pelo professor no PDF, baixada em 31/10/2026. O projeto funciona sem rede depois da instalação das bibliotecas.
2. **Leitura:** `pd.read_csv` em `carregar_dados`.
3. **Conversão:** `Series.replace` e `astype('int64')`: HoReCa → 0; Retail → 1; Lisbon → 0; Oporto → 1; Other → 2. Gastos também são validados e convertidos para int64.
4. **Ordenação:** `reindex` coloca Region e os seis gastos antes de Channel. O nome original `Detergents_Paper` é normalizado para `Detergents Paper`, conforme o enunciado.
5. **Divisão:** `train_test_split(test_size=0.20, random_state=42, stratify=y)`: 352 amostras de treino e 88 de teste. Estratificação mantém aproximadamente a proporção das classes. Channel é retirado dos atributos.
6. **Modelo:** `DecisionTreeClassifier`, com `max_depth` e `min_samples_leaf` escolhidos por `GridSearchCV` (validação cruzada estratificada, 5 folds, métrica f1_macro) usando somente o conjunto de treino.
7. **Treinamento e previsão:** `fit` somente no treino; `predict` no teste.
8. **Avaliação:** acurácia, `classification_report`, matriz de confusão e comparação com classe majoritária.
9. **Novas entradas:** `input` com validação e `predict`, exibindo HoReCa ou Retail. Alternativamente, use `--prever`.

## Justificativa e limites

A árvore de decisão é um dos modelos permitidos e permite inspecionar regras em `resultados/arvore.txt`. Divide os exemplos por limiares dos atributos, escolhidos pelo critério Gini. `max_depth` e `min_samples_leaf` são escolhidos por validação cruzada só no conjunto de treino (`GridSearchCV`, 5 folds, f1_macro); o teste nunca participa da busca, só da avaliação final. Árvores não exigem padronização dos gastos.

A escolha da Decision Tree entre os quatro modelos permitidos foi comparada empiricamente: veja [Comparação entre modelos](#comparação-entre-modelos) abaixo.

A codificação ordinal de Region segue a exigência da atividade, embora a região seja nominal. O conjunto é pequeno e desbalanceado (298 HoReCa e 142 Retail); por isso são apresentadas métricas por classe e baseline. A avaliação usa uma única divisão fixa, não validação cruzada. O desempenho em clientes de outra população ou período não foi verificado. Não se deve escolher parâmetros por repetidas consultas ao conjunto de teste.

## Resultados reproduzidos

Hiperparâmetros escolhidos por validação cruzada no treino: `max_depth=3`, `min_samples_leaf=10` (f1_macro de validação: 0,8782).

| Classe | Precisão | Recall | F1 | Amostras de teste |
|---|---:|---:|---:|---:|
| HoReCa | 0,9655 | 0,9333 | 0,9492 | 60 |
| Retail | 0,8667 | 0,9286 | 0,8966 | 28 |

Acurácia: **93,18% (82/88)**. Baseline: **68,18% (60/88)**.
F1 macro: **0,9229**.

| Real / previsto | HoReCa | Retail |
|---|---:|---:|
| HoReCa | 56 | 4 |
| Retail | 2 | 26 |

## Comparação entre modelos

Com `--comparar`, os quatro algoritmos permitidos pelo enunciado são treinados na mesma divisão treino/teste (`dados/wholesale.csv`, `random_state=42`, estratificada). kNN, MLP e Naive Bayes recebem um `StandardScaler`, já que são sensíveis à escala dos atributos (os gastos vão de dezenas a dezenas de milhares); a árvore não precisa, pois decide por limiares em cada atributo isoladamente.

| Modelo | Acurácia | F1 macro |
|---|---:|---:|
| Decision Tree | 93,18% | 0,9214 |
| k-Nearest Neighbors | 90,91% | 0,8931 |
| Multilayer Perceptron | 89,77% | 0,8853 |
| Naive Bayes | 90,91% | 0,8952 |

A árvore venceu nesta divisão, por isso foi o modelo escolhido para o restante do projeto. Resultado salvo em `resultados/comparacao.txt`.

## Arquivos gerados

- `resultados/avaliacao.txt`: relatório legível das métricas.
- `resultados/metricas.json`: métricas e parâmetros em formato estruturado.
- `resultados/dados_processados.csv`: oito colunas reordenadas, todas int64 ao carregar.
- `resultados/predicoes_teste.csv`: atributos, rótulos reais, previsões e índices originais das 88 amostras.
- `resultados/divisao.json`: índices que permitem auditar a separação treino/teste.
- `resultados/arvore.txt`: regras da árvore em texto.
- `resultados/comparacao.txt`: comparação entre os quatro modelos permitidos, gerado com `--comparar`.
- `resultados/modelo.joblib`: modelo treinado persistido. Deve ser carregado com versões compatíveis das dependências. Carregue somente arquivos joblib de origem confiável.

Cada execução treina de novo e atualiza os resultados. Caminhos alternativos podem ser informados com `--dados` e `--saida`.

## Testar em Docker

Para testar o modelo isolado do ambiente local, via API HTTP:

```bash
docker build -t ag2-classificador .
docker run -p 8000:8000 ag2-classificador
```

O `Dockerfile` treina o modelo no momento do build (`python main.py --sem-interacao`) e sobe uma API com `app.py` (FastAPI). Endpoints:

- `GET /health`: confirma que o modelo foi carregado.
- `POST /predict`: recebe os sete atributos em JSON e devolve o canal previsto e as probabilidades.

```bash
curl -s -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"Region":2,"Fresh":10000,"Milk":5000,"Grocery":8000,"Frozen":1000,"Detergents Paper":3000,"Delicatessen":1000}'
```

A API é só para teste manual do modelo treinado; não é a entrega exigida pelo enunciado, que pede apenas o script interativo.

Para usar o modelo salvo em Python:

```python
import joblib
from main import BASE, LABELS, criar_entrada

modelo = joblib.load(BASE / 'resultados' / 'modelo.joblib')
entrada = criar_entrada([2, 10000, 5000, 8000, 1000, 3000, 1000])
print(LABELS[int(modelo.predict(entrada)[0])])
```

Verificações automatizadas:

```powershell
.venv\Scripts\python.exe -m unittest discover -s testes -v
```

## Apresentação e entrega

Consulte `ROTEIRO_VIDEO.md` para uma apresentação de aproximadamente 6min30s.
O vídeo ainda precisa ser gravado pelos alunos. Incluam os nomes dos dois integrantes na identificação da entrega.
O PDF indica entrega até **02/11/2026**, com ZIP ou repositório e vídeo. O envio e as permissões de acesso devem ser conferidos pelos integrantes nas orientações do professor. Este projeto não realiza publicação nem envio.

## Fontes e atribuição

- CSV fornecido na atividade: https://raw.githubusercontent.com/marcelovca90-inatel/AG2/main/wholesale.csv
- Cardoso, M. Wholesale customers. UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C5030X
- Dataset original: https://archive.ics.uci.edu/dataset/292/wholesale+customers
- Licença dos dados originais: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/ . A versão usada é a adaptação categórica disponibilizada pelo professor; o projeto normaliza o cabeçalho e converte as categorias.
- API da árvore: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
- Divisão: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
- Métricas: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
