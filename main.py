"""AG2: treinamento, avaliação e classificação de canais de vendas."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

BASE = Path(__file__).resolve().parent
FEATURES = ['Region', 'Fresh', 'Milk', 'Grocery', 'Frozen',
            'Detergents Paper', 'Delicatessen']
CHANNELS = {'HoReCa': 0, 'Retail': 1}
REGIONS = {'Lisbon': 0, 'Oporto': 1, 'Other': 2}
LABELS = ['HoReCa', 'Retail']


def carregar_dados(caminho):
    """Lê o CSV fornecido, valida, converte e reordena as oito colunas."""
    df = pd.read_csv(caminho)
    # O CSV do professor usa underscore; o enunciado usa espaço.
    df = df.rename(columns={'Detergents_Paper': 'Detergents Paper'})
    if set(df.columns) != set(FEATURES + ['Channel']) or len(df.columns) != 8:
        raise ValueError('O CSV deve conter exatamente os sete atributos e Channel.')
    if df.empty or df.isna().any().any():
        raise ValueError('O CSV está vazio ou contém valores ausentes.')
    for coluna, mapa in [('Channel', CHANNELS), ('Region', REGIONS)]:
        if not df[coluna].isin(mapa).all():
            raise ValueError(f'Valores desconhecidos em {coluna}: use {list(mapa)}.')
        df[coluna] = df[coluna].replace(mapa).astype('int64')
    for coluna in FEATURES[1:]:
        valores = pd.to_numeric(df[coluna], errors='raise')
        if not (np.isfinite(valores) & (valores >= 0) & (valores % 1 == 0)
                & (valores < 2**63)).all():
            raise ValueError(f'{coluna} deve conter inteiros não negativos em int64.')
        df[coluna] = valores.astype('int64')
    return df.reindex(columns=FEATURES + ['Channel'])


def criar_entrada(valores):
    if len(valores) != 7:
        raise ValueError('Informe sete inteiros: região e seis gastos anuais.')
    if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in valores):
        raise ValueError('Todos os valores devem ser inteiros.')
    if valores[0] not in (0, 1, 2):
        raise ValueError('Região: 0 = Lisbon, 1 = Oporto, 2 = Other.')
    if any(v < 0 or v >= 2**63 for v in valores[1:]):
        raise ValueError('Gastos devem ser inteiros não negativos em int64.')
    return pd.DataFrame([valores], columns=FEATURES, dtype='int64')


def treinar(caminho, saida):
    df = carregar_dados(caminho)
    X, y = df[FEATURES], df['Channel']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    # Hiperparâmetros escolhidos por validação cruzada só no treino (5 folds,
    # métrica f1_macro). O teste nunca participa da busca, só da avaliação final.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    busca = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        {'max_depth': [3, 4, 5, 6, 8, None], 'min_samples_leaf': [1, 2, 5, 10, 15]},
        cv=cv, scoring='f1_macro')
    busca.fit(X_train, y_train)
    modelo = busca.best_estimator_
    pred = modelo.predict(X_test)
    relatorio = classification_report(y_test, pred, labels=[0, 1],
                                     target_names=LABELS, digits=4, zero_division=0)
    matriz = confusion_matrix(y_test, pred, labels=[0, 1])
    resumo = {'amostras': len(df), 'treino': len(X_train), 'teste': len(X_test),
              'random_state': 42, 'acuracia': accuracy_score(y_test, pred),
              'baseline_majoritaria': float((y_test == y_train.mode()[0]).mean()),
              'ordem_classes': LABELS, 'matriz_confusao': matriz.tolist(),
              'parametros': modelo.get_params(),
              'cv_melhores_parametros': busca.best_params_,
              'cv_f1_macro': busca.best_score_,
              'classification_report': classification_report(
                  y_test, pred, labels=[0, 1], target_names=LABELS,
                  output_dict=True, zero_division=0)}
    texto = (f'Amostras: {len(df)} | Treino: {len(X_train)} | Teste: {len(X_test)}\n'
             f'Melhores hiperparâmetros (validação cruzada no treino, 5 folds): '
             f'{resumo["cv_melhores_parametros"]} (f1_macro={resumo["cv_f1_macro"]:.4f})\n'
             f'Acurácia: {resumo["acuracia"]:.4%}\n'
             f'Baseline (classe majoritária): {resumo["baseline_majoritaria"]:.4%}\n\n'
             + relatorio + '\nMatriz de confusão (linhas=reais; colunas=previstas):\n'
             + 'Ordem: HoReCa, Retail\n' + str(matriz) + '\n')
    saida.mkdir(parents=True, exist_ok=True)
    (saida / 'avaliacao.txt').write_text(texto, encoding='utf-8')
    (saida / 'metricas.json').write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding='utf-8')
    (saida / 'arvore.txt').write_text(export_text(modelo, feature_names=FEATURES), encoding='utf-8')
    df.to_csv(saida / 'dados_processados.csv', index=False)
    resultados = X_test.copy()
    resultados['real'] = y_test
    resultados['previsto'] = pred
    resultados.to_csv(saida / 'predicoes_teste.csv', index_label='indice_original')
    (saida / 'divisao.json').write_text(json.dumps({
        'treino': X_train.index.tolist(), 'teste': X_test.index.tolist()}, indent=2), encoding='utf-8')
    joblib.dump(modelo, saida / 'modelo.joblib')
    print(texto)
    return modelo


def comparar_modelos(caminho, saida):
    """Treina os quatro algoritmos permitidos pelo enunciado na mesma divisão
    treino/teste e compara o desempenho, só para justificar a escolha da
    Decision Tree. kNN, MLP e Naive Bayes recebem um StandardScaler, já que,
    ao contrário da árvore, são sensíveis à escala dos atributos."""
    df = carregar_dados(caminho)
    X, y = df[FEATURES], df['Channel']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    candidatos = {
        'Decision Tree': DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=42),
        'k-Nearest Neighbors': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        'Multilayer Perceptron': make_pipeline(
            StandardScaler(), MLPClassifier(max_iter=2000, random_state=42)),
        'Naive Bayes': make_pipeline(StandardScaler(), GaussianNB()),
    }
    linhas = [f'{"Modelo":<22}{"Acurácia":>10}{"F1 macro":>10}']
    for nome, modelo in candidatos.items():
        modelo.fit(X_train, y_train)
        pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average='macro')
        linhas.append(f'{nome:<22}{acc:>10.4%}{f1:>10.4f}')
    texto = ('Comparação entre os modelos permitidos pelo enunciado\n'
              'Mesma divisão treino/teste (random_state=42, stratify=Channel).\n'
              'kNN, MLP e Naive Bayes usam StandardScaler; a árvore não precisa.\n\n'
              + '\n'.join(linhas) + '\n')
    saida.mkdir(parents=True, exist_ok=True)
    (saida / 'comparacao.txt').write_text(texto, encoding='utf-8')
    print(texto)


def interagir(modelo):
    print('Classificar novo cliente. Região: 0=Lisbon, 1=Oporto, 2=Other.')
    print('Gastos anuais em unidades monetárias, inteiros não negativos.')
    while True:
        try:
            valores = [int(input(f'{coluna}: ')) for coluna in FEATURES]
            entrada = criar_entrada(valores)
        except ValueError as erro:
            print(f'Entrada inválida: {erro} Tente novamente.\n')
            continue
        except (EOFError, KeyboardInterrupt):
            print('\nEncerrado.')
            break
        print(f'Canal previsto: {LABELS[int(modelo.predict(entrada)[0])]}')
        try:
            if input('Classificar outro cliente? [s/N]: ').strip().lower() != 's':
                break
        except (EOFError, KeyboardInterrupt):
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dados', type=Path, default=BASE / 'dados' / 'wholesale.csv')
    parser.add_argument('--saida', type=Path, default=BASE / 'resultados')
    parser.add_argument('--sem-interacao', action='store_true', help='Somente treinar e avaliar.')
    parser.add_argument('--prever', type=int, nargs=7, metavar='VALOR',
                        help='Região, Fresh, Milk, Grocery, Frozen, Detergents Paper, Delicatessen.')
    parser.add_argument('--comparar', action='store_true',
                        help='Compara Decision Tree, kNN, MLP e Naive Bayes na mesma divisão.')
    args = parser.parse_args()
    try:
        entrada = criar_entrada(args.prever) if args.prever is not None else None
        modelo = treinar(args.dados, args.saida)
        if args.comparar:
            comparar_modelos(args.dados, args.saida)
        if entrada is not None:
            print(f'Canal previsto: {LABELS[int(modelo.predict(entrada)[0])]}')
        elif not args.sem_interacao:
            interagir(modelo)
    except (ValueError, OSError) as erro:
        parser.exit(1, f'Erro: {erro}\n')


if __name__ == '__main__':
    main()
