import json
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
import io
from unittest.mock import patch

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import BASE, FEATURES, carregar_dados, criar_entrada, interagir, treinar


class ProjetoTest(unittest.TestCase):
    def test_dados_e_mapeamento(self):
        df = carregar_dados(BASE / 'dados' / 'wholesale.csv')
        self.assertEqual(df.shape, (440, 8))
        self.assertEqual(list(df.columns), FEATURES + ['Channel'])
        self.assertTrue(all(str(t) == 'int64' for t in df.dtypes))
        self.assertEqual(df.iloc[0]['Region'], 2)
        self.assertEqual(df.iloc[0]['Channel'], 1)
        self.assertEqual(df.Channel.value_counts().to_dict(), {0: 298, 1: 142})

    def test_entradas_invalidas(self):
        for valores in ([9, 0, 0, 0, 0, 0, 0], [0, -1, 0, 0, 0, 0, 0],
                        [0, 1.5, 0, 0, 0, 0, 0], [0, 1]):
            with self.subTest(valores=valores), self.assertRaises(ValueError):
                criar_entrada(valores)

    def test_csv_invalido(self):
        original = pd.read_csv(BASE / 'dados' / 'wholesale.csv')
        with tempfile.TemporaryDirectory(dir=BASE) as tmp:
            caminho = Path(tmp) / 'invalido.csv'
            for coluna, valor in [('Channel', 'Outro'), ('Region', 'Unknown'),
                                   ('Milk', -1), ('Grocery', None)]:
                df = original.copy()
                df.loc[0, coluna] = valor
                df.to_csv(caminho, index=False)
                with self.subTest(coluna=coluna), self.assertRaises(ValueError):
                    carregar_dados(caminho)

    def test_separacao_e_persistencia(self):
        with tempfile.TemporaryDirectory(dir=BASE) as tmp:
            saida = Path(tmp)
            with redirect_stdout(io.StringIO()):
                modelo = treinar(BASE / 'dados' / 'wholesale.csv', saida)
            divisao = json.loads((saida / 'divisao.json').read_text())
            treino, teste = set(divisao['treino']), set(divisao['teste'])
            self.assertEqual((len(treino), len(teste)), (352, 88))
            self.assertFalse(treino & teste)
            self.assertEqual(treino | teste, set(range(440)))
            dados = carregar_dados(BASE / 'dados' / 'wholesale.csv')
            X = dados.loc[divisao['teste'], FEATURES]
            salvo = joblib.load(saida / 'modelo.joblib')
            self.assertTrue((modelo.predict(X) == salvo.predict(X)).all())
            pred = pd.read_csv(saida / 'predicoes_teste.csv')
            self.assertTrue((salvo.predict(X) == pred.previsto.to_numpy()).all())
            self.assertEqual(salvo.n_features_in_, 7)

    def _interagir(self, respostas):
        class Falso:
            classes_ = [0, 1]
            def predict(self, X): return [1]
            def predict_proba(self, X): return [[0.25, 0.75]]
        saida = io.StringIO()
        with patch('builtins.input', side_effect=respostas), redirect_stdout(saida):
            interagir(Falso())
        return saida.getvalue()

    def test_interacao_regiao_por_nome_e_saida(self):
        texto = self._interagir(['oporto', '1', '2', '3', '4', '5', '6', 'n'])
        self.assertIn('Canal previsto: Retail (confiança 75.0%)', texto)
        self.assertIn('1 (Oporto)', texto)

    def test_interacao_repete_so_campo_invalido(self):
        texto = self._interagir(['2', 'abc', '10', '-5', '1', '2', '3', '4', '5', 'não'])
        self.assertEqual(texto.count('Valor inválido'), 2)
        self.assertIn('Canal previsto', texto)

    def test_interacao_encerra_no_eof(self):
        self.assertIn('Encerrado.', self._interagir(['2', '1', EOFError()]))


if __name__ == '__main__':
    unittest.main()
