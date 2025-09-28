import pandas as pd
from src.api import _predict_proba_flexible

# Modelo que só aceita DataFrame com coluna 'text'
class FakeModelTextDF:
    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame) and "job_text" in X.columns:
            # Falha no DF completo (colunas originais)
            raise ValueError("Expected 2D with specific columns")
        if isinstance(X, pd.DataFrame) and "text" in X.columns:
            # Sucesso quando vier {'text': ...}
            return [[0.3, 0.7]]
        if isinstance(X, list):
            # Não aceita lista; força caminho (2)
            raise ValueError("List not accepted")
        raise ValueError("Unsupported input")

# Modelo que só aceita lista 1D
class FakeModelListOnly:
    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            # Falha tanto no DF completo quanto no DF {'text': ...}
            raise ValueError("DataFrame not accepted")
        if isinstance(X, list):
            # Sucesso quando vier lista 1D
            return [[0.6, 0.4]]
        raise ValueError("Unsupported input")

def _req_dict_base():
    return {
        "job_text": "QA Selenium Java",
        "cand_text": "Selenium WebDriver, Java, testes automatizados",
        "situacao_norm": "prospect",
        "score_tecnico": 0.25,
    }

def test_predict_proba_path_2_text_df():
    model = FakeModelTextDF()
    proba = _predict_proba_flexible(model, _req_dict_base())
    # Esperamos pegar o ramo (2) e retornar 0.7
    assert abs(proba - 0.7) < 1e-9

def test_predict_proba_path_3_list():
    model = FakeModelListOnly()
    proba = _predict_proba_flexible(model, _req_dict_base())
    # Esperamos pegar o ramo (3) e retornar 0.4 (classe 1 em [0][1])
    assert abs(proba - 0.4) < 1e-9
