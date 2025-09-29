import pandas as pd
from src.api import _predict_proba_flexible


class FakeModelTextDF:
    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame) and "job_text" in X.columns:
            
            raise ValueError("Expected 2D with specific columns")
        if isinstance(X, pd.DataFrame) and "text" in X.columns:
           
            return [[0.3, 0.7]]
        if isinstance(X, list):
            
            raise ValueError("List not accepted")
        raise ValueError("Unsupported input")


class FakeModelListOnly:
    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
           
            raise ValueError("DataFrame not accepted")
        if isinstance(X, list):
            
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
    
    assert abs(proba - 0.7) < 1e-9

def test_predict_proba_path_3_list():
    model = FakeModelListOnly()
    proba = _predict_proba_flexible(model, _req_dict_base())
    
    assert abs(proba - 0.4) < 1e-9
