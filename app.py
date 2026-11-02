"""API HTTP para servir o modelo treinado (uso em container/teste manual)."""
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from main import BASE, FEATURES, LABELS, criar_entrada

app = FastAPI(title="AG2 - Classificador de canais de vendas")
MODELO_PATH = Path(BASE) / "resultados" / "modelo.joblib"
_modelo = None


class Cliente(BaseModel):
    Region: int = Field(..., ge=0, le=2, description="0=Lisbon, 1=Oporto, 2=Other")
    Fresh: int = Field(..., ge=0)
    Milk: int = Field(..., ge=0)
    Grocery: int = Field(..., ge=0)
    Frozen: int = Field(..., ge=0)
    Detergents_Paper: int = Field(..., ge=0, alias="Detergents Paper")
    Delicatessen: int = Field(..., ge=0)

    class Config:
        populate_by_name = True


@app.on_event("startup")
def carregar_modelo():
    global _modelo
    if not MODELO_PATH.exists():
        raise RuntimeError(
            f"Modelo não encontrado em {MODELO_PATH}. Rode 'python main.py --sem-interacao' antes.")
    _modelo = joblib.load(MODELO_PATH)


@app.get("/health")
def health():
    return {"status": "ok", "modelo_carregado": _modelo is not None}


@app.post("/predict")
def predict(cliente: Cliente):
    try:
        valores = [cliente.Region, cliente.Fresh, cliente.Milk, cliente.Grocery,
                   cliente.Frozen, cliente.Detergents_Paper, cliente.Delicatessen]
        entrada = criar_entrada(valores)
    except ValueError as erro:
        raise HTTPException(status_code=422, detail=str(erro))
    canal = LABELS[int(_modelo.predict(entrada)[0])]
    probabilidades = dict(zip(LABELS, _modelo.predict_proba(entrada)[0].tolist()))
    return {"canal_previsto": canal, "probabilidades": probabilidades}
