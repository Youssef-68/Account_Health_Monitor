from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.health_engine import calculate_health_score, categorize_health
from src.data_loader import load_data
from src.models import load_or_train
import pandas as pd

app = FastAPI(
    title="Account Health Monitor API",
    description="Real-time Account Health Scoring, Churn Risk Assessment, and MRR Forecasting API.",
    version="2.0.0"
)

df_base = load_data()
model = load_or_train(df_base)

class AccountHealthInput(BaseModel):
    account_id: str = Field("ACC-9999", example="ACC-1024")
    company_size: str = Field("SMB", example="Enterprise")
    industry: str = Field("Technology", example="Healthcare")
    contract_type: str = Field("Monthly", example="Annual")
    active_users: int = Field(25, example=50)
    usage_growth: float = Field(0.05, example=-0.15)
    feature_adoption_rate: float = Field(0.6, example=0.3)
    error_rate: float = Field(0.01, example=0.05)
    tickets_count: int = Field(2, example=8)
    payment_delay_flag: int = Field(0, example=1)
    current_mrr: float = Field(1500.0, example=3500.0)

@app.get("/")
def root():
    return {"app": "Account Health Monitor API", "status": "Operational"}

@app.post("/assess-health")
def assess_account_health(account: AccountHealthInput):
    try:
        data = account.dict()
        score = calculate_health_score(data)
        status = categorize_health(score)

        # Predict Future MRR
        df_input = pd.DataFrame([data])
        predicted_mrr = float(model.predict(df_input)[0])

        return {
            "account_id": account.account_id,
            "health_score": score,
            "health_status": status,
            "current_mrr": account.current_mrr,
            "predicted_next_mrr": round(predicted_mrr, 2),
            "mrr_risk_delta": round(predicted_mrr - account.current_mrr, 2),
            "action_required": status in ["At Risk", "Critical"]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))