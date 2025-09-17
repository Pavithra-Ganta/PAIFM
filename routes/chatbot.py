from fastapi import APIRouter, Header, HTTPException, Body
from auth.verify import verify_id_token
from utils.intent_engine import IntentEngine
from utils.slot_extraction import extract_time_window
from utils.context_engine import ContextEngine
from utils.prompt_builder import build_prompt
from utils.llm import ask_mistral
from google.cloud import firestore
from pydantic import BaseModel
from datetime import datetime, timedelta, timezone


router = APIRouter()
import os, json
from google.cloud import firestore
from google.oauth2 import service_account

firebase_creds = json.loads(os.environ["FIREBASE_CREDENTIALS"])
credentials = service_account.Credentials.from_service_account_info(firebase_creds)
db = firestore.Client(credentials=credentials, project=firebase_creds["project_id"])

intent_engine = IntentEngine()
context_engine = ContextEngine(db)

class ChatbotRequest(BaseModel):
    prompt: str
@router.post("/chatbot")
async def chatbot(request: ChatbotRequest, authorization: str = Header(None)):
    user_prompt = request.prompt
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")

    token_id = authorization.split(" ")[1]
    uid = verify_id_token(token_id)

    intent = intent_engine.detect_intent(user_prompt)

    # Fetch profile
    user_ref = db.collection("users").document(uid)
    user_doc = user_ref.get()
    profile = user_doc.to_dict() if user_doc.exists else {}

    # Fetch transactions if needed
    transactions = []
    if intent in ["expense_query", "budget_query", "expense_analysis", "savings_advice"]:
        days = extract_time_window(user_prompt) or 30
        cutoff_ts = datetime.now(timezone.utc) - timedelta(days=days)
        txn_ref = user_ref.collection("transactions")
        docs = txn_ref.where("timestamp", ">=", cutoff_ts).stream()
        transactions = []
        for doc in docs:
            data = doc.to_dict()
            transactions.append({
                "category": data.get("category"),
                "amount": float(data.get("amount", 0) or 0),
            })
        if not transactions:
            docs = (
                txn_ref.order_by("timestamp", direction=firestore.Query.DESCENDING)
                  .limit(100)
                  .stream()
            )
        for doc in docs:
            data = doc.to_dict()
            transactions.append({
                "category": data.get("category"),
                "amount": float(data.get("amount", 0) or 0),
            })

    # Build context dict
    context = {
        "budget": profile.get("budget"),
        "income": profile.get("income"),
        "goal": profile.get("goal"),
        "risk_level": profile.get("risk_level"),
        "last_bot_response": context_engine.get_user_context(uid, "last_bot_response")
    }

    # Build prompt and call LLM
    final_prompt = build_prompt(intent, user_prompt, context, transactions)
    print(final_prompt)
    bot_response = ask_mistral(final_prompt)

    # Update context memory
    context_engine.set_user_context(uid, "last_intent", intent)
    context_engine.set_user_context(uid, "last_bot_response", bot_response)

    return {"response": bot_response}
