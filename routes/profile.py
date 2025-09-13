from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from auth.verify import verify_id_token
from google.cloud import firestore
from firebase_admin import auth as firebase_auth
from typing import Dict,Any

router = APIRouter()

import os, json
from google.cloud import firestore
from google.oauth2 import service_account

firebase_creds = json.loads(os.environ["FIREBASE_CREDENTIALS"])
credentials = service_account.Credentials.from_service_account_info(firebase_creds)
db = firestore.Client(credentials=credentials, project=firebase_creds["project_id"])

# Request schema
class ProfileRequest(BaseModel):
    name: str
    age: int
    income: int
    goal: str

@router.post("/profile")
async def set_profile(profile: Dict[str, Any], authorization: str = Header(None)):
    # Verify Authorization header
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    id_token = authorization.split(" ")[1]
    
    try:
        decoded_token = firebase_auth.verify_id_token(id_token)
        uid = decoded_token['uid']
        db.collection("users").document(uid).set(profile, merge=True)
        return {"message": "Profile updated successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    
@router.get("/profile")
async def get_profile(authorization: str = Header(None)):
    # Verify Authorization header
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    id_token = authorization.split(" ")[1]
    
    try:
        decoded_token = firebase_auth.verify_id_token(id_token)
        uid = decoded_token['uid']
        
        # Retrieve user profile from Firestore
        doc_ref = db.collection("users").document(uid)
        doc = doc_ref.get()
        
        if doc.exists:
            return doc.to_dict()
        else:
            return {"message": "Profile not found"}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.patch("/profile")
async def update_profile(profile: dict, authorization: str = Header(None)):
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")

    id_token = authorization.split(" ")[1]

    try:
        decoded_token = firebase_auth.verify_id_token(id_token)
        uid = decoded_token['uid']

        # Update only provided fields
        db.collection("users").document(uid).set(profile, merge=True)

        return {"message": "Profile updated successfully"}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


