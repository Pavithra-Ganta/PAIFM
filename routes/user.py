from fastapi import APIRouter, Header, HTTPException
from firebase_admin import auth as firebase_auth
from google.cloud import firestore

router = APIRouter()

# Initialize Firestore client
import os, json
from google.cloud import firestore
from google.oauth2 import service_account

firebase_creds = json.loads(os.environ["FIREBASE_CREDENTIALS"])
credentials = service_account.Credentials.from_service_account_info(firebase_creds)
db = firestore.Client(credentials=credentials, project=firebase_creds["project_id"])

# will be using after signup
@router.post("/create-user")
async def create_user(authorization: str = Header(None)):
    # ✅ Verify Authorization header
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")

    id_token = authorization.split(" ")[1]

    try:
        # ✅ Verify Firebase ID token
        decoded_token = firebase_auth.verify_id_token(id_token)
        uid = decoded_token['uid']
        email = decoded_token['email']

        # ✅ Create user document in Firestore with email and profile_completed:false
        db.collection("users").document(uid).set({
            "email": email,
            "profile_completed": False
        }, merge=True)

        return {"message": "User document created successfully"}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
