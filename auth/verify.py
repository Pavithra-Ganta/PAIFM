import firebase_admin
from firebase_admin import credentials, auth

# Initialize Firebase Admin
import json, os

firebase_creds = json.loads(os.environ["FIREBASE_CREDENTIALS"])
cred = credentials.Certificate(firebase_creds)

firebase_admin.initialize_app(cred)

def verify_id_token(id_token):
    try:
        decoded_token = auth.verify_id_token(id_token)
        uid = decoded_token['uid']
        return uid
    except Exception as e:
        raise Exception("Invalid or expired token")
