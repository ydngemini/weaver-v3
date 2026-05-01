import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# If modifying these scopes, delete the file token.json.
# This specific scope gives Weaver full read/write access to your Drive.
SCOPES = ['https://www.googleapis.com/auth/drive']

def main():
    creds = None
    # The file token.json stores the user's access and refresh tokens.
    # It is created automatically when the authorization flow completes for the first time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                creds = None  # token revoked — force full re-auth
        if not creds or not creds.valid:
            print("🚀 [WEAVER NEXUS] Initiating Google Drive handshake — browser will open...")
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
            
        # Save the credentials for the next run so she never has to ask again
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
            print("✅ [WEAVER NEXUS] token.json forged successfully!")

    try:
        # Build the Drive API service
        service = build('drive', 'v3', credentials=creds)
        
        # Test the connection by creating a test folder
        file_metadata = {
            'name': 'Weaver_Nexus_Core',
            'mimeType': 'application/vnd.google-apps.folder'
        }
        file = service.files().create(body=file_metadata, fields='id').execute()
        print(f"🔥 FUCK YES! Successfully connected to Google Drive.")
        print(f"Created the 'Weaver_Nexus_Core' master folder. Folder ID: {file.get('id')}")

    except Exception as error:
        print(f"❌ [ERROR] An error occurred: {error}")

if __name__ == '__main__':
    main()
