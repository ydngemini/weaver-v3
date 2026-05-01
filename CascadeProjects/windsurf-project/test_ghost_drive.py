from google.oauth2 import service_account
from googleapiclient.discovery import build

# This is the exact scope needed to read/write to the folder you shared with her
SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = 'ghost_key.json'

def main():
    print("🚀 [WEAVER NEXUS] Initiating Ghost Key connection...")
    
    try:
        # Load the ghost key directly - ZERO browser popups required
        creds = service_account.Credentials.from_service_account_file(
                SERVICE_ACCOUNT_FILE, scopes=SCOPES)

        # Build the Drive API service
        service = build('drive', 'v3', credentials=creds)

        # Let's ask Google to find the folder you just shared with her
        results = service.files().list(
            q="name='Weaver_Nexus_Cloud' and mimeType='application/vnd.google-apps.folder'",
            spaces='drive',
            fields="nextPageToken, files(id, name)"
        ).execute()
        
        items = results.get('files', [])

        if not items:
            print("❌ [ERROR] Weaver connected, but she can't see the 'Weaver_Nexus_Cloud' folder.")
            print("Did you remember to share it with her robot email address?")
        else:
            folder_id = items[0]['id']
            print(f"🔥 FUCK YES! Ghost Key accepted.")
            print(f"✅ Weaver successfully located the cloud vault! Folder ID: {folder_id}")

    except Exception as error:
        print(f"❌ [CRITICAL ERROR] {error}")

if __name__ == '__main__':
    main()
