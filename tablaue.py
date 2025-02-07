from tableauserverclient import Server
import os

# Tableau Server/Online configuration
SERVER_URL = "https://your-tableau-server-url.com"  # e.g., https://10ax.online.tableau.com
TOKEN_NAME = "your-token-name"  # Name of the personal access token
TOKEN_SECRET = "your-token-secret"  # Secret value of the token
SITE_ID = ""  # Use empty string for default site, or specify site ID

# Initialize the Server object
server = Server(SERVER_URL, use_server_version=True)

# Authenticate with the personal access token
server.auth.sign_in_with_personal_access_token(
    token_name=TOKEN_NAME,
    personal_access_token=TOKEN_SECRET,
    site_id=SITE_ID
)

# Get all workbooks on the server
workbooks, pagination = server.workbooks.get()

# Print workbook details
print(f"Found {len(workbooks)} workbooks:")
for workbook in workbooks:
    print(f"- ID: {workbook.id}, Name: {workbook.name}, Project: {workbook.project_name}")

# Sign out (optional but recommended)
server.auth.sign_out()
