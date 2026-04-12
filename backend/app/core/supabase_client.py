from supabase import create_client, Client, ClientOptions

from app.core.config import settings

if not settings.supabase_url or not settings.supabase_service_key:
    # Allows the app to run (for tests/schema building) even if not set
    print("Warning: SUPABASE_URL or SUPABASE_SERVICE_KEY is missing.")
    url = settings.supabase_url or "http://localhost"
    key = settings.supabase_service_key or "dummy"
else:
    url = settings.supabase_url
    key = settings.supabase_service_key

# Increase timeout to handle slow handshakes
options = ClientOptions(
    postgrest_client_timeout=30,
    storage_client_timeout=30,
)

# Initialize Service Role Client (Bypasses RLS). Keep secure on server.
supabase: Client = create_client(url, key, options=options)
