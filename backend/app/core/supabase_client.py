import os
from supabase import create_client, Client
from app.core.config import settings

if not settings.supabase_url or not settings.supabase_service_key:
    # Allows the app to run (for tests/schema building) even if not set
    print("Warning: SUPABASE_URL or SUPABASE_SERVICE_KEY is missing.")
    url = settings.supabase_url or "http://localhost"
    key = settings.supabase_service_key or "dummy"
else:
    url = settings.supabase_url
    key = settings.supabase_service_key

# Initialize Service Role Client (Bypasses RLS). Keep secure on server.
supabase: Client = create_client(url, key)
