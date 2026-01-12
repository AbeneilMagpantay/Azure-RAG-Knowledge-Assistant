"""Supabase integration for chat history persistence."""

import os
from typing import List, Dict, Optional
from datetime import datetime


class ChatHistoryStore:
    """Store chat history in Supabase."""
    
    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self._client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize Supabase client."""
        # Try st.secrets first (Streamlit Cloud), then os.getenv (local)
        try:
            import streamlit as st
            supabase_url = st.secrets.get("SUPABASE_URL") or os.getenv("SUPABASE_URL")
            supabase_key = st.secrets.get("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_ANON_KEY")
        except Exception:
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_ANON_KEY")
        
        if not supabase_url or not supabase_key:
            print("⚠️ Supabase not configured - chat history will not persist")
            return
        
        try:
            from supabase import create_client
            self._client = create_client(supabase_url, supabase_key)
        except ImportError:
            print("⚠️ supabase package not installed")
        except Exception as e:
            print(f"⚠️ Supabase connection failed: {e}")
    
    def save_message(self, role: str, content: str) -> bool:
        """Save a chat message."""
        import streamlit as st
        
        if not self._client:
            st.toast("❌ Supabase client not initialized", icon="⚠️")
            return False
        
        try:
            result = self._client.table("chat_history").insert({
                "session_id": self.session_id,
                "role": role,
                "content": content
            }).execute()
            return True
        except Exception as e:
            st.toast(f"❌ Save failed: {str(e)[:50]}", icon="⚠️")
            return False
    
    def load_history(self) -> List[Dict[str, str]]:
        """Load chat history for current session."""
        if not self._client:
            return []
        
        try:
            response = self._client.table("chat_history") \
                .select("role, content") \
                .eq("session_id", self.session_id) \
                .order("created_at") \
                .execute()
            
            return [{"role": msg["role"], "content": msg["content"]} 
                    for msg in response.data]
        except Exception as e:
            print(f"Failed to load history: {e}")
            return []
    
    def clear_history(self) -> bool:
        """Clear chat history for current session."""
        if not self._client:
            return False
        
        try:
            self._client.table("chat_history") \
                .delete() \
                .eq("session_id", self.session_id) \
                .execute()
            return True
        except Exception as e:
            print(f"Failed to clear history: {e}")
            return False
    
    @property
    def is_connected(self) -> bool:
        """Check if Supabase is connected."""
        return self._client is not None
