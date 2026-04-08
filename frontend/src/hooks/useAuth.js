import { useEffect } from 'react'
import { supabase } from '../lib/supabaseClient'
import useAuthStore from '../store/authStore'

export function useAuth() {
  const { user, session, loading, setSession, setLoading, clearAuth } = useAuthStore()

  useEffect(() => {
    // Restore session on mount
    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session)
      setLoading(false)
    })

    // Subscribe to auth changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      setSession(session)
      setLoading(false)
    })

    return () => subscription.unsubscribe()
  }, [])

  const signInWithEmail = (email, password) =>
    supabase.auth.signInWithPassword({ email, password })

  const signUpWithEmail = (email, password, fullName) =>
    supabase.auth.signUp({ email, password, options: { data: { full_name: fullName } } })

  const signOut = () => {
    clearAuth()
    return supabase.auth.signOut()
  }

  return { user, session, loading, signInWithEmail, signUpWithEmail, signOut }
}
