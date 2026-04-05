import { create } from 'zustand'

const useAuthStore = create((set) => ({
  user: null,
  session: null,
  loading: true,
  setUser: (user) => set({ user }),
  setSession: (session) => set({ session, user: session?.user ?? null }),
  setLoading: (loading) => set({ loading }),
  clearAuth: () => set({ user: null, session: null }),
}))

export default useAuthStore
