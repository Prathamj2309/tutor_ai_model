import axios from 'axios'
import { supabase } from '../lib/supabaseClient'

const API = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '/api',
})

// Attach the Supabase Bearer token on every request
API.interceptors.request.use(async (config) => {
  const { data: { session } } = await supabase.auth.getSession()
  if (session?.access_token) {
    config.headers.Authorization = `Bearer ${session.access_token}`
  }
  return config
})

export default API
