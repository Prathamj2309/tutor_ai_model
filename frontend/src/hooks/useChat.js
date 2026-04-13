import { useState, useCallback } from 'react'
import API from '../lib/api'
import toast from 'react-hot-toast'
import { supabase } from '../lib/supabaseClient'

export function useChat() {
  const [conversations, setConversations] = useState([])
  const [messages, setMessages] = useState([])
  const [activeConversation, setActiveConversation] = useState(null)
  const [loading, setLoading] = useState(false)
  const [sending, setSending] = useState(false)

  const fetchConversations = useCallback(async () => {
    setLoading(true)
    try {
      const { data } = await API.get('/conversations')
      setConversations(data)
    } catch (err) {
      toast.error('Failed to load conversations')
    } finally {
      setLoading(false)
    }
  }, [])

  const fetchMessages = useCallback(async (conversationId) => {
    setLoading(true)
    try {
      const { data } = await API.get(`/conversations/${conversationId}/messages`)
      setMessages(data)
    } catch (err) {
      toast.error('Failed to load messages')
    } finally {
      setLoading(false)
    }
  }, [])

  const startConversation = useCallback(async (subject = 'general') => {
    try {
      const { data } = await API.post('/conversations', { subject })
      setActiveConversation(data)
      setConversations((prev) => [data, ...prev])
      setMessages([])
      return data
    } catch (err) {
      toast.error('Failed to create conversation')
    }
  }, [])

  const sendMessage = useCallback(async ({ conversationId, content, imageFile }) => {
    setSending(true)

    const tempUserId = Date.now().toString() + "_u";
    const tempAiId = Date.now().toString() + "_ai";
    const userMsg = { id: tempUserId, role: 'user', content }
    const aiMsg = { id: tempAiId, role: 'assistant', content: '' }
    
    setMessages((prev) => [...prev, userMsg, aiMsg])

    try {
      const formData = new FormData()
      formData.append('content', content || '')
      formData.append('conversationId', conversationId)
      if (imageFile) formData.append('image', imageFile)

      const { data: { session } } = await supabase.auth.getSession()
      const token = session?.access_token

      const baseURL = import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, '') || '/api'
      
      const response = await fetch(`${baseURL}/chat/stream`, {
        method: 'POST',
        headers: {
          ...(token ? { 'Authorization': `Bearer ${token}` } : {})
        },
        body: formData
      })

      if (!response.ok) {
        throw new Error('Network response was not ok')
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder("utf-8")
      let aiContent = ""

      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        
        const chunk = decoder.decode(value, { stream: true })
        const lines = chunk.split('\n')
        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const data = line.slice(6)
                if (data.trim() === '[DONE]') {
                    continue
                }
                try {
                  const parsed = JSON.parse(data)
                  if (parsed.content) {
                      aiContent += parsed.content
                      setMessages(prev => prev.map(m => m.id === tempAiId ? { ...m, content: aiContent } : m))
                  }
                } catch(e) {}
            }
        }
      }
      
      // We trigger a re-fetch of messages in the background to get their db IDs
      setTimeout(() => fetchMessages(conversationId), 1000)
    } catch (err) {
      toast.error('Failed to send message')
      console.error(err)
      setMessages(prev => prev.filter(m => m.id !== tempAiId))
      throw err
    } finally {
      setSending(false)
    }
  }, [fetchMessages])

  const selectConversation = useCallback(async (conv) => {
    setActiveConversation(conv)
    await fetchMessages(conv.id)
  }, [fetchMessages])

  return {
    conversations,
    messages,
    activeConversation,
    loading,
    sending,
    fetchConversations,
    startConversation,
    sendMessage,
    selectConversation,
  }
}
