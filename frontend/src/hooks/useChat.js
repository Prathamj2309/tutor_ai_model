import { useState, useCallback } from 'react'
import API from '../lib/api'
import toast from 'react-hot-toast'

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
    try {
      const formData = new FormData()
      formData.append('content', content || '')
      formData.append('conversationId', conversationId)
      if (imageFile) formData.append('image', imageFile)

      const { data } = await API.post('/chat', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })

      // data = { userMessage, aiMessage }
      setMessages((prev) => [...prev, data.userMessage, data.aiMessage])
      return data
    } catch (err) {
      toast.error('Failed to send message')
      throw err
    } finally {
      setSending(false)
    }
  }, [])

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
