import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { useChat } from '../hooks/useChat'
import ConversationList from '../components/chat/ConversationList'
import ChatWindow from '../components/chat/ChatWindow'
import MessageInput from '../components/chat/MessageInput'
import SubjectBadge from '../components/shared/SubjectBadge'
import toast from 'react-hot-toast'

const SUBJECTS = ['general', 'physics', 'chemistry', 'mathematics']

export default function ChatPage() {
  const {
    conversations, messages, activeConversation, loading, sending,
    fetchConversations, startConversation, sendMessage, selectConversation,
  } = useChat()

  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [subjectModal, setSubjectModal] = useState(false)

  useEffect(() => {
    fetchConversations()
  }, [])

  const handleNewChat = () => setSubjectModal(true)

  const handleSubjectSelect = async (subject) => {
    setSubjectModal(false)
    await startConversation(subject)
  }

  const handleSend = async ({ content, imageFile }) => {
    if (!activeConversation) {
      toast.error('Please start or select a conversation first.')
      return
    }
    await sendMessage({ conversationId: activeConversation.id, content, imageFile })
  }

  return (
    <div className="flex flex-1 overflow-hidden h-full">
      {/* Sidebar */}
      <motion.aside
        animate={{ width: sidebarOpen ? 280 : 0, opacity: sidebarOpen ? 1 : 0 }}
        transition={{ duration: 0.25 }}
        className="bg-surface-800 border-r border-white/8 flex-shrink-0 overflow-hidden"
      >
        <div className="w-[280px] h-full">
          <ConversationList
            conversations={conversations}
            activeId={activeConversation?.id}
            onSelect={selectConversation}
            onNew={handleNewChat}
            loading={loading && conversations.length === 0}
          />
        </div>
      </motion.aside>

      {/* Main chat area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Chat header */}
        <div className="h-12 border-b border-white/10 bg-surface-800/60 flex items-center px-4 gap-3">
          <button
            id="toggle-sidebar-btn"
            onClick={() => setSidebarOpen((p) => !p)}
            className="text-slate-400 hover:text-white transition-colors p-1.5 rounded-lg hover:bg-white/5"
          >
            ☰
          </button>
          {activeConversation ? (
            <>
              <p className="text-sm font-semibold text-slate-200 truncate flex-1">{activeConversation.title}</p>
              <SubjectBadge subject={activeConversation.subject} />
            </>
          ) : (
            <p className="text-sm text-slate-500 italic">Select or start a conversation</p>
          )}
        </div>

        {/* Messages */}
        <ChatWindow messages={messages} sending={sending} />

        {/* Input */}
        <MessageInput onSend={handleSend} disabled={sending || !activeConversation} />
      </div>

      {/* Subject selection modal */}
      {subjectModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <motion.div
            initial={{ opacity: 0, scale: 0.92 }}
            animate={{ opacity: 1, scale: 1 }}
            className="glass-card p-6 w-full max-w-xs"
          >
            <h3 className="text-white font-bold text-lg mb-1">New Conversation</h3>
            <p className="text-slate-400 text-sm mb-5">Select a subject to get started</p>
            <div className="space-y-2">
              {SUBJECTS.map((s) => (
                <button
                  key={s}
                  id={`subject-select-${s}`}
                  onClick={() => handleSubjectSelect(s)}
                  className="w-full text-left btn-ghost capitalize flex items-center gap-3"
                >
                  <span>{s === 'physics' ? '⚛️' : s === 'chemistry' ? '🧪' : s === 'mathematics' ? '📐' : '💬'}</span>
                  {s}
                </button>
              ))}
            </div>
            <button onClick={() => setSubjectModal(false)} className="mt-4 text-slate-500 hover:text-white text-sm transition-colors w-full text-center">
              Cancel
            </button>
          </motion.div>
        </div>
      )}
    </div>
  )
}
