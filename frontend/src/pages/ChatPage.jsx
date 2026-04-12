import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { useChat } from '../hooks/useChat'
import ConversationList from '../components/chat/ConversationList'
import ChatWindow from '../components/chat/ChatWindow'
import MessageInput from '../components/chat/MessageInput'
import SubjectBadge from '../components/shared/SubjectBadge'
import toast from 'react-hot-toast'

export default function ChatPage() {
  const {
    conversations, messages, activeConversation, loading, sending,
    fetchConversations, startConversation, sendMessage, selectConversation,
  } = useChat()
  const [sidebarOpen, setSidebarOpen] = useState(true)

  useEffect(() => { fetchConversations() }, [])

  const handleNewChat = async () => { await startConversation('general') }

  const handleSend = async ({ content, imageFile }) => {
    if (!activeConversation) {
      toast.error('Please start a new chat first.')
      return
    }
    await sendMessage({ conversationId: activeConversation.id, content, imageFile })
  }

  return (
    <div className="flex flex-1 overflow-hidden h-full">
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
      <div className="flex-1 flex flex-col overflow-hidden">
        <div className="h-12 border-b border-white/10 bg-surface-800/60 flex items-center px-4 gap-3">
          <button onClick={() => setSidebarOpen((p) => !p)} className="text-slate-400">☰</button>
          {activeConversation ? (
            <>
              <p className="text-sm font-semibold truncate flex-1">{activeConversation.title}</p>
              <SubjectBadge subject={activeConversation.subject} />
            </>
          ) : (
            <p className="text-sm italic">Start a new chat to begin</p>
          )}
        </div>
        <ChatWindow messages={messages} sending={sending} />
        <MessageInput onSend={handleSend} disabled={sending || !activeConversation} />
      </div>
    </div>
  )
}
