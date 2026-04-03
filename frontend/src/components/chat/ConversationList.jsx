import { motion } from 'framer-motion'
import SubjectBadge from '../shared/SubjectBadge'

export default function ConversationList({ conversations, activeId, onSelect, onNew, loading }) {
  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="p-4 border-b border-white/10">
        <button
          id="new-conversation-btn"
          onClick={onNew}
          className="btn-primary w-full flex items-center justify-center gap-2 text-sm"
        >
          <span className="text-base">✏️</span>
          New Chat
        </button>
      </div>

      {/* Conversation list */}
      <div className="flex-1 overflow-y-auto p-2 space-y-1">
        {loading ? (
          <div className="flex items-center justify-center py-8">
            <div className="w-5 h-5 border-2 border-brand-500/30 border-t-brand-500 rounded-full animate-spin" />
          </div>
        ) : conversations.length === 0 ? (
          <div className="text-center py-8">
            <p className="text-slate-500 text-sm">No conversations yet</p>
          </div>
        ) : (
          conversations.map((conv) => (
            <motion.button
              id={`conv-${conv.id}`}
              key={conv.id}
              onClick={() => onSelect(conv)}
              className={`w-full text-left px-3 py-2.5 rounded-xl transition-all duration-200 group ${
                conv.id === activeId
                  ? 'bg-brand-500/15 border border-brand-500/30'
                  : 'hover:bg-white/5 border border-transparent'
              }`}
              whileHover={{ x: 2 }}
            >
              <div className="flex items-start justify-between gap-2">
                <p className={`text-sm font-medium truncate flex-1 ${
                  conv.id === activeId ? 'text-white' : 'text-slate-300 group-hover:text-white'
                }`}>
                  {conv.title}
                </p>
                <SubjectBadge subject={conv.subject} size="xs" />
              </div>
              <p className="text-xs text-slate-600 mt-0.5">
                {new Date(conv.updated_at || conv.created_at).toLocaleDateString()}
              </p>
            </motion.button>
          ))
        )}
      </div>
    </div>
  )
}
