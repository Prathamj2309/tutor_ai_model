import { motion } from 'framer-motion'
import { useEffect, useRef } from 'react'
import MessageBubble from './MessageBubble'

export default function ChatWindow({ messages, sending }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, sending])

  if (messages.length === 0 && !sending) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-center p-8">
        <div className="text-5xl mb-4">💡</div>
        <h3 className="text-xl font-semibold text-slate-200 mb-2">Ask your first question</h3>
        <p className="text-slate-500 text-sm max-w-sm">
          Type a question or upload an image of a problem. I'll explain it step-by-step with full working.
        </p>
        <div className="mt-6 flex flex-wrap justify-center gap-2">
          {[
            'What is the photoelectric effect?',
            'Explain Le Chatelier\'s principle',
            'Derive the lens formula',
          ].map((q) => (
            <span key={q} className="text-xs bg-surface-700 border border-white/10 text-slate-400 px-3 py-1.5 rounded-lg cursor-default">
              {q}
            </span>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-y-auto px-4 py-6 space-y-2">
      {messages.map((msg, i) => (
        <motion.div
          key={msg.id || i}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.25 }}
        >
          <MessageBubble message={msg} />
        </motion.div>
      ))}

      {/* Typing indicator */}
      {sending && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-end gap-2"
        >
          <div className="w-8 h-8 rounded-full bg-brand-500/20 border border-brand-500/30 flex items-center justify-center text-sm flex-shrink-0">
            🤖
          </div>
          <div className="bg-surface-700 border border-white/10 rounded-2xl rounded-bl-sm px-4 py-3">
            <div className="flex gap-1 items-center h-4">
              {[0, 1, 2].map((i) => (
                <span
                  key={i}
                  className="w-2 h-2 bg-slate-400 rounded-full animate-bounce"
                  style={{ animationDelay: `${i * 0.15}s` }}
                />
              ))}
            </div>
          </div>
        </motion.div>
      )}

      <div ref={bottomRef} />
    </div>
  )
}
