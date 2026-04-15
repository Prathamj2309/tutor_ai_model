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
      <div className="flex-1 flex flex-col items-center justify-center text-center p-8 relative overflow-hidden">
        {/* Glow behind the icon */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-64 h-64 bg-brand-500/10 rounded-full blur-[80px] pointer-events-none"></div>
        <motion.div 
           initial={{ scale: 0.8, opacity: 0 }} 
           animate={{ scale: 1, opacity: 1 }} 
           transition={{ duration: 0.5, ease: 'easeOut' }}
           className="relative"
        >
          <div className="text-6xl mb-6 drop-shadow-2xl brightness-125 saturate-150">💡</div>
        </motion.div>
        <motion.h3 
           initial={{ y: 10, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ delay: 0.1 }}
           className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-white to-slate-400 mb-3"
        >
          Ask your first question
        </motion.h3>
        <motion.p 
           initial={{ y: 10, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ delay: 0.2 }}
           className="text-slate-400 text-sm max-w-sm"
        >
          Type a question or upload an image of a problem. Let's solve it together step-by-step.
        </motion.p>
        <motion.div 
           initial={{ y: 10, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ delay: 0.3 }}
           className="mt-8 flex flex-wrap justify-center gap-3 w-full max-w-lg"
        >
          {[
            'What is the photoelectric effect?',
            'Explain Le Chatelier\'s principle',
            'Derive the lens formula',
          ].map((q) => (
            <div key={q} className="group cursor-pointer">
              <span className="inline-block text-xs bg-white/5 hover:bg-brand-500/20 border border-white/10 hover:border-brand-500/30 text-slate-300 hover:text-brand-300 px-4 py-2 rounded-xl transition-all duration-300 shadow-sm backdrop-blur-md">
                {q}
              </span>
            </div>
          ))}
        </motion.div>
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-y-auto px-4 py-6 space-y-6 scroll-smooth styled-scrollbar">
      {messages.map((msg, i) => (
        <motion.div
          key={msg.id || i}
          initial={{ opacity: 0, scale: 0.98, y: 10 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          transition={{ duration: 0.3, ease: 'easeOut' }}
        >
          <MessageBubble message={msg} />
        </motion.div>
      ))}

      {/* Typing indicator logic updated for streaming */}
      {sending && messages.length > 0 && messages[messages.length - 1].content === '' && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-end gap-3 px-2"
        >
          <div className="w-10 h-10 rounded-full bg-gradient-to-tr from-indigo-500/30 to-purple-500/30 border border-indigo-500/40 flex items-center justify-center text-lg shadow-[0_0_15px_rgba(99,102,241,0.2)]">
            🤖
          </div>
          <div className="bg-surface-800/80 backdrop-blur-md border border-white/10 rounded-2xl rounded-bl-sm px-5 py-4 shadow-lg">
            <div className="flex gap-1.5 items-center h-2">
              {[0, 1, 2].map((i) => (
                <span
                  key={i}
                  className="w-2 h-2 bg-brand-400 rounded-full animate-bounce"
                  style={{ animationDelay: `${i * 0.15}s` }}
                />
              ))}
            </div>
          </div>
        </motion.div>
      )}

      <div ref={bottomRef} className="h-4" />
    </div>
  )
}
