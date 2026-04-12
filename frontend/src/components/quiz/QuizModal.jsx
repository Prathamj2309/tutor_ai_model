import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'

export default function QuizModal({ quiz, onSubmit, onClose, loading }) {
  const [responses, setResponses] = useState({}) // { [questionId]: selectedOption }
  const [submitted, setSubmitted] = useState(false)
  const [result, setResult] = useState(null)
  
  // Timer setup: User provided timeLimit (in minutes) or fallback to 1 min per question
  const [timeLeft, setTimeLeft] = useState((quiz?.timeLimit || quiz?.questions?.length || 0) * 60)

  if (!quiz) return null

  const { questions, subject } = quiz

  useEffect(() => {
    if (submitted || timeLeft <= 0) {
      if (timeLeft === 0 && !submitted) handleSubmit()
      return
    }
    const timer = setInterval(() => setTimeLeft(prev => prev - 1), 1000)
    return () => clearInterval(timer)
  }, [timeLeft, submitted])

  const selectOption = (qId, option) => {
    if (submitted) return
    setResponses((prev) => ({ ...prev, [qId]: option }))
  }

  const handleSubmit = async () => {
    // Check if responses are partial, but still submit them empty if forced by timer
    const res = await onSubmit(quiz.id, responses)
    setResult(res)
    setSubmitted(true)
  }

  const formatTime = (seconds) => {
    const m = Math.floor(seconds / 60).toString().padStart(2, '0')
    const s = (seconds % 60).toString().padStart(2, '0')
    return `${m}:${s}`
  }

  const allAnswered = Object.keys(responses).length === questions.length
  const OPTS = ['A', 'B', 'C', 'D']

  return (
    <AnimatePresence>
      <motion.div
        className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.95, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: 20 }}
          className="glass-card w-full max-w-2xl max-h-[90vh] flex flex-col overflow-hidden"
        >
          {/* Header */}
          <div className="flex items-center justify-between px-6 py-4 border-b border-white/10">
            <div>
              <h2 className="text-lg font-bold text-white">
                {submitted ? '📊 Quiz Results' : '📝 Mock Test'}
              </h2>
              <p className="text-xs text-slate-400 mt-0.5 capitalize">{subject} · {questions.length} questions</p>
            </div>
            <div className="flex items-center gap-4">
              {!submitted && (
                <div className={`text-sm font-mono px-3 py-1 rounded-lg border ${
                  timeLeft < 60 ? 'bg-red-500/20 border-red-500/50 text-red-400 animate-pulse' : 'bg-surface-700/50 border-white/10 text-brand-300'
                }`}>
                  ⏱️ {formatTime(timeLeft)}
                </div>
              )}
              <button onClick={onClose} className="text-slate-400 hover:text-white text-xl transition-colors">✕</button>
            </div>
          </div>

          {/* Result banner */}
          {submitted && result && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className={`px-6 py-4 border-b border-white/10 flex items-center gap-4 ${
                result.score / result.total >= 0.6 ? 'bg-green-500/10' : 'bg-red-500/10'
              }`}
            >
              <span className="text-3xl">{result.score / result.total >= 0.6 ? '🎉' : '📚'}</span>
              <div>
                <p className="text-white font-bold text-lg">{result.score}/{result.total} correct</p>
                <p className="text-slate-400 text-sm">
                  {result.score / result.total >= 0.6 ? 'Great job! Keep it up.' : 'Review these topics and try again.'}
                </p>
              </div>
            </motion.div>
          )}

          {/* Questions */}
          <div className="flex-1 overflow-y-auto px-6 py-4 space-y-6">
            {questions.map((q, idx) => {
              const selected = responses[q.id]
              const isCorrect = submitted && selected === q.correct_answer
              const isWrong = submitted && selected && selected !== q.correct_answer

              return (
                <div key={q.id} className="space-y-4 pt-4 border-t border-white/5 first:border-0 first:pt-0">
                  <div className="flex flex-wrap gap-2 pl-10 mb-2">
                    {q.chapter && <span className="px-2 py-0.5 rounded bg-brand-500/10 border border-brand-500/20 text-[10px] text-brand-300 font-bold uppercase tracking-wider">{q.chapter}</span>}
                    {q.topic && <span className="px-2 py-0.5 rounded bg-surface-700/50 border border-white/10 text-[10px] text-slate-400 font-medium">{q.topic}</span>}
                  </div>

                  <div className="flex gap-3">
                    <span className="flex-shrink-0 w-7 h-7 bg-brand-500/20 border border-brand-500/30 rounded-lg text-brand-300 text-xs font-bold flex items-center justify-center">
                      {idx + 1}
                    </span>
                    <div className="text-slate-200 text-sm flex-1 leading-relaxed">
                      <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]} 
                        components={{
                          img: ({src, alt}) => <img src={src} alt={alt} className="my-4 rounded-lg border border-white/10 max-w-full h-auto mx-auto" />
                        }}
                      >
                        {q.question}
                      </ReactMarkdown>
                    </div>
                  </div>

                  {/* AI Generated Diagram Placeholder */}
                  {q.needs_diagram && !q.image_url && (
                    <div className="pl-10">
                      <div className="bg-brand-500/5 border border-brand-500/10 rounded-xl p-6 flex flex-col items-center justify-center gap-3 text-center">
                        <span className="text-2xl">🎨</span>
                        <div>
                          <p className="text-xs font-bold text-brand-300 uppercase tracking-tighter">AI Visualization Recommendation</p>
                          <p className="text-xs text-slate-400 max-w-sm mt-1">This question involves a geometric or physical diagram. Our AI is generating a visual guide for you...</p>
                        </div>
                        <div className="w-full max-w-xs h-1.5 bg-brand-500/10 rounded-full overflow-hidden">
                          <motion.div 
                            className="h-full bg-brand-500" 
                            initial={{ width: 0 }} 
                            animate={{ width: "100%" }} 
                            transition={{ duration: 2, repeat: Infinity }} 
                          />
                        </div>
                      </div>
                    </div>
                  )}

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 pl-10">
                    {q.options.map((opt, i) => {
                      const optKey = OPTS[i]
                      const isSelected = selected === optKey
                      const isAnswer = submitted && q.correct_answer === optKey

                      let cls = 'border text-sm px-3 py-2 rounded-xl text-left transition-all duration-200 '
                      if (submitted) {
                        if (isAnswer) cls += 'bg-green-500/15 border-green-400/50 text-green-200'
                        else if (isSelected && !isAnswer) cls += 'bg-red-500/15 border-red-400/50 text-red-300 line-through'
                        else cls += 'bg-surface-700/40 border-white/5 text-slate-500'
                      } else {
                        cls += isSelected
                          ? 'bg-brand-500/20 border-brand-400/50 text-white'
                          : 'bg-surface-700/60 border-white/10 text-slate-300 hover:border-brand-500/40 hover:text-white cursor-pointer'
                      }

                      return (
                        <button
                          key={optKey}
                          id={`q${q.id}-opt-${optKey}`}
                          className={cls}
                          onClick={() => selectOption(q.id, optKey)}
                          disabled={submitted}
                        >
                          <span className="font-semibold mr-2 opacity-60">{optKey}.</span>
                          <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]} components={{ p: ({ children }) => <span>{children}</span> }}>
                            {opt}
                          </ReactMarkdown>
                        </button>
                      )
                    })}
                  </div>

                  {/* Explanation after submit */}
                  {submitted && q.explanation && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      className="pl-10 bg-surface-700/40 border border-white/10 rounded-xl px-4 py-3 text-sm text-slate-300"
                    >
                      <span className="font-semibold text-brand-300">Explanation: </span>
                      <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]} components={{ p: ({ children }) => <span>{children}</span> }}>
                        {q.explanation}
                      </ReactMarkdown>
                    </motion.div>
                  )}
                </div>
              )
            })}
          </div>

          {/* Footer */}
          <div className="px-6 py-4 border-t border-white/10 flex justify-end gap-3">
            {!submitted ? (
              <button
                id="submit-quiz-btn"
                onClick={handleSubmit}
                disabled={!allAnswered || loading}
                className="btn-primary flex items-center gap-2"
              >
                {loading ? <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" /> : null}
                {!allAnswered ? `Answer all ${questions.length - Object.keys(responses).length} remaining` : 'Submit Quiz'}
              </button>
            ) : (
              <button id="close-quiz-btn" onClick={onClose} className="btn-primary">
                Done
              </button>
            )}
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  )
}
