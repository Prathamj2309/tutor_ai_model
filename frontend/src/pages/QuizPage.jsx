import { useState } from 'react'
import { motion } from 'framer-motion'
import { useWeakness } from '../hooks/useWeakness'
import QuizModal from '../components/quiz/QuizModal'
import SubjectBadge from '../components/shared/SubjectBadge'

const SUBJECTS = ['physics', 'chemistry', 'mathematics']

export default function QuizPage() {
  const { quiz, quizLoading, generateQuiz, submitQuiz } = useWeakness()
  const [selectedSubject, setSelectedSubject] = useState('physics')
  const [numQuestions, setNumQuestions] = useState(5)
  const [timeLimit, setTimeLimit] = useState(0) // 0 means Auto (1 min/q)
  const [showModal, setShowModal] = useState(false)
  const [submitLoading, setSubmitLoading] = useState(false)

  const handleGenerate = async () => {
    const result = await generateQuiz(selectedSubject, numQuestions, timeLimit || undefined)
    if (result) setShowModal(true)
  }

  const handleSubmit = async (quizId, responses) => {
    setSubmitLoading(true)
    const result = await submitQuiz(quizId, responses)
    setSubmitLoading(false)
    return result
  }

  return (
    <div className="flex-1 overflow-y-auto p-6">
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-2xl mx-auto space-y-6"
      >
        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold text-white">📝 Mock Test Generator</h1>
          <p className="text-slate-400 mt-1">
            Generate a personalized 5-question quiz based on your weak topics.
          </p>
        </div>

        {/* Number of Questions select */}
        <div className="glass-card p-6 space-y-4">
          <h2 className="text-base font-semibold text-white">Number of Questions</h2>
          <div className="grid grid-cols-4 gap-3">
            {[5, 10, 15, 20].map((n) => (
              <button
                key={n}
                onClick={() => setNumQuestions(n)}
                className={`flex flex-col items-center justify-center p-3 rounded-xl border transition-all duration-200 ${
                  numQuestions === n
                    ? 'bg-brand-500/20 border-brand-500/50 text-white'
                    : 'bg-surface-700/40 border-white/10 text-slate-400 hover:border-white/20 hover:text-white'
                }`}
              >
                <span className="text-sm font-bold">{n} Qs</span>
              </button>
            ))}
          </div>
        </div>

        {/* Time Limit select */}
        <div className="glass-card p-6 space-y-4">
          <h2 className="text-base font-semibold text-white">Time Limit</h2>
          <div className="grid grid-cols-5 gap-3">
            {[0, 15, 30, 60, 90].map((t) => (
              <button
                key={t}
                onClick={() => setTimeLimit(t)}
                className={`flex flex-col items-center justify-center p-3 rounded-xl border transition-all duration-200 ${
                  timeLimit === t
                    ? 'bg-brand-500/20 border-brand-500/50 text-white'
                    : 'bg-surface-700/40 border-white/10 text-slate-400 hover:border-white/20 hover:text-white'
                }`}
              >
                <span className="text-sm font-bold">{t === 0 ? 'Auto' : `${t}m`}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Subject select */}
        <div className="glass-card p-6 space-y-4">
          <h2 className="text-base font-semibold text-white">Select Subject</h2>
          <div className="grid grid-cols-3 gap-3">
            {SUBJECTS.map((s) => (
              <button
                key={s}
                id={`quiz-subject-${s}`}
                onClick={() => setSelectedSubject(s)}
                className={`flex flex-col items-center gap-2 p-4 rounded-xl border transition-all duration-200 ${
                  selectedSubject === s
                    ? 'bg-brand-500/20 border-brand-500/50 text-white'
                    : 'bg-surface-700/40 border-white/10 text-slate-400 hover:border-white/20 hover:text-white'
                }`}
              >
                <span className="text-2xl">{s === 'physics' ? '⚛️' : s === 'chemistry' ? '🧪' : '📐'}</span>
                <span className="text-sm font-medium capitalize">{s}</span>
              </button>
            ))}
          </div>

          <button
            id="generate-quiz-btn"
            onClick={handleGenerate}
            disabled={quizLoading}
            className="btn-primary w-full flex items-center justify-center gap-2"
          >
            {quizLoading ? (
              <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            ) : '⚡'}
            {quizLoading ? 'Generating quiz...' : 'Generate Personalized Quiz'}
          </button>
        </div>

        {/* How it works */}
        <div className="glass-card p-6 space-y-4">
          <h2 className="text-base font-semibold text-white">How it works</h2>
          <div className="space-y-3">
            {[
              { icon: '💬', text: 'Real JEE questions selected from our 150k+ database based on your subject' },
              { icon: '🎨', text: 'AI identifies and generates missing diagrams/charts for complex physics & math' },
              { icon: '⏱️', text: 'Extended time limits (up to 90m) to simulate real exam pressure' },
              { icon: '📈', text: 'View detailed chapter-wise analysis and LaTeX explanations' },
            ].map(({ icon, text }) => (
              <div key={text} className="flex items-start gap-3">
                <span className="text-xl flex-shrink-0">{icon}</span>
                <p className="text-slate-400 text-sm">{text}</p>
              </div>
            ))}
          </div>
        </div>
      </motion.div>

      {/* Quiz Modal */}
      {showModal && quiz && (
        <QuizModal
          quiz={quiz}
          onSubmit={handleSubmit}
          onClose={() => setShowModal(false)}
          loading={submitLoading}
        />
      )}
    </div>
  )
}
