import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import useAuthStore from '../store/authStore'
import API from '../lib/api'
import SubjectBadge from '../components/shared/SubjectBadge'

export default function ProfilePage() {
  const user = useAuthStore((s) => s.user)
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function fetchHistory() {
      try {
        const { data } = await API.get('/quiz/history')
        setHistory(data)
      } catch (err) {
        console.error(err)
      } finally {
        setLoading(false)
      }
    }
    fetchHistory()
  }, [])

  return (
    <div className="flex-1 overflow-y-auto p-8 space-y-10 bg-slate-950">
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="flex items-center gap-6 glass-card p-6 border-white/5">
        <div className="w-24 h-24 rounded-full bg-gradient-to-tr from-brand-500/20 to-purple-500/20 border border-brand-500/30 flex items-center justify-center text-4xl font-bold text-brand-300 shadow-xl">
          {user?.user_metadata?.full_name?.[0]?.toUpperCase() || 'U'}
        </div>
        <div>
          <h1 className="text-4xl font-bold text-white mb-1">{user?.user_metadata?.full_name || 'Student Profile'}</h1>
          <p className="text-slate-400">{user?.email}</p>
        </div>
      </motion.div>

      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
        <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-2">
          <span>📈</span> Mock Test History
        </h2>
        
        {loading ? (
             <div className="w-8 h-8 border-2 border-brand-500/30 border-t-brand-500 rounded-full animate-spin" />
        ) : history.length === 0 ? (
          <div className="text-center py-10 glass-card border-dashed border-slate-700">
            <p className="text-slate-500">No mock tests taken yet. Start practicing to see your progress here!</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {history.map((test) => (
              <div key={test.id} className="glass-card p-6 bg-surface-800/80 backdrop-blur-md border border-white/5 rounded-2xl flex items-center justify-between hover:scale-[1.02] transition-transform">
                <div className="flex-1">
                  <div className="flex gap-3 items-center mb-3">
                    <SubjectBadge subject={test.subject} />
                    <span className="text-xs font-semibold text-slate-500 bg-slate-900/50 px-2 py-1 rounded-md">
                      {new Date(test.created_at).toLocaleDateString()}
                    </span>
                  </div>
                  <p className="text-sm text-slate-300 pr-4">
                    <span className="text-slate-500">Focus:</span> {test.weak_topics?.join(', ') || 'General topics'}
                  </p>
                </div>
                <div className="text-right pl-4 border-l border-white/10 shrink-0">
                  <p className="text-3xl font-black text-white">{test.score} <span className="text-lg text-slate-500 font-normal">/ {test.total}</span></p>
                  <p className="text-xs uppercase tracking-wider text-green-400 font-bold mt-1">Score</p>
                </div>
              </div>
            ))}
          </div>
        )}
      </motion.div>
    </div>
  )
}
