import { useEffect } from 'react'
import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'
import { useWeakness } from '../hooks/useWeakness'
import useAuthStore from '../store/authStore'
import SubjectBadge from '../components/shared/SubjectBadge'

const SUBJECTS = ['physics', 'chemistry', 'mathematics']

export default function Dashboard() {
  const user = useAuthStore((s) => s.user)
  const { report, loading, fetchWeaknessReport } = useWeakness()

  useEffect(() => {
    fetchWeaknessReport()
  }, [])

  const name = user?.user_metadata?.full_name?.split(' ')[0] || 'Student'
  const hour = new Date().getHours()
  const greeting = hour < 12 ? 'Good morning' : hour < 17 ? 'Good afternoon' : 'Good evening'

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-8">
      {/* Welcome */}
      <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-3xl font-bold text-white">
              {greeting}, {name}! 👋
            </h1>
            <p className="text-slate-400 mt-1">Here's your learning summary for today.</p>
          </div>
        </div>
      </motion.div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {[
          { to: '/chat',  icon: '💬', title: 'Ask a Doubt',   desc: 'Get step-by-step solutions', color: 'from-brand-500/20 to-brand-600/10 border-brand-500/30' },
          { to: '/quiz',  icon: '📝', title: 'Take Mock Test', desc: 'Test your weak topics',       color: 'from-orange-500/20 to-orange-600/10 border-orange-500/30' },
          { to: '/chat',  icon: '📷', title: 'Image Scan',    desc: 'Upload a photo of a problem',  color: 'from-purple-500/20 to-purple-600/10 border-purple-500/30' },
        ].map(({ to, icon, title, desc, color }, i) => (
          <motion.div
            key={title}
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.1 + 0.15 }}
          >
            <Link
              to={to}
              id={`action-${title.toLowerCase().replace(/\s+/g, '-')}`}
              className={`glass-card bg-gradient-to-br ${color} p-5 flex items-start gap-4 hover:scale-[1.02] transition-transform duration-200 group block`}
            >
              <span className="text-3xl group-hover:scale-110 transition-transform">{icon}</span>
              <div>
                <p className="font-semibold text-white">{title}</p>
                <p className="text-slate-400 text-sm mt-0.5">{desc}</p>
              </div>
            </Link>
          </motion.div>
        ))}
      </div>

      {/* Weakness Report */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="glass-card p-6"
      >
        <div className="flex items-center justify-between mb-5">
          <h2 className="text-lg font-bold text-white">📊 Weak Topics Tracker</h2>
          <Link to="/quiz" className="text-brand-400 hover:text-brand-300 text-sm font-medium transition-colors">
            Generate Test →
          </Link>
        </div>

        {loading ? (
          <div className="flex justify-center py-6">
            <div className="w-6 h-6 border-2 border-brand-500/30 border-t-brand-500 rounded-full animate-spin" />
          </div>
        ) : (!report || Object.values(report).every((v) => v.length === 0)) ? (
          <div className="text-center py-8">
            <p className="text-slate-400 text-sm">Start asking questions to see your weak topics here.</p>
          </div>
        ) : (
          <div className="space-y-4">
            {SUBJECTS.map((subj) => {
              const topics = report[subj] || []
              if (topics.length === 0) return null
              return (
                <div key={subj}>
                  <div className="flex items-center gap-2 mb-3">
                    <SubjectBadge subject={subj} />
                  </div>
                  <div className="space-y-2">
                    {topics.slice(0, 5).map((t) => (
                      <div key={t.topic} className="flex items-center gap-3">
                        <span className="text-sm text-slate-300 flex-1 truncate capitalize">{t.topic.replace(/-/g, ' ')}</span>
                        <div className="flex-1 max-w-[180px] h-1.5 bg-surface-700 rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full bg-gradient-to-r from-orange-500 to-red-500"
                            style={{ width: `${Math.round(t.error_rate * 100)}%` }}
                          />
                        </div>
                        <span className="text-xs text-slate-500 w-10 text-right">{Math.round(t.error_rate * 100)}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </motion.div>
    </div>
  )
}
