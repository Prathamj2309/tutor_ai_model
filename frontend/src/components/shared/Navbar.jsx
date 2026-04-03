import { Link, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'
import { useAuth } from '../../hooks/useAuth'
import useAuthStore from '../../store/authStore'

const NAV_ITEMS = [
  { to: '/dashboard', icon: '⚡', label: 'Dashboard' },
  { to: '/chat',      icon: '💬', label: 'Ask AI' },
  { to: '/quiz',      icon: '📝', label: 'Mock Test' },
]

export default function Navbar() {
  const location = useLocation()
  const { signOut } = useAuth()
  const user = useAuthStore((s) => s.user)

  return (
    <header className="h-16 bg-surface-800/90 backdrop-blur-md border-b border-white/8 flex items-center px-4 gap-4 z-50">
      {/* Logo */}
      <Link to="/dashboard" className="flex items-center gap-2.5 flex-shrink-0">
        <span className="text-xl">🎓</span>
        <span className="font-bold text-white text-lg hidden sm:block">TutorAI</span>
      </Link>

      {/* Nav links */}
      <nav className="flex-1 flex items-center gap-1 ml-4">
        {NAV_ITEMS.map(({ to, icon, label }) => {
          const active = location.pathname.startsWith(to)
          return (
            <Link key={to} to={to} className="relative">
              <motion.div
                className={`flex items-center gap-1.5 px-3 py-2 rounded-xl text-sm font-medium transition-colors ${
                  active ? 'text-white' : 'text-slate-400 hover:text-white hover:bg-white/5'
                }`}
              >
                <span>{icon}</span>
                <span className="hidden sm:block">{label}</span>
              </motion.div>
              {active && (
                <motion.div
                  layoutId="nav-indicator"
                  className="absolute inset-0 bg-brand-500/15 border border-brand-500/30 rounded-xl"
                  style={{ zIndex: -1 }}
                  transition={{ type: 'spring', stiffness: 400, damping: 30 }}
                />
              )}
            </Link>
          )
        })}
      </nav>

      {/* User */}
      <div className="flex items-center gap-3">
        <div className="hidden sm:block text-right">
          <p className="text-sm font-medium text-slate-200">{user?.user_metadata?.full_name || 'Student'}</p>
          <p className="text-xs text-slate-500 truncate max-w-[140px]">{user?.email}</p>
        </div>
        <button
          id="signout-btn"
          onClick={signOut}
          className="btn-ghost text-xs px-3 py-2"
          title="Sign out"
        >
          Sign Out
        </button>
      </div>
    </header>
  )
}
