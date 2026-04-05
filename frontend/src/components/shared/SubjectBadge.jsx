const SUBJECT_CONFIG = {
  physics:     { label: 'Physics',     color: 'bg-blue-500/15 text-blue-300 border-blue-500/30' },
  chemistry:   { label: 'Chemistry',   color: 'bg-green-500/15 text-green-300 border-green-500/30' },
  mathematics: { label: 'Mathematics', color: 'bg-orange-500/15 text-orange-300 border-orange-500/30' },
  general:     { label: 'General',     color: 'bg-purple-500/15 text-purple-300 border-purple-500/30' },
}

export default function SubjectBadge({ subject, size = 'sm' }) {
  const cfg = SUBJECT_CONFIG[subject] || SUBJECT_CONFIG.general
  const sizeClass = size === 'xs' ? 'text-xs px-1.5 py-0.5' : 'text-xs px-2.5 py-1'

  return (
    <span className={`inline-flex items-center gap-1 border rounded-full font-medium ${cfg.color} ${sizeClass}`}>
      {cfg.label}
    </span>
  )
}
