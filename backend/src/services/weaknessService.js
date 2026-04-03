const supabase = require('../config/supabase')

/**
 * Analyze a user's message history to find their weakest topics per subject.
 * Returns: { physics: [{topic, error_rate, attempts}], chemistry: [...], mathematics: [...] }
 */
async function getWeaknessReport(userId, subject = null) {
  let query = supabase
    .from('messages')
    .select('topic_tags, is_correct, created_at')
    .eq('user_id', userId)
    .eq('role', 'user')
    .not('topic_tags', 'is', null)

  const { data: messages, error } = await query
  if (error) throw error

  // Combine with conversation subject info
  const { data: convData, error: convError } = await supabase
    .from('messages')
    .select('topic_tags, is_correct, conversations!inner(subject)')
    .eq('user_id', userId)
    .eq('role', 'assistant')
    .not('topic_tags', 'is', null)
    .order('created_at', { ascending: false })
    .limit(200)

  if (convError) throw convError

  // Aggregate: { [subject]: { [topic]: { attempts, errors } } }
  const stats = { physics: {}, chemistry: {}, mathematics: {} }

  for (const msg of convData || []) {
    const subj = msg.conversations?.subject || 'general'
    if (!stats[subj]) continue

    for (const tag of msg.topic_tags || []) {
      if (!stats[subj][tag]) stats[subj][tag] = { attempts: 0, errors: 0 }
      stats[subj][tag].attempts++
      if (msg.is_correct === false) stats[subj][tag].errors++
    }
  }

  // Convert to sorted array of weak topics (error_rate >= 0.3 and attempts >= 2)
  const result = {}
  for (const [subj, topics] of Object.entries(stats)) {
    if (subject && subj !== subject) continue
    result[subj] = Object.entries(topics)
      .filter(([_, v]) => v.attempts >= 2)
      .map(([topic, v]) => ({
        topic,
        attempts: v.attempts,
        error_rate: v.attempts > 0 ? v.errors / v.attempts : 0,
      }))
      .sort((a, b) => b.error_rate - a.error_rate)
  }

  return result
}

/**
 * Get the top N weakest topic names for a subject (to pass to quiz generator)
 */
async function getTopWeakTopics(userId, subject, n = 3) {
  const report = await getWeaknessReport(userId, subject)
  const topics = report[subject] || []

  if (topics.length === 0) {
    // Fallback: pick general topics for the subject
    const fallbacks = {
      physics: ['kinematics', 'thermodynamics', 'electrostatics'],
      chemistry: ['chemical-equilibrium', 'coordination-compounds', 'electrochemistry'],
      mathematics: ['integration', 'probability', 'matrices'],
    }
    return fallbacks[subject] || []
  }

  return topics.slice(0, n).map((t) => t.topic)
}

module.exports = { getWeaknessReport, getTopWeakTopics }
