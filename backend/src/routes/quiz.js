const express = require('express')
const supabase = require('../config/supabase')
const { getTopWeakTopics } = require('../services/weaknessService')
const { generateQuizFromAI } = require('../services/aiService')

const router = express.Router()

// POST /quiz/generate — generate a personalized quiz based on weak topics
router.post('/quiz/generate', async (req, res) => {
  const { subject = 'physics' } = req.body

  // 1. Find weak topics
  const weakTopics = await getTopWeakTopics(req.user.id, subject, 3)

  // 2. Request quiz from AI service
  const quizData = await generateQuizFromAI(weakTopics, subject)

  // 3. Save quiz attempt to Supabase
  const { data: attempt, error } = await supabase
    .from('quiz_attempts')
    .insert({
      user_id: req.user.id,
      subject,
      weak_topics: weakTopics,
      questions: quizData.questions,
      total: quizData.questions.length,
    })
    .select()
    .single()

  if (error) throw error

  res.status(201).json({
    id: attempt.id,
    subject,
    weakTopics,
    questions: quizData.questions,
  })
})

// POST /quiz/submit — submit quiz answers and calculate score
router.post('/quiz/submit', async (req, res) => {
  const { quizId, responses } = req.body

  if (!quizId || !responses) {
    return res.status(400).json({ error: 'quizId and responses are required' })
  }

  // Verify ownership
  const { data: attempt, error: fetchErr } = await supabase
    .from('quiz_attempts')
    .select('*')
    .eq('id', quizId)
    .eq('user_id', req.user.id)
    .single()

  if (fetchErr || !attempt) return res.status(404).json({ error: 'Quiz not found' })
  if (attempt.completed_at) return res.status(400).json({ error: 'Quiz already submitted' })

  // Calculate score
  const questions = attempt.questions
  let score = 0
  const detailedResponses = []

  for (const q of questions) {
    const selected = responses[q.id]
    const isCorrect = selected === q.correct_answer
    if (isCorrect) score++
    detailedResponses.push({ question_id: q.id, selected, is_correct: isCorrect })

    // Mark messages with this topic as correct/incorrect for weakness tracking
    // (This is a simplified heuristic based on quiz performance)
  }

  // Save results
  const { data: updated, error: updateErr } = await supabase
    .from('quiz_attempts')
    .update({
      responses: detailedResponses,
      score,
      completed_at: new Date().toISOString(),
    })
    .eq('id', quizId)
    .select()
    .single()

  if (updateErr) throw updateErr

  res.json({
    score,
    total: attempt.total,
    responses: detailedResponses,
  })
})

// GET /quiz/history — get all past quiz attempts
router.get('/quiz/history', async (req, res) => {
  const { data, error } = await supabase
    .from('quiz_attempts')
    .select('id, subject, weak_topics, score, total, completed_at, created_at')
    .eq('user_id', req.user.id)
    .order('created_at', { ascending: false })

  if (error) throw error
  res.json(data)
})

module.exports = router
