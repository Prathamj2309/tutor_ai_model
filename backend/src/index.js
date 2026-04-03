require('dotenv').config()
require('express-async-errors')

const express = require('express')
const cors = require('cors')
const authMiddleware = require('./middleware/auth')
const errorHandler = require('./middleware/errorHandler')
const chatRoutes = require('./routes/chat')
const quizRoutes = require('./routes/quiz')
const profileRoutes = require('./routes/profile')

const app = express()
const PORT = process.env.PORT || 4000

// ── Middleware ─────────────────────────────────────────────────────────────────
app.use(cors({
  origin: process.env.FRONTEND_URL || 'http://localhost:5173',
  credentials: true,
}))
app.use(express.json())
app.use(express.urlencoded({ extended: true }))

// ── Health check ───────────────────────────────────────────────────────────────
app.get('/health', (req, res) => res.json({ status: 'ok', time: new Date().toISOString() }))

// ── Protected routes (all require valid Supabase JWT) ──────────────────────────
app.use(authMiddleware)
app.use('/', chatRoutes)
app.use('/', quizRoutes)
app.use('/', profileRoutes)

// ── Global error handler ───────────────────────────────────────────────────────
app.use(errorHandler)

app.listen(PORT, () => {
  console.log(`✅ TutorAI Express backend running on http://localhost:${PORT}`)
})

module.exports = app
