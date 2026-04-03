const express = require('express')
const supabase = require('../config/supabase')
const { getWeaknessReport } = require('../services/weaknessService')

const router = express.Router()

// GET /profile — get the current user's profile
router.get('/profile', async (req, res) => {
  const { data, error } = await supabase
    .from('profiles')
    .select('*')
    .eq('id', req.user.id)
    .single()

  if (error) return res.status(404).json({ error: 'Profile not found' })
  res.json(data)
})

// PATCH /profile — update profile details
router.patch('/profile', async (req, res) => {
  const { username, full_name, grade } = req.body
  const updates = {}
  if (username !== undefined) updates.username = username
  if (full_name !== undefined) updates.full_name = full_name
  if (grade !== undefined) updates.grade = grade
  updates.updated_at = new Date().toISOString()

  const { data, error } = await supabase
    .from('profiles')
    .update(updates)
    .eq('id', req.user.id)
    .select()
    .single()

  if (error) throw error
  res.json(data)
})

// GET /weakness-report — get the user's weakness analysis
router.get('/weakness-report', async (req, res) => {
  const { subject } = req.query
  const report = await getWeaknessReport(req.user.id, subject || null)
  res.json(report)
})

module.exports = router
