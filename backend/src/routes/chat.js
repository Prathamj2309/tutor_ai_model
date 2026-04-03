const express = require('express')
const multer = require('multer')
const supabase = require('../config/supabase')
const { getConversationHistory, saveUserMessage, saveAssistantMessage } = require('../services/historyService')
const { askAI, extractImageText } = require('../services/aiService')

const router = express.Router()
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 10 * 1024 * 1024 } })

// GET /conversations — list all conversations for the authenticated user
router.get('/conversations', async (req, res) => {
  const { data, error } = await supabase
    .from('conversations')
    .select('*')
    .eq('user_id', req.user.id)
    .order('updated_at', { ascending: false })

  if (error) throw error
  res.json(data)
})

// POST /conversations — create a new conversation
router.post('/conversations', async (req, res) => {
  const { subject = 'general', title } = req.body
  const { data, error } = await supabase
    .from('conversations')
    .insert({ user_id: req.user.id, subject, title: title || `New ${subject} chat` })
    .select()
    .single()

  if (error) throw error
  res.status(201).json(data)
})

// GET /conversations/:id/messages — get all messages in a conversation
router.get('/conversations/:id/messages', async (req, res) => {
  // Verify ownership
  const { data: conv, error: convErr } = await supabase
    .from('conversations')
    .select('id')
    .eq('id', req.params.id)
    .eq('user_id', req.user.id)
    .single()

  if (convErr || !conv) return res.status(404).json({ error: 'Conversation not found' })

  const { data, error } = await supabase
    .from('messages')
    .select('*')
    .eq('conversation_id', req.params.id)
    .order('created_at', { ascending: true })

  if (error) throw error
  res.json(data)
})

// POST /chat — send a message and get an AI response
router.post('/chat', upload.single('image'), async (req, res) => {
  const { conversationId, content } = req.body
  const imageFile = req.file

  if (!conversationId) return res.status(400).json({ error: 'conversationId is required' })
  if (!content?.trim() && !imageFile) return res.status(400).json({ error: 'Content or image required' })

  // Verify conversation ownership
  const { data: conv, error: convErr } = await supabase
    .from('conversations')
    .select('id, subject')
    .eq('id', conversationId)
    .eq('user_id', req.user.id)
    .single()

  if (convErr || !conv) return res.status(404).json({ error: 'Conversation not found' })

  let imageUrl = null
  let imageOcrText = null

  // 1. Upload image to Supabase Storage & extract text via AI
  if (imageFile) {
    const filename = `${req.user.id}/${conversationId}/${Date.now()}-${imageFile.originalname}`
    const { data: upData, error: upErr } = await supabase.storage
      .from('question-images')
      .upload(filename, imageFile.buffer, { contentType: imageFile.mimetype })

    if (!upErr) {
      const { data: urlData } = supabase.storage.from('question-images').getPublicUrl(filename)
      imageUrl = urlData?.publicUrl || null
    }

    // Extract text from image
    try {
      const ocrResult = await extractImageText(imageFile.buffer, imageFile.mimetype)
      imageOcrText = ocrResult.extracted_text || null
    } catch (ocrErr) {
      console.warn('OCR failed, proceeding without image text:', ocrErr.message)
    }
  }

  // 2. Save user message
  const userMessage = await saveUserMessage({
    conversationId,
    userId: req.user.id,
    content: content || null,
    imageUrl,
    imageOcrText,
  })

  // 3. Fetch conversation history for AI context
  const history = await getConversationHistory(conversationId, 5)

  // 4. Build the question for the AI (combine text + OCR)
  const aiQuestion = [content, imageOcrText ? `[From image]: ${imageOcrText}` : '']
    .filter(Boolean)
    .join('\n')
    .trim()

  // 5. Get AI response
  const aiResult = await askAI({ question: aiQuestion, history, subject: conv.subject })

  // 6. Save assistant message
  const aiMessage = await saveAssistantMessage({
    conversationId,
    userId: req.user.id,
    content: aiResult.answer,
    topicTags: aiResult.topic_tags || [],
  })

  res.json({ userMessage, aiMessage })
})

module.exports = router
