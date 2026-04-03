const supabase = require('../config/supabase')

/**
 * Fetch the last N messages from a conversation for AI context.
 * Returns messages sorted oldest-first.
 */
async function getConversationHistory(conversationId, limit = 5) {
  const { data, error } = await supabase
    .from('messages')
    .select('role, content, image_ocr_text, created_at')
    .eq('conversation_id', conversationId)
    .order('created_at', { ascending: false })
    .limit(limit)

  if (error) throw error

  // Reverse to get chronological order
  return (data || []).reverse().map((m) => ({
    role: m.role,
    content: m.image_ocr_text
      ? `${m.content || ''}\n[Image content: ${m.image_ocr_text}]`.trim()
      : m.content || '',
  }))
}

/**
 * Save a user message to Supabase.
 */
async function saveUserMessage({ conversationId, userId, content, imageUrl, imageOcrText }) {
  const { data, error } = await supabase
    .from('messages')
    .insert({
      conversation_id: conversationId,
      user_id: userId,
      role: 'user',
      content: content || null,
      image_url: imageUrl || null,
      image_ocr_text: imageOcrText || null,
    })
    .select()
    .single()

  if (error) throw error
  return data
}

/**
 * Save the AI assistant response to Supabase.
 */
async function saveAssistantMessage({ conversationId, userId, content, topicTags }) {
  const { data, error } = await supabase
    .from('messages')
    .insert({
      conversation_id: conversationId,
      user_id: userId,
      role: 'assistant',
      content,
      topic_tags: topicTags || [],
    })
    .select()
    .single()

  if (error) throw error

  // Update conversation timestamp
  await supabase
    .from('conversations')
    .update({ updated_at: new Date().toISOString() })
    .eq('id', conversationId)

  return data
}

module.exports = { getConversationHistory, saveUserMessage, saveAssistantMessage }
