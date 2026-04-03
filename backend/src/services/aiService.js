const axios = require('axios')
require('dotenv').config()

const AI_SERVICE_URL = process.env.AI_SERVICE_URL || 'http://localhost:8000'

const client = axios.create({
  baseURL: AI_SERVICE_URL,
  timeout: 120000, // 2 minutes for LLM inference
})

/**
 * Send a chat message with history to the Python AI service.
 * @param {Object} params
 * @param {string} params.question - The current user question (text only)
 * @param {Array}  params.history  - [{role, content}] array of past messages
 * @param {string} params.subject  - 'physics' | 'chemistry' | 'mathematics' | 'general'
 * @returns {{ answer: string, topic_tags: string[] }}
 */
async function askAI({ question, history = [], subject = 'general' }) {
  const { data } = await client.post('/ai/chat', { question, history, subject })
  return data
}

/**
 * Send an image to the AI service for OCR/extraction.
 * @param {Buffer} imageBuffer
 * @param {string} mimetype
 * @returns {{ extracted_text: string }}
 */
async function extractImageText(imageBuffer, mimetype) {
  const FormData = require('form-data')
  const form = new FormData()
  form.append('file', imageBuffer, { filename: 'upload.jpg', contentType: mimetype })

  const { data } = await client.post('/ai/ocr', form, {
    headers: form.getHeaders(),
  })
  return data
}

/**
 * Request the AI service to generate a quiz for the given topics.
 * @param {string[]} weakTopics
 * @param {string}  subject
 * @returns {{ questions: Array }}
 */
async function generateQuizFromAI(weakTopics, subject) {
  const { data } = await client.post('/ai/quiz/generate', { weak_topics: weakTopics, subject })
  return data
}

module.exports = { askAI, extractImageText, generateQuizFromAI }
