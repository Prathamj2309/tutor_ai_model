import { useState, useCallback } from 'react'
import API from '../lib/api'
import toast from 'react-hot-toast'

export function useWeakness() {
  const [report, setReport] = useState(null)
  const [quiz, setQuiz] = useState(null)
  const [loading, setLoading] = useState(false)
  const [quizLoading, setQuizLoading] = useState(false)

  const fetchWeaknessReport = useCallback(async (subject) => {
    setLoading(true)
    try {
      const params = subject ? { subject } : {}
      const { data } = await API.get('/weakness-report', { params })
      setReport(data)
    } catch (err) {
      toast.error('Failed to load weakness report')
    } finally {
      setLoading(false)
    }
  }, [])

  const generateQuiz = useCallback(async (subject, numQuestions, timeLimit) => {
    setQuizLoading(true)
    try {
      const { data } = await API.post('/quiz/generate', { subject, numQuestions, timeLimit })
      setQuiz(data)
      return data
    } catch (err) {
      toast.error('Failed to generate quiz')
    } finally {
      setQuizLoading(false)
    }
  }, [])

  const submitQuiz = useCallback(async (quizId, responses) => {
    try {
      const { data } = await API.post('/quiz/submit', { quizId, responses })
      return data
    } catch (err) {
      toast.error('Failed to submit quiz')
    }
  }, [])

  return { report, quiz, loading, quizLoading, fetchWeaknessReport, generateQuiz, submitQuiz }
}
