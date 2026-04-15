import { useState, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import API from '../lib/api'
import toast from 'react-hot-toast'
import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'

// Parse legacy "Subject: X\nQuestion: Y" format as fallback only
function parseModelOutput(raw = '') {
  let subject = ''
  let question = ''
  const lines = raw.split('\n')
  const questionLines = []
  let inQuestion = false

  for (const line of lines) {
    const lower = line.trim().toLowerCase()
    if (lower.startsWith('subject:')) {
      subject = line.split(':').slice(1).join(':').trim()
      inQuestion = false
    } else if (lower.startsWith('question:')) {
      question = line.split(':').slice(1).join(':').trim()
      inQuestion = true
    } else if (inQuestion && line.trim()) {
      questionLines.push(line.trim())
    }
  }
  if (questionLines.length) question += '\n' + questionLines.join('\n')
  const sl = subject.toLowerCase()
  if (sl.includes('math')) subject = 'Mathematics'
  else if (sl.includes('phys')) subject = 'Physics'
  else if (sl.includes('chem')) subject = 'Chemistry'
  return { subject: subject || null, question: question || raw }
}

const SUBJECT_COLORS = {
  Physics:     'from-blue-500/20 to-cyan-500/10 border-blue-500/30 text-blue-300',
  Chemistry:   'from-green-500/20 to-emerald-500/10 border-green-500/30 text-green-300',
  Mathematics: 'from-purple-500/20 to-violet-500/10 border-purple-500/30 text-purple-300',
}
const SUBJECT_ICONS = { Physics: '⚛️', Chemistry: '🧪', Mathematics: '∑' }

const ACCEPTED_TYPES = ['image/png', 'image/jpeg', 'image/jpg', 'image/bmp', 'image/webp']

export default function OCRPage() {
  const navigate  = useNavigate()
  const fileInput = useRef(null)

  const [file,      setFile]      = useState(null)
  const [preview,   setPreview]   = useState(null)
  const [dragging,  setDragging]  = useState(false)
  const [loading,   setLoading]   = useState(false)
  const [rawResult, setRawResult] = useState('')
  const [parsed,    setParsed]    = useState(null)   // { subject, question }
  const [tab,       setTab]       = useState('parsed') // 'parsed' | 'raw'

  // ── File selection / validation ───────────────────────────────────────────
  const applyFile = useCallback((f) => {
    if (!f) return
    if (!ACCEPTED_TYPES.includes(f.type)) {
      toast.error('Unsupported file type. Use PNG, JPG, BMP or WebP.')
      return
    }
    if (f.size > 20 * 1024 * 1024) {
      toast.error('File too large (max 20 MB).')
      return
    }
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setRawResult('')
    setParsed(null)
    setTab('parsed')
  }, [])

  const handleFileChange = (e) => applyFile(e.target.files[0])

  // ── Drag & Drop ───────────────────────────────────────────────────────────
  const onDragOver  = (e) => { e.preventDefault(); setDragging(true)  }
  const onDragLeave = ()  => setDragging(false)
  const onDrop      = (e) => {
    e.preventDefault(); setDragging(false)
    applyFile(e.dataTransfer.files[0])
  }

  const clearImage = () => {
    setFile(null); setPreview(null)
    setRawResult(''); setParsed(null)
    if (fileInput.current) fileInput.current.value = ''
  }

  // ── API call ──────────────────────────────────────────────────────────────
  const handleExtract = async () => {
    if (!file) return
    setLoading(true)
    try {
      const formData = new FormData()
      formData.append('image', file)
      const { data } = await API.post('/ocr/extract', formData)
      const raw = data.text || ''
      setRawResult(raw)
      // Use server-parsed subject/question if available, else fall back to client parse
      if (data.subject || data.question) {
        setParsed({ subject: data.subject || null, question: data.question || raw })
      } else {
        setParsed(parseModelOutput(raw))
      }
      toast.success('Extraction complete!')
    } catch {
      toast.error('Failed to extract text. Make sure the backend is running.')
    } finally {
      setLoading(false)
    }
  }

  // ── Copy helpers ──────────────────────────────────────────────────────────
  const copyText = (text) => {
    navigator.clipboard.writeText(text)
    toast.success('Copied to clipboard!')
  }

  const sendToChat = () => {
    const text = parsed?.question || rawResult
    sessionStorage.setItem('ocr_prefill', text)
    toast.success('Opening chat with extracted question…')
    navigate('/chat')
  }

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="flex-1 overflow-y-auto p-6">
      <motion.div
        initial={{ opacity: 0, y: 14 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="max-w-5xl mx-auto space-y-6"
      >
        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold text-white flex items-center gap-3">
            <span className="text-4xl">📸</span> Vision OCR
          </h1>
          <p className="text-slate-400 mt-1 text-sm">
            Upload a JEE question image — Qwen2.5-VL will identify the subject, extract the question and describe any diagrams.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

          {/* ── Left: Upload panel ── */}
          <div className="glass-card p-6 flex flex-col gap-4">
            <h2 className="font-semibold text-white text-base flex items-center gap-2">
              🖼️ Image Upload
            </h2>

            {/* Drop zone / Preview */}
            {preview ? (
              <div className="relative group rounded-xl overflow-hidden border border-white/10">
                <img
                  src={preview}
                  alt="Preview"
                  className="w-full object-contain max-h-72 bg-black/30"
                />
                <button
                  onClick={clearImage}
                  className="absolute top-2 right-2 bg-black/60 hover:bg-red-500/80 text-white rounded-full w-7 h-7 flex items-center justify-center text-sm opacity-0 group-hover:opacity-100 transition-all"
                >✕</button>
                <div className="absolute bottom-0 left-0 right-0 bg-black/50 text-xs text-slate-300 px-3 py-1.5 truncate">
                  {file?.name} · {(file?.size / 1024).toFixed(1)} KB
                </div>
              </div>
            ) : (
              <label
                onDragOver={onDragOver}
                onDragLeave={onDragLeave}
                onDrop={onDrop}
                className={`
                  flex flex-col items-center justify-center h-56 rounded-2xl border-2 border-dashed cursor-pointer
                  transition-all duration-200
                  ${dragging
                    ? 'border-brand-400 bg-brand-500/10 scale-[1.01]'
                    : 'border-white/10 hover:border-brand-500/50 hover:bg-brand-500/5'
                  }
                `}
              >
                <span className={`text-5xl mb-3 transition-transform ${dragging ? 'scale-125' : 'group-hover:scale-110'}`}>
                  {dragging ? '📂' : '🖼️'}
                </span>
                <p className="text-slate-300 font-medium text-sm">
                  {dragging ? 'Drop it here!' : 'Drag & drop or click to upload'}
                </p>
                <p className="text-slate-500 text-xs mt-1">PNG · JPG · BMP · WebP · max 20 MB</p>
                <input
                  ref={fileInput}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={handleFileChange}
                />
              </label>
            )}

            {/* Steps info */}
            <div className="bg-surface-900/50 rounded-xl border border-white/5 p-3 space-y-1 text-xs text-slate-400">
              <p className="font-semibold text-slate-300 mb-1.5">What the model does:</p>
              {[
                '1. Identifies subject — Physics / Chemistry / Math',
                '2. Extracts the full question with LaTeX math',
                '3. Describes diagrams, circuits, or figures in detail',
                '4. Uses IUPAC naming for chemical compounds',
              ].map((s) => (
                <p key={s} className="flex items-start gap-1.5">
                  <span className="text-brand-400 mt-0.5">›</span> {s}
                </p>
              ))}
            </div>

            {/* Extract button */}
            <button
              onClick={handleExtract}
              disabled={!file || loading}
              className="btn-primary w-full flex items-center justify-center gap-2 py-3.5 text-sm"
            >
              {loading
                ? <><span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" /> Processing with Qwen2.5-VL…</>
                : <><span>🔍</span> Extract Text from Image</>
              }
            </button>
          </div>

          {/* ── Right: Results panel ── */}
          <div className="glass-card p-6 flex flex-col gap-4">

            {/* Tabs + Copy */}
            <div className="flex items-center justify-between">
              <div className="flex gap-1 bg-surface-900/60 rounded-xl p-1">
                {['parsed', 'raw'].map((t) => (
                  <button
                    key={t}
                    onClick={() => setTab(t)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                      tab === t
                        ? 'bg-brand-500 text-white shadow'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    {t === 'parsed' ? '📋 Parsed' : '📄 Raw'}
                  </button>
                ))}
              </div>
              {rawResult && (
                <div className="flex gap-2">
                  <button
                    onClick={() => copyText(tab === 'parsed' ? (parsed?.question || rawResult) : rawResult)}
                    className="text-xs text-brand-400 hover:text-brand-300 font-medium transition-colors"
                  >
                    Copy
                  </button>
                  <button
                    onClick={sendToChat}
                    className="text-xs bg-brand-500/20 hover:bg-brand-500/30 text-brand-300 px-2.5 py-1 rounded-lg font-medium transition-colors"
                    title="Send extracted question to chat"
                  >
                    💬 Send to Chat
                  </button>
                </div>
              )}
            </div>

            {/* Subject badge */}
            <AnimatePresence>
              {parsed?.subject && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0 }}
                  className={`flex items-center gap-2 px-4 py-2.5 rounded-xl border bg-gradient-to-r text-sm font-semibold ${SUBJECT_COLORS[parsed.subject] || 'from-slate-500/20 to-slate-600/10 border-slate-500/30 text-slate-300'}`}
                >
                  <span>{SUBJECT_ICONS[parsed.subject] || '📚'}</span>
                  Subject Detected: {parsed.subject}
                </motion.div>
              )}
            </AnimatePresence>

            {/* Content area */}
            <div className="flex-1 bg-surface-900/50 rounded-xl border border-white/5 p-4 overflow-y-auto min-h-[300px] max-h-[420px]">
              <AnimatePresence mode="wait">
                {!rawResult && !loading && (
                  <motion.div
                    key="empty"
                    initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                    className="h-full flex flex-col items-center justify-center text-slate-600 gap-3"
                  >
                    <span className="text-5xl">🔬</span>
                    <p className="text-sm italic">Upload an image and click Extract to begin</p>
                  </motion.div>
                )}

                {loading && (
                  <motion.div
                    key="loading"
                    initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                    className="h-full flex flex-col items-center justify-center gap-4 text-slate-400"
                  >
                    <div className="w-10 h-10 border-[3px] border-brand-500/30 border-t-brand-400 rounded-full animate-spin" />
                    <div className="text-center">
                      <p className="text-sm font-medium text-slate-300">Qwen2.5-VL is analyzing…</p>
                      <p className="text-xs text-slate-500 mt-1">This may take 15–60 seconds</p>
                    </div>
                  </motion.div>
                )}

                {rawResult && !loading && tab === 'parsed' && (
                  <motion.div
                    key="parsed"
                    initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}
                    className="prose prose-invert prose-sm max-w-none text-slate-200 text-sm leading-relaxed"
                  >
                    <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>
                      {parsed?.question || rawResult}
                    </ReactMarkdown>
                  </motion.div>
                )}

                {rawResult && !loading && tab === 'raw' && (
                  <motion.pre
                    key="raw"
                    initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}
                    className="text-xs text-slate-300 font-mono whitespace-pre-wrap break-words"
                  >
                    {rawResult}
                  </motion.pre>
                )}
              </AnimatePresence>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  )
}
