import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import API from '../lib/api'
import toast from 'react-hot-toast'
import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'

export default function OCRPage() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState('')

  const handleFileChange = (e) => {
    const selected = e.target.files[0]
    if (selected) {
      setFile(selected)
      setPreview(URL.createObjectURL(selected))
      setResult('')
    }
  }

  const handleExtract = async () => {
    if (!file) return
    setLoading(true)
    const formData = new FormData()
    formData.append('image', file)

    try {
      const { data } = await API.post('/ocr/extract', formData)
      setResult(data.text)
      toast.success('Text extracted successfully!')
    } catch (err) {
      toast.error('Failed to extract text')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex-1 overflow-y-auto p-6">
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-4xl mx-auto space-y-6"
      >
        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold text-white">📸 Vision OCR</h1>
          <p className="text-slate-400 mt-1">
            Upload an image of a question to extract text and math using Qwen2.5-VL.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Upload Section */}
          <div className="glass-card p-6 flex flex-col items-center justify-center space-y-4 min-h-[400px]">
            {preview ? (
              <div className="relative w-full group">
                <img src={preview} alt="Upload preview" className="w-full rounded-xl border border-white/10" />
                <button 
                  onClick={() => {setPreview(null); setFile(null)}}
                  className="absolute top-2 right-2 bg-black/50 text-white p-2 rounded-full opacity-0 group-hover:opacity-100 transition-opacity"
                >✕</button>
              </div>
            ) : (
              <label className="w-full h-64 border-2 border-dashed border-white/10 rounded-2xl flex flex-col items-center justify-center cursor-pointer hover:border-brand-500/50 hover:bg-brand-500/5 transition-all group">
                <span className="text-4xl group-hover:scale-110 transition-transform">🖼️</span>
                <span className="mt-2 text-slate-400 font-medium">Click to upload image</span>
                <input type="file" className="hidden" accept="image/*" onChange={handleFileChange} />
              </label>
            )}

            <button
              onClick={handleExtract}
              disabled={!file || loading}
              className="btn-primary w-full flex items-center justify-center gap-2 py-4"
            >
              {loading ? <span className="w-5 h-5 border-3 border-white/30 border-t-white rounded-full animate-spin" /> : '🔍'}
              {loading ? 'Processing with Qwen...' : 'Extract Text from Image'}
            </button>
          </div>

          {/* Results Section */}
          <div className="glass-card p-6 flex flex-col space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-bold text-white flex items-center gap-2">
                📋 Extracted Result
              </h2>
              {result && (
                <button 
                  onClick={() => {navigator.clipboard.writeText(result); toast.success('Copied!')}}
                  className="text-xs text-brand-400 font-medium hover:underline"
                >Copy Text</button>
              )}
            </div>

            <div className="flex-1 bg-surface-900/50 rounded-xl border border-white/5 p-4 overflow-y-auto whitespace-pre-wrap text-slate-200 text-sm font-mono min-h-[300px]">
              {result ? (
                 <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>
                   {result}
                 </ReactMarkdown>
              ) : (
                <div className="h-full flex flex-col items-center justify-center text-slate-500 italic">
                  <span>Results will appear here...</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  )
}
