import { useCallback, useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { io, type Socket } from 'socket.io-client'
import './App.css'

export const CHAT_SESSION_STORAGE_KEY = 'chat_session_id'

/** Shown once per new session (no prior session in localStorage). */
const WELCOME_ASSISTANT_MESSAGE = `Hi — welcome!

I'm your **Omnimarket assistant**. I can help you with:

- **Orders** — look up order status, details, and shipping-related questions
- **Returns** — check return eligibility and start a return when your order qualifies
- **General questions** — store policies, product info, and FAQs

Ask me anything in plain language. How can I help you today?`

type ChatMessage = { role: 'user' | 'assistant' | 'system'; content: string }

function ChatBubbleBody({ msg }: { msg: ChatMessage }) {
  if (msg.role === 'assistant') {
    return (
      <div className="chat-msg-body chat-msg-body--markdown">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
      </div>
    )
  }
  return <span className="chat-msg-body">{msg.content}</span>
}

function App() {
  const [connectionKey, setConnectionKey] = useState(0)
  const [status, setStatus] = useState<'connecting' | 'open' | 'closed' | 'error'>(
    'connecting',
  )
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const socketRef = useRef<Socket | null>(null)

  const appendMessage = useCallback((msg: ChatMessage) => {
    setMessages((prev) => [...prev, msg])
  }, [])

  useEffect(() => {
    const stored = localStorage.getItem(CHAT_SESSION_STORAGE_KEY)
    const hadStoredSessionAtConnect = stored !== null
    setStatus('connecting')

    const socket = io({
      path: '/socket.io/',
      transports: ['websocket', 'polling'],
      query: stored ? { session_id: stored } : {},
    })
    socketRef.current = socket

    socket.on('connect', () => {
      setStatus('open')
    })

    socket.on('disconnect', () => {
      setStatus('closed')
    })

    socket.on('connect_error', () => {
      setStatus('error')
    })

    socket.on('session', (data: { type?: string; session_id?: string }) => {
      if (data?.session_id) {
        setSessionId(data.session_id)
        localStorage.setItem(CHAT_SESSION_STORAGE_KEY, data.session_id)
        
        if (!hadStoredSessionAtConnect) {
          appendMessage({ role: 'assistant', content: WELCOME_ASSISTANT_MESSAGE })
        }
      }
    })

    socket.on('assistant', (data: { type?: string; content?: string }) => {
      if (typeof data?.content === 'string') {
        appendMessage({ role: 'assistant', content: data.content })
      }
    })

    socket.on('chat_error', (data: { type?: string; message?: string }) => {
      appendMessage({
        role: 'system',
        content: `Error: ${data?.message ?? 'unknown'}`,
      })
    })

    return () => {
      socket.removeAllListeners()
      socket.disconnect()
      socketRef.current = null
    }
  }, [connectionKey, appendMessage])

  const startNewSession = () => {
    localStorage.removeItem(CHAT_SESSION_STORAGE_KEY)
    setSessionId(null)
    setMessages([])
    socketRef.current?.disconnect()
    setConnectionKey((k) => k + 1)
  }

  const send = () => {
    const text = input.trim()
    const socket = socketRef.current
    if (!text || !socket?.connected) return
    socket.emit('user_message', { content: text })
    appendMessage({ role: 'user', content: text })
    setInput('')
  }

  return (
    <div className="chat-app">
      <header className="chat-header">
        <div className="chat-title">
          <h1>Chat</h1>
          <span className={`chat-status chat-status--${status}`}>{status}</span>
        </div>
        <div className="chat-toolbar">
          {sessionId && (
            <code className="chat-session-id" title="Stored in localStorage">
              {sessionId}
            </code>
          )}
          <button type="button" className="btn btn-primary" onClick={startNewSession}>
            New session
          </button>
        </div>
      </header>

      <ul className="chat-messages" aria-live="polite">
        {messages.map((m, i) => (
          <li key={i} className={`chat-msg chat-msg--${m.role}`}>
            <span className="chat-msg-role">{m.role}</span>
            <ChatBubbleBody msg={m} />
          </li>
        ))}
      </ul>

      <form
        className="chat-input-row"
        onSubmit={(e) => {
          e.preventDefault()
          send()
        }}
      >
        <input
          type="text"
          className="chat-input"
          placeholder="Type a message…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={status !== 'open'}
          autoComplete="off"
        />
        <button type="submit" className="btn btn-send" disabled={status !== 'open'}>
          Send
        </button>
      </form>
    </div>
  )
}

export default App
