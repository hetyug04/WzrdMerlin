import { useState, useEffect } from 'react'
import './App.css'

interface MerlinEvent {
  id: string
  type: string
  timestamp: string
  source_actor: string
  target_actor?: string
  correlation_id: string
  payload: any
}

function App() {
  const [events, setEvents] = useState<MerlinEvent[]>([])
  const [capabilities, setCapabilities] = useState<MerlinEvent[]>([])
  const [taskInput, setTaskInput] = useState("")

  useEffect(() => {
    // Setup SSE connection to the FastAPI bridge
    const eventSource = new EventSource('http://localhost:8000/api/stream')

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as MerlinEvent
        
        // General event log
        setEvents((prev) => [data, ...prev].slice(0, 100)) // Keep last 100 events
        
        // Track capability gaps explicitly
        if (data.type === 'capability.gap' || data.type === 'improvement.queued' || data.type === 'improvement.deployed') {
            setCapabilities((prev) => [data, ...prev].slice(0, 20))
        }

      } catch (err) {
        console.error("Failed to parse event", err)
      }
    }

    return () => eventSource.close()
  }, [])

  const submitTask = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!taskInput.trim()) return

    try {
      await fetch(`http://localhost:8000/api/task?description=${encodeURIComponent(taskInput)}`, {
        method: 'POST'
      })
      setTaskInput("")
    } catch (err) {
      console.error("Failed to submit task", err)
    }
  }

  return (
    <div className="dashboard-container">
      <header className="dashboard-header">
        <h1>WzrdMerlin v2 🧙‍♂️</h1>
        <div className="status-badge">NATS Connected</div>
      </header>

      <div className="main-content">
        {/* Left Column: Task Input & Improvement Queue */}
        <section className="left-panel">
          <div className="panel card">
            <h2>Submit Task</h2>
            <form onSubmit={submitTask} className="task-form">
              <input 
                type="text" 
                value={taskInput} 
                onChange={(e) => setTaskInput(e.target.value)} 
                placeholder="e.g. Write a script to scrape wikipedia..."
              />
              <button type="submit">Dispatch to Router</button>
            </form>
          </div>

          <div className="panel card capability-queue">
            <h2>Emergent Capability Queue</h2>
            {capabilities.length === 0 ? (
              <p className="empty-state">No capability gaps detected.</p>
            ) : (
              <ul>
                {capabilities.map(cap => (
                  <li key={cap.id} className={`cap-item ${cap.type.replace('.', '-')}`}>
                    <strong>{cap.type}</strong>
                    <p>{cap.payload.gap_description || JSON.stringify(cap.payload)}</p>
                    <span className="timestamp">{new Date(cap.timestamp).toLocaleTimeString()}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </section>

        {/* Right Column: Global Event Stream */}
        <section className="right-panel">
          <div className="panel card global-stream">
            <h2>DisCo Event Stream</h2>
            <div className="stream-container">
              {events.map(evt => (
                <div key={evt.id} className="event-row">
                  <span className="time">[{new Date(evt.timestamp).toLocaleTimeString()}]</span>
                  <span className={`actor ${evt.source_actor}`}>{evt.source_actor}</span>
                  <span className="type">{evt.type}</span>
                </div>
              ))}
            </div>
          </div>
        </section>
      </div>
    </div>
  )
}

export default App
