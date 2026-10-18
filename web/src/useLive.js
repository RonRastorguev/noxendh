import { useEffect, useRef, useState } from 'react'

// The server's WS /live stream (scan_started, scan_progress, event_created, scan_done, scan_error, baseline_set,
// event_updated). Reconnects with backoff; `connected` drives the header pill.
export function useLive(onMessage) {
  const [connected, setConnected] = useState(false)
  const handler = useRef(onMessage)
  handler.current = onMessage

  useEffect(() => {
    let ws
    let stopped = false
    let retry = 500
    let ping
    const connect = () => {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws'
      ws = new WebSocket(`${proto}://${location.host}/live`)
      ws.onopen = () => {
        setConnected(true)
        retry = 500
        ping = setInterval(() => ws.readyState === 1 && ws.send('ping'), 15000)
      }
      ws.onmessage = (e) => {
        try { handler.current(JSON.parse(e.data)) } catch { /* ignore malformed */ }
      }
      ws.onclose = () => {
        setConnected(false)
        clearInterval(ping)
        if (!stopped) setTimeout(connect, (retry = Math.min(retry * 2, 8000)))
      }
    }
    connect()
    return () => { stopped = true; clearInterval(ping); ws && ws.close() }
  }, [])

  return connected
}
