import { useCallback, useEffect, useRef, useState } from 'react'

export type Card = string

export interface SeatState {
  stack: number
  street_bet: number
  total_bet: number
  folded: boolean
  all_in: boolean
  shown_cards: Card[] | null
}

export interface HandEvent {
  type: 'post_blind' | 'action' | 'deal_board' | 'return_uncalled' | 'showdown' | 'award'
  street: string
  seat: number | null
  action: string | null
  amount: number
  cards: Card[]
}

export interface HandView {
  hole_cards: Card[]
  board: Card[]
  street: string
  button: number
  pot: number
  current_bet: number
  to_act: number | null
  is_over: boolean
  seats: Record<string, SeatState>
  events: HandEvent[]
  awards: Record<string, number>
  hand_names: Record<string, string>
}

export interface Legal {
  can_fold: boolean
  can_check: boolean
  call_amount: number
  raise_type: 'bet' | 'raise' | null
  min_raise_to: number | null
  max_raise_to: number | null
}

export interface TableSeat {
  seat: number
  name: string
  kind: 'human' | 'ai'
  stack: number
  away: boolean
  in_hand: boolean
}

export interface Snapshot {
  type: 'state'
  game_id: number
  status: 'idle' | 'playing' | 'hand_over'
  hand_no: number
  human_seat: number
  small_blind: number
  big_blind: number
  ai_delay: number
  seats: TableSeat[]
  hand: HandView | null
  legal: Legal | null
  can_rebuy: boolean
}

export interface GameSummary {
  game_id: number
  created_at: string
  players: number
  small_blind: number
  big_blind: number
  hands: number
}

export interface HandSummary {
  hand_id: number
  hand_no: number
  board: string
  net: number | null
  pot: number
  showdown: boolean
  created_at: string
}

export interface HandDetail {
  hand_id: number
  game_id: number
  hand_no: number
  human_seat: number | null
  button: number
  small_blind: number
  big_blind: number
  players: { seat: number; name: string; start_stack: number }[]
  net: Record<string, number>
  board: string
  steps: { action: { seat: number; type: string; amount: number } | null; view: HandView }[]
}

export interface CreateGameRequest {
  name: string
  players: number
  stack: number
  small_blind: number
  big_blind: number
  opponents: 'random' | 'station' | 'mixed'
  seed?: number
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.text()
    throw new Error(`${res.status}: ${body}`)
  }
  return res.json() as Promise<T>
}

export const api = {
  createGame: (req: CreateGameRequest) =>
    fetch('/api/games', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    }).then((r) => json<{ game_id: number; seed: number }>(r)),
  listGames: () => fetch('/api/games').then((r) => json<GameSummary[]>(r)),
  listHands: (gameId: number) => fetch(`/api/games/${gameId}/hands`).then((r) => json<HandSummary[]>(r)),
  hand: (handId: number) => fetch(`/api/hands/${handId}`).then((r) => json<HandDetail>(r)),
}

export type Command =
  | { type: 'start_hand' }
  | { type: 'rebuy' }
  | { type: 'set_delay'; seconds: number }
  | { type: 'action'; action: string; amount?: number }

/** Live game state over a websocket, reconnecting with backoff. */
export function useGameSocket(gameId: number) {
  const [snap, setSnap] = useState<Snapshot | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [connected, setConnected] = useState(false)
  const [gone, setGone] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    let closed = false
    let retry = 0
    let timer: number | undefined

    const connect = () => {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws'
      const ws = new WebSocket(`${proto}://${location.host}/ws/games/${gameId}`)
      wsRef.current = ws
      ws.onopen = () => {
        retry = 0
        setConnected(true)
      }
      ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data)
        if (msg.type === 'state') {
          setSnap(msg as Snapshot)
          setError(null)
        } else if (msg.type === 'error') {
          setError(msg.message)
        }
      }
      ws.onclose = (ev) => {
        setConnected(false)
        if (closed) return
        if (ev.code === 4404) {
          setGone(true)
          return
        }
        timer = window.setTimeout(connect, Math.min(500 * 2 ** retry++, 8000))
      }
    }
    connect()
    return () => {
      closed = true
      window.clearTimeout(timer)
      wsRef.current?.close()
    }
  }, [gameId])

  const send = useCallback((cmd: Command) => {
    const ws = wsRef.current
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(cmd))
  }, [])

  return { snap, error, connected, gone, send, clearError: () => setError(null) }
}
