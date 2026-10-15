import { useEffect, useState } from 'react'
import type { HandDetail } from '../api'
import { api } from '../api'
import { EventLog } from '../components/EventLog'
import { TableFelt } from '../components/TableFelt'
import { actionText, chips } from '../format'

/** Step through a finished hand as the human saw it. */
export function ReplayPage({ handId }: { handId: number }) {
  const [hand, setHand] = useState<HandDetail | null>(null)
  const [step, setStep] = useState(0)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api
      .hand(handId)
      .then((h) => {
        setHand(h)
        setStep(h.steps.length - 1)
      })
      .catch((e) => setError(String(e)))
  }, [handId])

  if (error) return <main className="p-6 text-red-400">{error}</main>
  if (!hand) return <main className="p-6 opacity-70">載入中…</main>

  const names = new Map(hand.players.map((p) => [p.seat, p.name]))
  const name = (seat: number) => names.get(seat) ?? `座位 ${seat}`
  const cur = hand.steps[step]
  const last = hand.steps.length - 1
  const heroNet = hand.human_seat !== null ? hand.net[String(hand.human_seat)] : null

  return (
    <div className="mx-auto max-w-6xl space-y-3 px-4 py-4">
      <header className="flex items-center justify-between text-sm">
        <a href={`#/game/${hand.game_id}/history`} className="text-amber-300 hover:underline">
          ← 歷史牌局
        </a>
        <span className="tabular-nums opacity-80">
          第 {hand.hand_no} 手 · 盲注 {chips(hand.small_blind)}/{chips(hand.big_blind)}
          {heroNet !== null && ` · 你 ${heroNet >= 0 ? '+' : ''}${chips(heroNet)}`}
        </span>
      </header>
      <div className="grid gap-3 lg:grid-cols-[1fr_18rem]">
        <TableFelt
          seats={hand.players.map((p) => ({ seat: p.seat, name: p.name, stack: p.start_stack }))}
          view={cur.view}
          heroSeat={hand.human_seat}
        />
        <aside className="h-64 lg:h-auto lg:max-h-[70vh]">
          <EventLog events={cur.view.events} name={name} />
        </aside>
      </div>
      <div className="space-y-2 rounded-xl bg-slate-900/80 p-3">
        <div className="text-center text-sm" data-testid="replay-step">
          {cur.action ? `${name(cur.action.seat)} ${actionText(cur.action.type, cur.action.amount)}` : '發牌'} · 第 {step} / {last} 步
        </div>
        <input type="range" className="w-full" min={0} max={last} value={step} onChange={(e) => setStep(Number(e.target.value))} aria-label="步驟" />
        <div className="flex gap-2">
          <button className="flex-1 rounded-lg bg-slate-700 py-2" onClick={() => setStep(0)}>
            ⏮
          </button>
          <button className="flex-1 rounded-lg bg-slate-700 py-2" onClick={() => setStep((s) => Math.max(0, s - 1))} data-testid="replay-prev">
            上一步
          </button>
          <button className="flex-1 rounded-lg bg-slate-700 py-2" onClick={() => setStep((s) => Math.min(last, s + 1))} data-testid="replay-next">
            下一步
          </button>
          <button className="flex-1 rounded-lg bg-slate-700 py-2" onClick={() => setStep(last)}>
            ⏭
          </button>
        </div>
      </div>
    </div>
  )
}
