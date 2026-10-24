import { useMemo, useState } from 'react'
import { useGameSocket } from '../api'
import { ActionBar } from '../components/ActionBar'
import { EventLog } from '../components/EventLog'
import { TableFelt } from '../components/TableFelt'
import { actionText, chips, handName } from '../format'

const SPEEDS: [string, number][] = [
  ['即時', 0],
  ['快', 0.3],
  ['正常', 0.6],
  ['慢', 1.2],
]

export function GamePage({ gameId }: { gameId: number }) {
  const { snap, error, connected, gone, send, clearError } = useGameSocket(gameId)
  const [showLog, setShowLog] = useState(false)

  const names = useMemo(() => {
    const m = new Map<number, string>()
    snap?.seats.forEach((s) => m.set(s.seat, s.name))
    return m
  }, [snap?.seats])
  const name = (seat: number) => names.get(seat) ?? `座位 ${seat}`

  // Last voluntary action per seat on the current street.
  const lastActions = useMemo(() => {
    const out: Record<number, string> = {}
    const hand = snap?.hand
    if (!hand || hand.is_over) return out
    for (const e of hand.events) {
      if (e.type === 'deal_board') for (const k of Object.keys(out)) delete out[Number(k)]
      if (e.type === 'action' && e.seat !== null) out[e.seat] = actionText(e.action ?? '', e.amount)
    }
    return out
  }, [snap?.hand])

  if (gone) {
    return (
      <main className="p-6">
        找不到這張牌桌。 <a className="text-amber-300" href="#/">回大廳</a>
      </main>
    )
  }
  if (!snap) return <main className="p-6 opacity-70">連線中…</main>

  const hand = snap.hand
  const hero = snap.seats.find((s) => s.seat === snap.human_seat)!
  const heroWon = hand?.is_over ? hand.awards[String(snap.human_seat)] : undefined
  const winners = hand?.is_over
    ? Object.entries(hand.awards).map(([seat, amt]) => {
        const hn = hand.hand_names[seat]
        return `${name(Number(seat))} 贏得 ${chips(amt)}${hn ? `（${handName(hn)}）` : ''}`
      })
    : []

  return (
    <div className="mx-auto flex min-h-dvh max-w-6xl flex-col gap-3 px-4 pb-4 pt-3">
      <header className="flex flex-wrap items-center justify-between gap-2 text-sm">
        <a href="#/" className="text-amber-300 hover:underline">
          ← 大廳
        </a>
        <span className="tabular-nums opacity-80" data-testid="status">
          #{snap.game_id} · 第 {snap.hand_no} 手 · 盲注 {chips(snap.small_blind)}/{chips(snap.big_blind)}
          {snap.is_replay && ' · 同牌試玩'}
          {!connected && ' · 重新連線中…'}
        </span>
        <span className="flex items-center gap-3">
          <label className="flex items-center gap-1">
            AI 速度
            <select
              className="rounded bg-slate-800 px-1 py-0.5"
              value={snap.ai_delay}
              onChange={(e) => send({ type: 'set_delay', seconds: Number(e.target.value) })}
            >
              {SPEEDS.map(([label, v]) => (
                <option key={v} value={v}>
                  {label}
                </option>
              ))}
            </select>
          </label>
          <a href={`#/game/${gameId}/history`} className="text-sky-300 hover:underline">
            歷史牌局
          </a>
        </span>
      </header>

      <div className="grid flex-1 gap-3 lg:grid-cols-[1fr_18rem]">
        <section className="min-w-0">
          <TableFelt
            seats={snap.seats.map((s) => ({ seat: s.seat, name: s.name, stack: s.stack, away: s.away,
              strategy: s.strategy && s.strategy.street === hand?.street ? s.strategy.label : undefined }))}
            view={hand}
            heroSeat={snap.human_seat}
            lastActions={lastActions}
          />
          <p className="mt-2 text-center text-xs text-slate-400" data-testid="strategy-legend">
            翻前：位置表 HEURISTIC；5–20 BB 適用時用 push/fold。2 人翻後：solver，失敗時顯示替代方法。
            3–9 人翻後：HEURISTIC。
          </p>
        </section>
        <aside className={`${showLog ? 'block' : 'hidden'} h-64 lg:block lg:h-auto lg:max-h-[70vh]`}>
          <EventLog events={hand?.events ?? []} name={name} />
        </aside>
      </div>

      <div className="sticky bottom-0 space-y-2 bg-gradient-to-t from-ink via-ink/95 to-transparent pb-[env(safe-area-inset-bottom)] pt-3">
        {error && (
          <div className="flex items-center justify-between rounded-lg bg-red-900/70 px-3 py-2 text-sm" role="alert">
            <span>{error}</span>
            <button onClick={clearError} className="opacity-70">
              ✕
            </button>
          </div>
        )}
        {winners.length > 0 && (
          <div className="rounded-xl bg-slate-900/80 px-3 py-2 text-sm" data-testid="result">
            {heroWon ? <b className="text-emerald-300">你贏了 {chips(heroWon)}！ </b> : null}
            {winners.join('；')}
          </div>
        )}

        {snap.is_replay && (
          <p className="text-center text-xs text-sky-200" data-testid="replay-notice">
            同一手牌情境重播，可改變你的行動觀察 AI 回應；本次結果不影響正式籌碼或歷史紀錄。
          </p>
        )}
        {snap.status === 'playing' && snap.legal && hand ? (
          <ActionBar
            legal={snap.legal}
            view={hand}
            heroSeat={snap.human_seat}
            bigBlind={snap.big_blind}
            onAction={(action, amount) => send({ type: 'action', action, amount })}
          />
        ) : (
          <div className="flex gap-2">
            {snap.status === 'playing' ? (
              <div className="flex-1 rounded-xl bg-slate-800/80 py-3 text-center opacity-80" data-testid="waiting">
                {hand?.to_act !== null && hand?.to_act !== undefined ? `等待 ${name(hand.to_act)}…` : '發牌中…'}
              </div>
            ) : snap.can_rebuy && hero.stack === 0 ? null : (
              <button
                className="flex-1 rounded-xl bg-amber-500 py-3 font-semibold text-slate-900"
                onClick={() => send({ type: 'start_hand' })}
                data-testid="next-hand"
              >
                {snap.status === 'idle' ? '開始第一手' : '下一手'}
              </button>
            )}
            {snap.status === 'hand_over' && snap.can_replay && (
              <button
                className="flex-1 rounded-xl bg-sky-700 py-3 font-semibold"
                onClick={() => send({ type: 'replay_hand' })}
                data-testid="replay-hand"
              >
                同牌重玩
              </button>
            )}
            {snap.can_rebuy && (
              <button
                className="flex-1 rounded-xl bg-sky-700 py-3 font-semibold"
                onClick={() => send({ type: 'rebuy' })}
                data-testid="rebuy"
              >
                補充籌碼
              </button>
            )}
          </div>
        )}
        <button className="w-full text-xs opacity-60 lg:hidden" onClick={() => setShowLog((v) => !v)}>
          {showLog ? '隱藏紀錄' : '顯示牌局紀錄'}
        </button>
      </div>
    </div>
  )
}
