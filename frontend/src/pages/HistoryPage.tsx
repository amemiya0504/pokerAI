import { useEffect, useState } from 'react'
import type { HandSummary } from '../api'
import { api } from '../api'
import { cardLabel, chips } from '../format'

export function HistoryPage({ gameId }: { gameId: number }) {
  const [hands, setHands] = useState<HandSummary[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.listHands(gameId).then(setHands).catch((e) => setError(String(e)))
  }, [gameId])

  const total = hands?.reduce((s, h) => s + (h.net ?? 0), 0) ?? 0
  return (
    <main className="mx-auto max-w-2xl space-y-4 px-4 py-6">
      <header className="flex items-center justify-between text-sm">
        <a href={`#/game/${gameId}`} className="text-amber-300 hover:underline">
          ← 回牌桌
        </a>
        <a href="#/" className="text-sky-300 hover:underline">
          大廳
        </a>
      </header>
      <h1 className="text-xl font-bold">牌桌 #{gameId} 的歷史牌局</h1>
      {error && <p className="text-red-400">{error}</p>}
      {hands && (
        <p className="text-sm opacity-80">
          共 {hands.length} 手 · 你的累計輸贏{' '}
          <b className={total >= 0 ? 'text-emerald-300' : 'text-red-300'}>
            {total >= 0 ? '+' : ''}
            {chips(total)}
          </b>
        </p>
      )}
      <ul className="divide-y divide-white/10 rounded-2xl bg-slate-900/70" data-testid="hand-list">
        {hands?.length === 0 && <li className="px-4 py-3 opacity-70">還沒有打完的牌局</li>}
        {hands?.map((h) => {
          const board = h.board.match(/../g) ?? []
          return (
            <li key={h.hand_id}>
              <a href={`#/hand/${h.hand_id}`} className="flex items-center justify-between gap-3 px-4 py-3 hover:bg-white/5">
                <span className="tabular-nums">第 {h.hand_no} 手</span>
                <span className="flex-1 truncate text-sm opacity-80">{board.length ? board.map(cardLabel).join(' ') : '未見翻牌'}</span>
                <span className="text-sm opacity-70 tabular-nums">底池 {chips(h.pot)}</span>
                <span className={`w-20 text-right tabular-nums ${(h.net ?? 0) >= 0 ? 'text-emerald-300' : 'text-red-300'}`}>
                  {(h.net ?? 0) >= 0 ? '+' : ''}
                  {chips(h.net ?? 0)}
                </span>
              </a>
            </li>
          )
        })}
      </ul>
    </main>
  )
}
