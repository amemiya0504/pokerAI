import { useEffect, useState } from 'react'
import type { CreateGameRequest, GameSummary } from '../api'
import { api } from '../api'
import { chips } from '../format'

const OPPONENTS: { value: CreateGameRequest['opponents']; label: string }[] = [
  { value: 'mixed', label: '混合（隨機 + 跟注站）' },
  { value: 'rule', label: '規則型（啟發式）' },
  { value: 'realtime', label: '即時求解（單挑；轉牌與河牌每步最多思考 10 秒）' },
  { value: 'pushfold', label: '求解器 push/fold（短籌碼 5～20 BB；其他情況改用規則型）' },
  { value: 'random', label: '隨機 Agent' },
  { value: 'station', label: '跟注站' },
]

export function Lobby() {
  const [form, setForm] = useState<CreateGameRequest>({
    name: '你',
    players: 6,
    stack: 10000,
    small_blind: 50,
    big_blind: 100,
    opponents: 'mixed',
  })
  const [games, setGames] = useState<GameSummary[]>([])
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    api.listGames().then(setGames).catch((e) => setError(String(e)))
  }, [])

  const set = <K extends keyof CreateGameRequest>(k: K, v: CreateGameRequest[K]) => setForm((f) => ({ ...f, [k]: v }))

  const create = async (e: React.FormEvent) => {
    e.preventDefault()
    setBusy(true)
    setError(null)
    try {
      const { game_id } = await api.createGame(form)
      location.hash = `#/game/${game_id}`
    } catch (err) {
      setError(String(err))
      setBusy(false)
    }
  }

  const field = 'w-full rounded-lg bg-slate-800 px-3 py-2 tabular-nums'
  return (
    <main className="mx-auto max-w-2xl space-y-6 px-4 py-6">
      <header>
        <h1 className="text-2xl font-bold">Poker AI 牌桌</h1>
        <p className="text-sm opacity-70">No-Limit Hold'em · 虛擬籌碼 · 你對戰 AI</p>
      </header>

      <form onSubmit={create} className="space-y-4 rounded-2xl bg-slate-900/70 p-4" data-testid="create-game">
        <h2 className="text-lg font-semibold">開新牌桌</h2>
        <div className="grid grid-cols-2 gap-3">
          <label className="space-y-1 text-sm">
            <span>你的名字</span>
            <input className={field} value={form.name} maxLength={32} onChange={(e) => set('name', e.target.value)} />
          </label>
          <label className="space-y-1 text-sm">
            <span>人數（含你）</span>
            <select className={field} value={form.players} onChange={(e) => set('players', Number(e.target.value))} data-testid="players">
              {[2, 3, 4, 5, 6, 7, 8, 9].map((n) => (
                <option key={n} value={n}>
                  {n} 人
                </option>
              ))}
            </select>
          </label>
          <label className="space-y-1 text-sm">
            <span>起始籌碼</span>
            <input type="number" className={field} min={1} value={form.stack} onChange={(e) => set('stack', Number(e.target.value))} />
          </label>
          <label className="space-y-1 text-sm">
            <span>盲注（小 / 大）</span>
            <div className="flex gap-2">
              <input type="number" className={field} min={1} value={form.small_blind} onChange={(e) => set('small_blind', Number(e.target.value))} />
              <input type="number" className={field} min={1} value={form.big_blind} onChange={(e) => set('big_blind', Number(e.target.value))} />
            </div>
          </label>
          <label className="col-span-2 space-y-1 text-sm">
            <span>AI 對手</span>
            <select className={field} value={form.opponents} onChange={(e) => set('opponents', e.target.value as CreateGameRequest['opponents'])}>
              {OPPONENTS.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>
          </label>
        </div>
        <p className="text-xs opacity-60">求解器 AI 只在已求解的情況（短籌碼 push/fold）使用近似均衡策略，其餘情況改用規則型策略並標示為啟發式。</p>
        {error && <p className="text-sm text-red-400">{error}</p>}
        <button disabled={busy} className="w-full rounded-xl bg-amber-500 py-3 font-semibold text-slate-900 disabled:opacity-50" data-testid="create-submit">
          入座
        </button>
      </form>

      {games.length > 0 && (
        <section className="space-y-2">
          <h2 className="text-lg font-semibold">最近的牌桌</h2>
          <ul className="divide-y divide-white/10 rounded-2xl bg-slate-900/70">
            {games.map((g) => (
              <li key={g.game_id} className="flex items-center justify-between gap-2 px-4 py-3 text-sm">
                <span>
                  #{g.game_id} · {g.players} 人 · {chips(g.small_blind)}/{chips(g.big_blind)} · {g.hands} 手
                </span>
                <span className="flex gap-3">
                  <a className="text-amber-300 hover:underline" href={`#/game/${g.game_id}`}>
                    繼續
                  </a>
                  <a className="text-sky-300 hover:underline" href={`#/game/${g.game_id}/history`}>
                    歷史
                  </a>
                </span>
              </li>
            ))}
          </ul>
        </section>
      )}
    </main>
  )
}
