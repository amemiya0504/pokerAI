import { useEffect, useState } from 'react'
import type { HandView, Legal } from '../api'
import { chips } from '../format'

interface Props {
  legal: Legal
  view: HandView
  heroSeat: number
  bigBlind: number
  onAction: (action: string, amount?: number) => void
}

/** Fold / check / call / bet-or-raise with a slider and pot-based presets. */
export function ActionBar({ legal, view, heroSeat, bigBlind, onAction }: Props) {
  const min = legal.min_raise_to ?? 0
  const max = legal.max_raise_to ?? 0
  const [amount, setAmount] = useState(min)
  useEffect(() => {
    setAmount(min)
  }, [min])

  const myBet = view.seats[String(heroSeat)]?.street_bet ?? 0
  const clamp = (x: number) => Math.min(max, Math.max(min, Math.round(x)))
  // Raise to: current bet + fraction of the pot after calling.
  const potAfterCall = view.pot + legal.call_amount
  const preset = (f: number) => clamp(view.current_bet + f * potAfterCall)
  const presets: [string, number][] = [
    ['最小', min],
    ['½ 池', preset(0.5)],
    ['¾ 池', preset(0.75)],
    ['1 池', preset(1)],
    ['全下', max],
  ]
  const raiseLabel = legal.raise_type === 'bet' ? '下注' : '加注到'
  const btn = 'flex-1 rounded-xl px-3 py-3 text-base font-semibold shadow active:scale-[.98] transition'

  return (
    <div className="space-y-2" data-testid="action-bar">
      {legal.raise_type && (
        <div className="space-y-2 rounded-xl bg-slate-900/80 p-2">
          <div className="flex flex-wrap gap-1.5">
            {presets.map(([label, v]) => (
              <button
                key={label}
                className="flex-1 rounded-lg bg-slate-700 px-2 py-1.5 text-sm hover:bg-slate-600"
                onClick={() => setAmount(v)}
              >
                {label}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-2">
            <input
              type="range"
              className="flex-1"
              min={min}
              max={max}
              step={Math.max(1, Math.floor(bigBlind / 10))}
              value={amount}
              onChange={(e) => setAmount(clamp(Number(e.target.value)))}
              aria-label="下注金額"
            />
            <input
              type="number"
              inputMode="numeric"
              className="w-24 rounded-lg bg-slate-800 px-2 py-1 text-right tabular-nums"
              min={min}
              max={max}
              value={amount}
              onChange={(e) => setAmount(Number(e.target.value))}
              onBlur={() => setAmount(clamp(amount))}
              data-testid="raise-amount"
            />
          </div>
        </div>
      )}
      <div className="flex gap-2">
        {legal.can_fold && (
          <button className={`${btn} bg-slate-600 hover:bg-slate-500`} onClick={() => onAction('fold')} data-testid="action-fold">
            棄牌
          </button>
        )}
        {legal.can_check ? (
          <button className={`${btn} bg-sky-700 hover:bg-sky-600`} onClick={() => onAction('check')} data-testid="action-check">
            過牌
          </button>
        ) : (
          <button className={`${btn} bg-sky-700 hover:bg-sky-600`} onClick={() => onAction('call')} data-testid="action-call">
            跟注 {chips(legal.call_amount)}
            {legal.call_amount < view.current_bet - myBet && '（全下）'}
          </button>
        )}
        {legal.raise_type && (
          <button
            className={`${btn} bg-amber-500 text-slate-900 hover:bg-amber-400`}
            onClick={() => onAction(legal.raise_type!, clamp(amount))}
            data-testid="action-raise"
          >
            {clamp(amount) === max ? '全下' : raiseLabel} {chips(clamp(amount))}
          </button>
        )}
      </div>
    </div>
  )
}
