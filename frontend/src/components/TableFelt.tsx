import type { HandView } from '../api'
import { chips, handName } from '../format'
import { PlayingCard } from './PlayingCard'

export interface FeltSeat {
  seat: number
  name: string
  stack: number // used when the seat is not in the hand
  away?: boolean
}

interface Props {
  seats: FeltSeat[]
  view: HandView | null
  heroSeat: number | null
  lastActions?: Record<number, string>
}

/** Oval table with seats around it; the hero sits at the bottom. */
export function TableFelt({ seats, view, heroSeat, lastActions = {} }: Props) {
  const n = seats.length
  const heroIndex = Math.max(0, seats.findIndex((s) => s.seat === heroSeat))
  const winners = view?.is_over ? view.awards : {}

  return (
    <div className="relative mx-auto w-full max-w-4xl aspect-[3/4] sm:aspect-[16/10]">
      <div className="absolute inset-[9%_6%] sm:inset-[12%_9%] rounded-[50%] border-[10px] border-felt-edge bg-[radial-gradient(ellipse_at_center,#167a4b_0%,#0f5132_70%)] shadow-[inset_0_0_40px_rgba(0,0,0,.45)]" />

      <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 px-[18%]">
        <div className="flex gap-1 sm:gap-2" data-testid="board">
          {[0, 1, 2, 3, 4].map((i) => (
            <PlayingCard key={i} card={view?.board[i]} size="md" />
          ))}
        </div>
        {view && (
          <div className="rounded-full bg-black/35 px-3 py-0.5 text-sm tabular-nums" data-testid="pot">
            {view.is_over ? '本手結束' : `底池 ${chips(view.pot)}`}
          </div>
        )}
      </div>

      {seats.map((s, i) => {
        const angle = Math.PI / 2 + ((i - heroIndex) * 2 * Math.PI) / n
        const left = 50 + 41 * Math.cos(angle)
        const top = 50 + 41 * Math.sin(angle)
        const st = view?.seats[String(s.seat)]
        const isHero = s.seat === heroSeat
        const toAct = view && !view.is_over && view.to_act === s.seat
        const won = winners[String(s.seat)]
        const cards = st
          ? st.shown_cards ?? (isHero ? view!.hole_cards : st.folded ? [] : [null, null])
          : []
        // Bet chips sit between the seat and the table centre.
        const betLeft = 50 + 24 * Math.cos(angle)
        const betTop = 50 + 22 * Math.sin(angle)
        return (
          <div key={s.seat}>
            <div
              className="absolute -translate-x-1/2 -translate-y-1/2 flex flex-col items-center"
              style={{ left: `${left}%`, top: `${top}%` }}
              data-testid={`seat-${s.seat}`}
            >
              <div className="relative z-10 mb-[-6px] flex -space-x-2" data-testid={isHero ? 'hole-cards' : undefined}>
                {cards.map((c, j) => (
                  <PlayingCard key={j} card={c} size={isHero ? 'lg' : 'sm'} />
                ))}
              </div>
              <div
                className={`relative min-w-[5.5rem] sm:min-w-28 rounded-lg px-2 pb-1 pt-2 text-center text-xs sm:text-sm shadow-lg transition ${
                  toAct ? 'bg-amber-400 text-slate-900 ring-2 ring-amber-200' : 'bg-slate-900/90'
                } ${st?.folded || s.away ? 'opacity-50' : ''} ${won ? 'ring-2 ring-emerald-300' : ''}`}
              >
                {view && view.button === s.seat && st && (
                  <span className="absolute -right-2 -top-2 z-20 grid h-5 w-5 place-items-center rounded-full bg-white text-[10px] font-bold text-slate-900">
                    D
                  </span>
                )}
                <div className="truncate font-medium">{s.name}</div>
                <div className="tabular-nums opacity-90">{chips(st ? st.stack : s.stack)}</div>
                {(st?.all_in || st?.folded || lastActions[s.seat] || won) && (
                  <div className="text-[10px] sm:text-xs font-semibold">
                    {won ? `+${chips(won)}` : st?.all_in ? '全下' : st?.folded ? '棄牌' : lastActions[s.seat]}
                  </div>
                )}
                {view?.hand_names[String(s.seat)] && (
                  <div className="text-[10px] sm:text-xs text-amber-300">{handName(view.hand_names[String(s.seat)])}</div>
                )}
              </div>
            </div>
            {st && st.street_bet > 0 && !view?.is_over && (
              <div
                className="absolute -translate-x-1/2 -translate-y-1/2 rounded-full bg-amber-300 px-2 text-[11px] sm:text-xs font-semibold text-slate-900 tabular-nums shadow"
                style={{ left: `${betLeft}%`, top: `${betTop}%` }}
              >
                {chips(st.street_bet)}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}
