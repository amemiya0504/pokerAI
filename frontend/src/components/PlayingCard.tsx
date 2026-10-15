import { rankLabel, SUIT } from '../format'

type Size = 'sm' | 'md' | 'lg'

const SIZES: Record<Size, string> = {
  sm: 'w-7 h-10 text-[11px] sm:w-9 sm:h-12 sm:text-sm',
  md: 'w-9 h-13 text-sm sm:w-11 sm:h-15 sm:text-base',
  lg: 'w-12 h-17 text-lg sm:w-14 sm:h-20 sm:text-xl',
}

/** A face-up card, a face-down card (card = null) or an empty slot (card = undefined). */
export function PlayingCard({ card, size = 'md' }: { card: string | null | undefined; size?: Size }) {
  const box = `${SIZES[size]} rounded-md shrink-0 select-none`
  if (card === undefined) {
    return <div className={`${box} border border-dashed border-white/15`} aria-hidden />
  }
  if (card === null) {
    return (
      <div
        className={`${box} border border-white/30 bg-[repeating-linear-gradient(45deg,#7f1d1d_0_4px,#991b1b_4px_8px)] shadow`}
        aria-label="蓋著的牌"
        data-card="hidden"
      />
    )
  }
  const red = card[1] === 'h' || card[1] === 'd'
  return (
    <div
      className={`${box} flex flex-col items-center justify-center bg-white font-semibold leading-none shadow ${
        red ? 'text-red-600' : 'text-slate-900'
      }`}
      data-card={card}
      aria-label={card}
    >
      <span>{rankLabel(card)}</span>
      <span>{SUIT[card[1]]}</span>
    </div>
  )
}
