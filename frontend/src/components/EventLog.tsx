import { useEffect, useRef } from 'react'
import type { HandEvent } from '../api'
import { eventText } from '../format'

export function EventLog({ events, name }: { events: HandEvent[]; name: (seat: number) => string }) {
  const end = useRef<HTMLDivElement>(null)
  useEffect(() => {
    end.current?.scrollIntoView({ block: 'nearest' })
  }, [events.length])
  const lines = events.map((e) => eventText(e, name)).filter((x): x is string => x !== null)
  return (
    <div className="h-full overflow-y-auto rounded-xl bg-slate-900/70 p-3 text-sm leading-6" data-testid="log">
      {lines.length === 0 && <div className="opacity-60">尚未開始</div>}
      {lines.map((line, i) => (
        <div key={i} className={line.startsWith('──') ? 'mt-1 font-semibold text-amber-200' : ''}>
          {line}
        </div>
      ))}
      <div ref={end} />
    </div>
  )
}
