import { useEffect, useState } from 'react'
import { GamePage } from './pages/GamePage'
import { HistoryPage } from './pages/HistoryPage'
import { Lobby } from './pages/Lobby'
import { ReplayPage } from './pages/ReplayPage'

function useHashRoute(): string[] {
  const read = () => location.hash.replace(/^#\/?/, '').split('/').filter(Boolean)
  const [parts, setParts] = useState(read)
  useEffect(() => {
    const onChange = () => setParts(read())
    window.addEventListener('hashchange', onChange)
    return () => window.removeEventListener('hashchange', onChange)
  }, [])
  return parts
}

// Routes: #/  ·  #/game/:id  ·  #/game/:id/history  ·  #/hand/:id
export default function App() {
  const [page, id, sub] = useHashRoute()
  const num = Number(id)
  if (page === 'game' && num && sub === 'history') return <HistoryPage gameId={num} />
  if (page === 'game' && num) return <GamePage key={num} gameId={num} />
  if (page === 'hand' && num) return <ReplayPage key={num} handId={num} />
  return <Lobby />
}
