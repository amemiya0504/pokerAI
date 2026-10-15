import type { HandEvent } from './api'

const RANK: Record<string, string> = { T: '10' }
export const SUIT: Record<string, string> = { c: '♣', d: '♦', h: '♥', s: '♠' }

export const rankLabel = (card: string) => RANK[card[0]] ?? card[0]

export const cardLabel = (card: string) => `${rankLabel(card)}${SUIT[card[1]]}`

export const chips = (n: number) => n.toLocaleString('en-US')

const STREET: Record<string, string> = {
  preflop: '翻牌前',
  flop: '翻牌',
  turn: '轉牌',
  river: '河牌',
}

export const streetLabel = (street: string) => STREET[street] ?? street

const HAND_NAME: Record<string, string> = {
  'High Card': '高牌',
  'One Pair': '一對',
  'Two Pair': '兩對',
  'Three of a Kind': '三條',
  Straight: '順子',
  Flush: '同花',
  'Full House': '葫蘆',
  'Four of a Kind': '四條',
  'Straight Flush': '同花順',
}

export const handName = (name: string) => HAND_NAME[name] ?? name

export function actionText(type: string, amount: number, called?: number): string {
  switch (type) {
    case 'fold':
      return '棄牌'
    case 'check':
      return '過牌'
    case 'call':
      return `跟注 ${chips(called ?? amount)}`
    case 'bet':
      return `下注 ${chips(amount)}`
    case 'raise':
      return `加注到 ${chips(amount)}`
    default:
      return type
  }
}

export function eventText(e: HandEvent, name: (seat: number) => string): string | null {
  const who = e.seat !== null ? name(e.seat) : ''
  switch (e.type) {
    case 'post_blind':
      return `${who} 下盲注 ${chips(e.amount)}`
    case 'action':
      return `${who} ${actionText(e.action ?? '', e.amount)}`
    case 'deal_board':
      return `── ${streetLabel(e.street)}：${e.cards.map(cardLabel).join(' ')}`
    case 'return_uncalled':
      return `退回 ${who} 未被跟注的 ${chips(e.amount)}`
    case 'showdown':
      return `${who} 亮牌 ${e.cards.map(cardLabel).join(' ')}`
    case 'award':
      return `${who} 贏得 ${chips(e.amount)}`
    default:
      return null
  }
}
