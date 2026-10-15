# Poker AI 研究平台

只使用虛擬籌碼的德州撲克（No-Limit Hold'em）AI 研究與模擬平台。設計依據為〈Poker AI 平台技術設計文件〉。

## 進度

| Milestone | 狀態 |
|---|---|
| M1 · 2～9 人 NLHE 引擎 | ✅ 完成（見下方驗收結果） |
| M2 · 網頁牌桌 Human vs AI | ✅ 完成（見下方驗收結果） |
| M3 · AI vs AI 大量模擬 | ✅ 完成（見下方驗收結果） |
| M4～M10 | 未開始 |

## 快速開始

需要 Python 3.12 以上；網頁前端另需 Node.js 22。引擎本身零外部依賴。

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[web,sim,dev,oracle]"
```

### 網頁牌桌（M2）

先建置前端，再啟動伺服器，然後開啟 http://127.0.0.1:8000 ：

```bash
cd frontend && npm install && npm run build && cd ..
uvicorn --factory backend.api.app:create_app
```

環境變數：`POKER_DB_URL`（預設 `sqlite:///poker.db`，PostgreSQL 用 `postgresql+psycopg://…`）、`POKER_AI_DELAY`（AI 每個動作前的等待秒數，預設 0.6，牌桌上也能調整）。啟動時會自動執行 Alembic 遷移。

開發前端時可另開 `npm run dev`（Vite 會把 `/api` 與 `/ws` 轉給 :8000）。

### 測試

```bash
scripts/run_tests.sh               # Windows: scripts\run_tests.ps1
```

端對端測試需要建好的前端與 Playwright Chromium（`python -m playwright install chromium`）；缺少時會自動跳過。Linux 若缺系統函式庫，執行 `sudo .venv/bin/python -m playwright install-deps chromium`。

### AI vs AI 模擬（M3）

```bash
python -m backend.arena rule random station --deals 20000 --seed 1 --out out/arena/run1 --db sqlite:///poker.db
```

Agent 寫成 `kind` 或 `name=kind`（kind：`rule`、`random`、`station`），同一種可用不同名字重複出現。每個 deal 會依座位輪換打 N 次（N = Agent 數），所以總手數 = deals × N。`--workers` 預設使用全部 CPU，`--boundary process` 讓每個 Agent 在獨立行程執行。

### CLI 示範

隨機 Agent 對戰，輸出 JSON 手牌紀錄：

```bash
python -m backend.cli --players 6 --hands 5 --seed 42 --out out/hands.jsonl
```

## 專案結構

```
backend/
  engine/            # 純 Python、零外部依賴，唯一的規則來源
    cards.py         # 0～51 整數牌、Deck（固定種子洗牌）、derive_seed
    evaluator.py     # 5～7 張牌直接評估，回傳可比較的整數分數
    actions.py       # ActionType、Action、LegalActions、IllegalActionError
    hand.py          # HandState 單手牌狀態機、PlayerView、HandResult
    table.py         # Table：座位、入座/離桌/暫離、按鈕移動、開新局
    records.py       # JSON 手牌紀錄與重播
    runner.py        # Table + Agent 的最小迴圈（M3 擴充成 Arena）
  agents/
    base.py          # Agent 介面、Decision、StrategySource（五級）
    baseline.py      # RandomAgent、CallingStationAgent（僅供測試）
    rule_based.py    # RuleBasedAgent：Chen 公式 + 成手強度與底池賠率（HEURISTIC）
  arena/             # M3：鏡像牌局模擬、多進程、bb/100 統計、Parquet 與資料庫輸出
  db/                # SQLAlchemy 2 模型、Alembic 遷移、存取函式
  api/
    app.py           # FastAPI：REST（牌桌、歷史）+ 每張牌桌一條 WebSocket
    sessions.py      # GameSession：牌桌狀態、AI 行動迴圈、存檔、重啟後恢復
    views.py         # 送給瀏覽器的 JSON，只從 PlayerView 產生
  cli.py             # CLI 示範
frontend/            # React + TypeScript + Vite + Tailwind
  src/pages/         # 大廳、牌桌、歷史牌局、逐步重播
  src/components/    # 牌桌、撲克牌、動作列、牌局紀錄
tests/               # pytest（含 tests/e2e 的 Playwright 測試）
scripts/             # run_tests.sh / run_tests.ps1
```

## 引擎介面

```python
state.legal_actions() -> LegalActions   # can_fold, can_check, call_amount, min_raise_to, max_raise_to
state.apply(Action) -> list[Event]      # 非法動作丟出 IllegalActionError，不會靜默修正
state.observation(seat) -> PlayerView   # 只含該座位可見資訊
state.result() -> HandResult            # 底池分層、贏家、每人淨輸贏
```

BET / RAISE 的 `amount` 是「加注到」的本輪總額。全下不是獨立動作，`state.all_in_action()` 會轉成對應的 CALL / BET / RAISE。

## 規則決策與已知簡化

| 規則 | 實作 |
|---|---|
| 籌碼 | 全部為整數 |
| 大盲不足額 | 翻前仍需跟滿整個大盲（Robert's Rules）；最小加注增量為 1 BB |
| 不足額全下 | 不改變最小加注增量；已行動者只有在累計被加注量達到一次完整加注時才重新開放加注（TDA 累計規則） |
| 無人可對抗 | 其他仍在牌局的玩家都無法再投入超過目前注額（已全下或籌碼不足）時，只能跟注或棄牌 |
| 強制過牌 | 注額已平且沒人能回應加注時，跳過該玩家，不要求無意義的過牌 |
| Heads-Up | 按鈕位下小盲、翻前先行動、翻後最後行動 |
| 未被跟注的下注 | 每輪結束時退回 |
| 側池 | 依每位未棄牌玩家的總投入切分；棄牌者籌碼留在池中但無資格分配 |
| 零頭 | 每個底池分別平分，零頭從按鈕左側第一位贏家起依序每人 1 枚 |
| 攤牌 | 簡化：進入攤牌的玩家全部亮牌 |
| 按鈕 | 簡化：每手移到下一位在桌玩家；不實作死按鈕 / 補盲 |
| Ante | 尚不支援（參數保留） |
| 離桌 / 暫離 | 牌局中提出離桌，該手結束後生效；輪到時自動過牌或棄牌。籌碼歸零者自動暫離，補籌碼後可重新加入 |

## M1 驗收結果

- **評估器**：10 萬組隨機 7 張牌與「21 組 5 張牌暴力組合」參考實作比對一致；另與 PokerKit 評估器交叉比對 3,000 組。
- **情境測試**：每條規則決策至少一個手寫情境（`tests/test_hand_scenarios.py`）。
- **模糊測試**：1 萬手隨機 2～9 人、隨機籌碼與盲注；每個動作後檢查籌碼守恆、底池非負、行動者合法、觀測不洩漏他人手牌，並以 JSON 紀錄重播比對事件與結果完全相同。
- **PokerKit 差分測試**：1 萬手，9,975 手最終籌碼完全一致；其餘 25 手只差在零頭落點（每人差距不超過該手零頭數）。

### 與 PokerKit 的已知差異（書面解釋）

1. **大盲不足額**：PokerKit 只要求跟到大盲實際下的金額，我們要求跟滿整個大盲。這類手牌（約 8%）不納入比對，改由情境測試 `test_short_big_blind_must_be_called_in_full` 覆蓋。
2. **零頭**：PokerKit 在輸家蓋牌後會把贏家相同的邊池合併，且把整份零頭給同一位贏家；我們每個邊池分別計算，並逐枚分配。總額相同，只有零頭落點不同。
3. **強制過牌**：沒人能回應加注且注額已平時，我們跳過該玩家；PokerKit 仍要求一次過牌。差分測試會自動補上這一步。
4. **單挑且大小盲相同**：PokerKit 以盲注大小判斷單挑的按鈕位，無法對應，這類手牌不納入比對。

### 效能基準（M3 參考值）

純 Python 單核心、隨機 Agent、含 JSON 紀錄：6～9 人約每秒 2,600 手；單挑深籌碼約每秒 900 手（隨機 Agent 會反覆加注）。

## M2 驗收結果

- **資料庫**：`players`、`poker_tables`、`games`、`hands`、`actions` 五張表（設計文件第 6 節中 M2 用得到的部分），以 Alembic 遷移建立。每手只存「種子 + 起始籌碼 + 動作序列」與衍生結果；讀取時重播並核對結果，不一致會直接報錯。
- **資訊隔離**：瀏覽器收到的 JSON 只從 `PlayerView` 產生，AI 也只拿得到自己的觀測。API 測試與 Playwright 測試都會在每手結束後，用伺服器端的真實手牌比對**每一個** WebSocket 訊框與畫面上的每張牌：未攤牌的對手手牌從未出現，攤牌的牌只在本手結束後出現。刻意把視角換成別的座位時，兩組測試都會失敗（突變測試已驗證）。
- **桌面與手機**：Playwright 以 1280×800 與 390×844 兩種尺寸各打完兩手、補充籌碼、瀏覽歷史並逐步重播，且無水平捲動。
- **其他**：伺服器重啟後牌桌可從資料庫恢復（籌碼、手數、按鈕位置）；非法動作會回傳錯誤訊息，不會改變狀態；Windows 與 Linux 都通過後端測試。

M2 的 AI 只有基準 Agent（隨機、跟注站），策略型 AI 從 M4 開始。AI 籌碼歸零時會自動補回起始籌碼；你則需按「補充籌碼」。

## M3 驗收結果

**方法**：鏡像牌局（duplicate）。每個 deal 用同一副牌，讓 Agent 依座位循環輪換各打一次，每個 Agent 都會在每個位置拿到每個座位的牌；每手籌碼重設為 100 BB，按鈕固定在 0 號座位。所有隨機來源都由主種子依（deal、輪換、Agent）派生，與 worker 數量和分批方式無關。

**統計**：同一 deal 的各手彼此相關，所以以 deal 為獨立單位：取每個 Agent 在該 deal 的平均輸贏，再以這些區塊平均值的平均數與標準誤，搭配 Student-t 分位數計算 95% 信賴區間。

- **同種子結果完全相同**：1、3、4、16 個 worker 跑出的逐手資料與統計逐位元相同；換種子則不同。同一組設定在 Windows 與 Linux 上的結果也完全一致。
- **輸出 bb/100 與 95% 信賴區間**：CLI、`summary.json` 與資料庫 `benchmark_results` 都有 bb/100、標準差、信賴區間；以已知平均的合成資料重複 2,000 次，區間涵蓋率落在 93.5%～96.5%。
- **鏡像牌局正確**：相同且決定性的 Agent（2、3、6 人）每個 deal 的淨輸贏都**剛好為 0**；每手零和；每個 Agent 坐過每個座位。
- **可重播**：`hands.parquet` 每列保存牌堆種子與動作序列，測試逐手重建後與 `players.parquet` 的輸贏完全一致。
- **隔離**：`inline` 模式下 Agent 拿到的是不可變的 `PlayerView`，改不動牌局；`process` 模式下 Agent 在獨立行程，只透過 JSON 資料溝通。兩種模式的結果完全相同（同時證明 Agent 只依賴可序列化的資料）。
- **紀錄**：每次模擬記錄主種子、git commit（有未提交變更時加 `-dirty`）、Agent 設定與耗時（資料表 `simulation_runs`、`benchmark_results`，Alembic 遷移 0002）。

### 吞吐量（24 核心 WSL2，6 人桌、規則/隨機/跟注站 Agent）

| worker 數 | 每秒手數 |
|---|---|
| 1 | 約 1,400 |
| 8 | 約 8,100 |
| 16 | 約 12,500 |
| 24 | 約 14,100 |
| 1（`process` 隔離） | 約 130 |

### 基準結果（種子 1）

| 對戰 | 手數 | 結果（bb/100，95% 信賴區間） |
|---|---|---|
| 單挑 rule vs station | 100,000 | rule +153.7 [149.0, 158.3] |
| 單挑 rule vs random | 100,000 | rule +107.5 [98.8, 116.1] |
| 6 人 rule×2、random×2、station×2 | 120,000 | rule +516 / +507；station +66 / +45；random −535 / −599 |

6 人桌中兩個相同的 Agent 結果落在彼此的信賴區間內，符合預期。鏡像牌局在這些對戰中只讓變異數降低約 1.0～1.2 倍：Agent 風格差異很大時，同一副牌的結果相關性低；Agent 越接近，降幅越大（完全相同時變異為 0）。

