# Web AI 2–9 人過度棄牌修正規劃

狀態：2026-10-23；Phase 0–2 第一版與 A1–A6 CPU 診斷已完成，Phase 3 Arena 驗收中。A6 的 1,000-deal 配對差對照的是合成 25% floor control，並非歷史版本；6-max CI 跨 0。當前 Arena 使用 v7 frozen chart、3,000 deals，結果寫入獨立報告。M9 Arena 未通過 E1；本計畫不改 M9 的 rule benchmark。
適用範圍：網頁對戰（Human vs AI）的 `realtime` 座位，2–9 人。

## 1. 問題與資料

使用者回報即時求解 AI 過度棄牌。實際牌局：

- 單挑 game 10：62 手，AI 翻前棄牌 57 次，只有 5 手看到翻牌。
- 6-max game 4：5 個 AI 座位共 145 個 seat-hand，翻前棄牌 132 次（91%），VPIP 19/145（13%），只有 6/29 手看到翻牌。
- 6-max game 8：60 個 seat-hand，棄牌 56 次（93%），只有 4/12 手看到翻牌。

根因：

1. 深籌碼翻前用 `RuleBasedAgent` 的固定 Chen 門檻：無人加注 Chen ≥8 才開牌，面對加注 Chen ≥9 才跟。枚舉 1,326 種起手牌，Chen ≥8 只有 10.7%，≥9 只有 6.6%。
2. 同一個門檻套在所有位置，沒有位置、人數、open size、pot odds 的區分。
3. 3 人以上 `RealtimeAgent` 直接退回 `RuleBasedAgent`；100 BB 翻前也沒有 solved push/fold 資料，同樣退回 rule。6-max 沒有進 solver，也沒有使用對手模型。
4. 翻後 rule 的 weak 一律 fold；draw 只在 pot odds ≤25% 跟；medium ≤33%。沒有 bluff-catch、range advantage、多人底牌權重。

這不是 GPU contention，也不是 M9 的剝削模型；是 rule fallback 的品質缺口。M9 的 `rule` 對手就是同一支，不能直接改它，否則之後重跑的比較基準會變。

## 2. 目標與非目標

### 目標

- 支援 2–9 人；每個 `(人數, 位置)` 有獨立的 169 格翻前表。
- 3–9 人翻後用可玩的多人 heuristic，不再出現明顯過度棄牌。
- 單挑翻後仍使用即時求解；只換翻前表。
- UI 正確標示：2 人翻後 `SOLVER_APPROX`；3–9 人 `HEURISTIC`。
- 不改 M9 的 `rule` 對手、門檻、種子與報告。
- 先 CPU 驗證，M9 結束後才做 GPU/重啟。

### 非目標

- 不宣稱 3–9 人 GTO；多人牌局沒有唯一均衡。
- 不做 deep-stack preflop solver（這是 M10 級工作）。
- 不使用來源不明的商業 preflop 圖表。
- 不改 M9 的 E1/E2 門檻。

## 3. 設計原則

- **位置表是資料，不是 runtime 公式**：每個位置的表先產生、校對、凍結成 JSON；執行時只查表。
- **不同位置一定不同表**：不能只用一個「by 人數」或全桌共用 Chen 門檻。
- **HU 與多人分開**：HU 翻後 solver；3–9 人多人 heuristic。翻前則 2–9 都查位置表。
- **短籌碼沿用已求解資料**：2–9 人、5–20 BB 的 push/fold 已存在；深籌碼才用新表。
- **可追溯**：每份表有 version、source、生成腳本、hash；標明 heuristic。
- **先隔離再上線**：新 agent 只給 web sessions；M9/Arena 預設路徑不變；用 feature flag 切換。

## 4. 位置模型

使用引擎既有的 `preflop_order(view)`：

- `n = len(view.seats)`。
- `pos = preflop_order(view).index(view.seat)`。
- `pos = 0` 是第一個行動的位置；`pos = n-1` 是 BB。
- 2 人：`pos 0 = BTN/SB`、`pos 1 = BB`。
- 3 人以上：`pos 0 = UTG`、…、`pos n-2 = SB`、`pos n-1 = BB`。

2–9 人共 44 個位置。名稱只用在 UI；圖表 key 用 `(n, pos)`，避免 7/8/9-max 命名混亂。

每個位置至少區分這些 context：

| context | 意義 | 第一版 |
|---|---|---|
| `unopened` | 前面全部蓋牌，沒人 limping | 必要 |
| `vs_open` | 面對一個 open raise | 必要 |
| `vs_3bet` | 面對 3-bet | 必要 |
| `bb_option` | SB limp，BB 可 raise/check | 必要 |
| `vs_limp` | 面對一個或多個 limper | V1.1 |
| `vs_open_call` | open 後有一個以上 caller | V1.1 |
| `vs_4bet` | 面對 4-bet | V1.1 |

`vs_open` 再分 open size：

- small：≤2.5 BB
- medium：2.5–4 BB
- large：>4 BB 或 all-in

第一版可先做 small/medium/large 三格；如果時間不夠，先做 medium，但要標明近似。

## 5. 翻前表資料模型

位置表放在：

- `data/preflop/web_charts/2max.json`
- `data/preflop/web_charts/3max.json`
- …
- `data/preflop/web_charts/9max.json`

每個檔案的結構：

```json
{
  "version": "web-preflop-v1",
  "source": "heuristic",
  "table_size": 9,
  "stack_bb": 100,
  "positions": {
    "0": {
      "label": "UTG",
      "unopened": {
        "fold": [169 floats],
        "call": [169 floats],
        "raise": [169 floats]
      },
      "vs_open_medium": { "...": "..." },
      "vs_3bet": { "...": "..." },
      "bb_option": { "...": "..." }
    }
  }
}
```

規則：

- 169 格順序和 `backend/dataset/spots.py` 的 `CLASS_ORDER` 一致；另存 `class_order_version`。
- 每個 cell 的 `fold + call + raise = 1`；不合法 action（例如不能 raise）在 runtime 過濾後重新正規化。
- 短籌碼（≤20 BB）先走現有 `PushFoldAgent`；只有沒有 solved data 的深籌碼才查表。
- 第一版 deep chart 以 100 BB 為基準；20–100 BB 共用同一張表，manifest 標明近似。V1.1 再增加 25/50 BB bucket。
- 表格產生後 freeze；runtime 不執行 generator。

## 6. 表的來源與產生

目前沒有授權明確的 deep-stack GTO preflop 資料；本計畫的 2–9 人深籌碼表是 **heuristic**，不是 solver 產物。

產生方式：

- `scripts/build_web_preflop_charts.py`：
  - 輸入每個 `(n, pos, context)` 的 range spec（open/continue/3-bet 百分比、hand-class 限制、playability 調整）。
  - 用共用的 169 手牌排序（`CLASS_ORDER` 或等價 ranking）產生 fold/call/raise 頻率。
  - 每個位置輸出獨立表，不共用一個 runtime 公式。
- 產生後逐位置校對，尤其是：
  - UTG/UTG+1 是否比 CO/BTN 緊；
  - SB 是否過緊；
  - BB 對 small/medium/large open 的防守是否合理；
  - 9-max 的 UTG+2 與 MP 是否被正確區分。
- `scripts/validate_web_charts.py` 檢查 schema、機率總和、缺漏位置、極端值。
- 如果之後要做 solver 產生的 GTO 表，另開 M10；不在本計畫假造。169 格只是輸出格式，不是求解證據。M10 必須記錄 solver／演算法、狀態與動作抽象、迭代數、種子、EV 及 exploitability 或可驗證誤差界；多人局不能沿用二人零和 CFR 收斂聲明。

## 7. 多人翻後 heuristic

檔案：`backend/agents/web_multiway.py`。

輸入：`PlayerView`、`LegalActions`、active opponents、position、pot odds。

第一版策略：

1. 用現有 evaluator 判斷成手（strong/medium/weak）與聽牌（flush draw、straight draw、combo draw）。
2. 計算 pot odds；用「estimated equity + implied odds」決定 call/fold，不再 weak 一律 fold。
3. 多人調整：
   - 對手越多，對 weak/medium 的 continue 門檻越嚴；
   - 但 pot odds 很好且有強聽牌時仍要跟；
   - bluff 頻率隨人數下降；只在 late position、對手少時 bluff。
4. 擲骰使用固定 seed，允許混合頻率。
5. 不做 solver，不假裝 GTO；decision reason 寫 `HEURISTIC multiway`。

效能目標：CPU p95 <50 ms/decision；若之後加入 Monte Carlo equity，先限制 sample 數並量測。

## 8. HU 與 RangeTracker

`backend/agents/web_realtime.py`：

- `WebRealtimeAgent(RealtimeAgent)`。
- 翻前：2–9 人都查新位置表。
- 2 人翻後：呼叫 `super().act()`，保留原本即時求解。
- 3–9 人翻後：呼叫 `WebMultiwayAgent`。

重要一致性：

- `RealtimeAgent` 目前用 module-level `preflop_range()` 建初始範圍；HU 換翻前表後，solver 的範圍也必須用同一份表。
- 做法：`RealtimeAgent` 增加 optional `preflop_range_fn`（預設現有函式）；`WebRealtimeAgent` 傳入 chart-based range。
- 自己的範圍：用實際 chart 頻率更新；對手範圍：用 chart 的 action 機率。
- 回歸測試：同一手、同一 board，solver 看到的 hero/opponent range 與 chart 推導的一致。

M9 的 `RealtimeAgent` 不傳這個參數，行為不變。

## 9. 程式架構

新增：

- `backend/agents/web_preflop.py`：chart lookup、position/context 選擇、抽樣。
- `backend/agents/web_multiway.py`：3–9 人翻後 heuristic。
- `backend/agents/web_realtime.py`：wrapper。
- `data/preflop/web_charts/*.json`：2–9 位置表。
- `scripts/build_web_preflop_charts.py`、`scripts/validate_web_charts.py`。
- `tests/test_web_preflop.py`、`tests/test_web_multiway.py`、`tests/test_web_realtime.py`。
- `scripts/experiments/web_agent_bench.py`。

修改：

- `backend/api/sessions.py`：web 的 `realtime` 改成 `WebRealtimeAgent`；M9/Arena registry 不動。
- `backend/api/app.py`／frontend：大廳與牌局標示 `HEURISTIC`（3–9 人），2 人翻後才標 `SOLVER_APPROX`。
- `README.md`、`docs/realtime-agent.md`：補上 2–9 人 web heuristic 的範圍與限制。

不動：

- `backend/agents/rule_based.py`
- `backend/opponent/m9.py`
- M9 gates/hash

## 10. 測試與驗收

### 單元測試

- 2–9 每個位置、每個必要 context 都有表；沒有缺 key。
- 每個 169 格機率總和為 1；runtime 過濾後仍合法。
- 位置 mapping：2/3/6/9 人各抽固定座位，`(n,pos)` 與 UI label 正確。
- 短籌碼走 push/fold；深籌碼走表。
- 多人 heuristic 不會在 `can_check` 時 fold；不會產生非法 raise。
- HU chart-based range 與 `preflop_range_fn` 一致。

### CPU 模擬

`scripts/experiments/web_agent_bench.py`：

- 2–9 每個人數各 5,000–10,000 手；固定 seeds；workers=1。
- 對手：`rule`、`station`、`random`。
- 輸出：VPIP、PFR、3-bet、fold-to-open、BB defense、fold-to-cbet、WTSD、bb/100。
- **逐位置**輸出，不只看全桌平均。

### Arena

- 新增 Arena kind `web`（使用 `WebRealtimeAgent`），但 M9 pool 不變。
- 2/6/9 各 10,000 手；其他人數 5,000 手。
- 對 `rule`：不顯著為負；對 `station`：為正。
- 效能：多人決策 p95 <50 ms；無非法動作、無 timeout。

### 驗收門檻（草案，跑之前凍結）

| 人數 | 翻前 VPIP／防守目標 |
|---|---|
| 2 | SB VPIP 70–85%；BB 對 minraise 防守 50–65%；3-bet 10–18% |
| 3–5 | 全桌 VPIP 30–45%；BB 對 open 防守 35–50% |
| 6 | 全桌 VPIP 22–32%；UTG 14–20%；BTN 35–50%；BB 防守 30–42% |
| 7–9 | 全桌 VPIP 18–28%；UTG 12–18%；BTN 30–45%；BB 防守 25–38% |

共同門檻：

- 任何位置不得 0% 或 100% 棄牌。
- 面對下注且 pot odds ≥3:1 時，fold-to-bet ≤65%。
- 不得出現「weak 一律 fold」造成對手任何兩張都能自動獲利。
- M9 `rule` 輸出、門檻、種子不變。

## 11. 時程

| 階段 | 內容 | 估計 |
|---|---|---|
| Phase 0 | schema、位置 mapping、2/6/9 代表表、lookup 測試 | 0.5 天 |
| Phase 1 | 2–9 全部 44 位置 × 必要 context 的表、校對 | 1–1.5 天 |
| Phase 2 | 多人翻後 heuristic、HU range 整合、單元測試 | 1 天 |
| Phase 3 | 2–9 CPU 模擬、逐位置調參 | 1 天 |
| Phase 4 | Arena、驗收報告；M9 結束後 HU/GPU 回歸 | 0.5–1 天 |

第一版可玩：約 **1.5–2 天**。
完整驗收：約 **4–5 個工作天**。

M9 目前約還要 1.5–2.5 小時；不阻塞 Phase 0–3，只阻塞 GPU 回歸與重啟。

## 12. 風險與對策

| 風險 | 對策 |
|---|---|
| 沒有合法 deep GTO 表 | 明確標 heuristic；要 GTO 就另開 M10 solve |
| 44 位置 × context 表太多、容易錯 | generator + schema validation + 逐位置測試；先做 2/6/9 代表表 |
| 位置 mapping 錯 | 用 `preflop_order` 的 index，名稱只做 UI；2/3/6/9 固定 fixture |
| HU solver 範圍與新翻前表不一致 | `preflop_range_fn` 注入 + 回歸測試 |
| 多人 heuristic 還是過度棄牌 | CPU bench gate；先修 fold-to-bet，再調 bluff |
| 改動影響 M9 | 新 agent 只進 web sessions；不改 rule/m9；GPU 等 M9 結束 |
| 重啟中斷目前牌局 | 先寫碼與 CPU 測試；等使用者說停再重啟 |
| UI 標示誤導 | 3–9 人明確顯示 `HEURISTIC`，2 人翻後才顯示 solver |

## 13. 已採用的第一版設定

- 深籌碼翻前位置表標 `HEURISTIC`；不宣稱 GTO。
- `vs_open` 小尺寸 ≤2.5 BB、中尺寸 >2.5 BB 且 ≤4 BB、大尺寸 >4 BB，並分早／晚開池者與跟注價格。
- 以 100 BB 校準；20–100 BB 共用此表，5–20 BB 優先使用舊 push/fold 解。
- 多人翻後用 192 次 CPU Monte Carlo，依翻前 prior 與公開翻後行動 likelihood 抽 opponent ranges，含 blocker、runout 與 split pot；動作 likelihood 仍是簡化 heuristic，玩家 ranges 的相依性近似。
- 面對下注以 range equity 對 pot odds 計算近似 call EV；不設逐手牌繼續率下限。pot odds ≥3:1 的 fold-to-bet ≤65% 是逐位置診斷／驗收指標，不得改寫單一手牌動作來湊門檻。
- 多人弱牌的固定隨機 bluff 已移除；沒有 solver 或可驗證參考解前，不宣稱 GTO，也不公布 heuristic 混頻為均衡頻率。
- 驗收要逐位置統計，對 rule 不顯著為負、對 station 為正；CPU 種子與門檻需預先固定，不能依結果回頭修改。

## 14. Definition of Done

- 2–9 每個位置都有獨立、可追溯的翻前表。
- Web `realtime` 在 2 人翻後仍走 solver；3–9 人走多人和翻前表。
- 2–9 CPU bench 與 Arena 達到門檻；逐位置報告可讀。
- HU solver 回歸無顯著退步；M9 `rule` 完全不變。
- UI 不再把 3–9 人 rule fallback 標成「即時求解」。
- README／HANDOFF 更新；使用者可以在 2–9 人遊玩，不再看到明顯的 90%+ 翻前棄牌。
