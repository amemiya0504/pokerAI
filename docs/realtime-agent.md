# 即時求解 Agent 設計（以即時求解為主、網路為輔）

## 網頁專用 2–9 人策略（2026-10-23，驗證中）

backend/api/sessions.py 的 realtime factory 使用 WebRealtimeAgent；Arena 與 M9 仍使用原本的 RealtimeAgent。目前深籌碼翻前 runtime 使用 backend/agents/data/web_preflop_v7.json：44 個位置、3,168 個位置／情境表，每格 169 手牌機率。表由離線 heuristic generator 產生，runtime 只 lookup；來源、版本與 100 BB 校準均寫在 JSON。169 格只是資料格式，不是求解證據；這不是 GTO 解。真正 solver-derived 頻率需由 CFR/MCCFR 求解或可驗證參考解產生，並記錄抽象、迭代及 exploitability／誤差。適用的 5–20 BB solved push/fold cells 優先。3–9 人翻後 UI 標 HEURISTIC。

A1–A4 對 rule 的配對 3,000 手診斷在 out/web-realtime/a*-paired-rule-3000.json。A4 舊的 opener 倍率在 6-max 退步 −2.00 bb/100 [−3.79,−0.21]；A4R 按當桌相對位置重算後仍退步 −0.68 [−1.30,−0.05]。因此 A4 候選保留作分析，runtime 使用 A3 context 的 v7 no-floor 表。詳細 bb/100 與 CI 見 README「網頁 2–9 人策略改版」。

HU 翻後用私有子類 _WebHeadsUp 接原本 solver。它覆寫初始範圍建構，使用相同 frozen lookup 的每次公開翻前行動機率。重播僅使用公開起始籌碼、button、盲注與行動；synthetic deck 的底牌不參與 lookup。M8 的翻後更新保留，M9 exploit 在網頁停用，因參考模型來自舊翻前策略。這個接法沒有改 M9 的預設範圍邏輯或 rule。

多人 equity 依公開翻前行動形成起始 range，再以翻後行動的簡化 strength likelihood 更新；MC 避免手牌與公共牌重疊並算 split pot。此為 heuristic posterior，不含聽牌／下注尺寸／完整 blocker likelihood，也未精確處理多人 range 相依性。面對下注時用估計 equity 對照 pot odds，不再保底繼續；call EV 假設 equity 完全實現。多人弱牌隨機 bluff 已移除，尚無 solver-backed bluff frequencies。

A7 bluff 候選目前為 opt-in 且預設關閉。它只在可 check 的多人翻後節點，以牌力百分位近似各對手 call range，結合 blocker、獨立棄牌假設與 Monte Carlo equity 估算 bluff mix；沒有校準 joint response range，也不是 solver-derived 頻率。20-deal 冒煙配對太小，v5 報告 A7−A6 為 −7.17 bb/100、95% CI [−21.21,+6.88]，只作管線檢查；其中僅抽到 1 次 bluff，該手整手淨結果不能解讀成 bluff action EV。mix 是依下注相對 check 的估計 EV 優勢做非均衡 proxy，候選仍預設關閉。真正 GTO 頻率仍需 CFR/MCCFR 或可驗證參考解，並報告抽象、迭代、EV 與 exploitability／誤差界。現有 tabular CFR+ 僅解 Kuhn／Leduc、MCCFR push/fold 僅解短籌碼抽象、GPU 子賽局僅解 HU 翻後；沒有深籌碼 2–9 人翻前 solver。三人以上屬多人一般和問題，需先定義均衡與可計算的 best-response 診斷，不能直接套用雙人零和 NashConv 宣稱已解。

A5 range-source paired A/B 已完成：對 rule 的 5–9-max 各 3,000 deals，public-action-conditioned range 相對 uniform random-hand equity 假設的配對差為 +65.39 至 +72.70 bb/100，95% CI 均高於 0；但這只表示優於隨機手牌假設，不是相對 GTO 的收益證明。6-max 的 station、nit、maniac 各 200 deals 篩查 CI 都跨 0，仍未驗證；該對照兩組都使用 A3 v3 frozen table，不能視為 v7 chart 的驗收。A6 以 v7/public ranges 對 rule 做每桌 1,000 deals 配對：A6 相對合成 25% floor control 在 5、7–9-max 的 95% CI 高於 0，6-max CI 跨 0；對 rule 絕對結果在 6、7、9-max CI 跨 0。控制組不是歷史版本，結果不代表 GTO。v7 的 3,000-deal 2–9-max GPU Arena 正在執行。Deep-stack 2–9 人翻前與多人翻後的 Solver 求解仍是後續工作。

狀態：**A、B、C 階段已實作**（`backend/agents/realtime.py`，結果見 README「即時求解 Agent」）（C 階段含接上 M8 的對手模型）。B 階段的結論：翻牌用深度 2 的深度限制求解（河牌直接發完比牌）；網路熱啟動實測沒有幫助，不採用。已決定：每個決策 10 秒、深籌碼翻前先用規則型範圍、第一版只做單挑。

## 為什麼要這樣設計

M7 的 Policy Network 在「模仿 solver」上達標（測試集 KL 0.093、EV regret 0.19% 底池），但把網路的策略放回賽局樹、讓對手用最佳回應來打時，可被剝削量的中位數在河牌測試題約 9.6% 底池、轉牌測試題約 47% 底池，solver 分別只有約 0.2% 與 0.3%（`python -m backend.policy exploit`，160 題）。均衡策略的關鍵在無差異點的混合比例，網路在這裡的小誤差會被最佳回應放大，而且誤差會沿著賽局樹累積。

所以「接近 GTO」的決策要靠**做決策當下對子賽局求解**；網路只在求解不可行或時間不夠時使用，並清楚標示。DeepStack、Libratus、ReBeL 這類強 AI 都是這個結構。

## 範圍

- **第一版只做單挑（heads-up）**。三人以上的翻牌後牌局沒有可計算的唯一均衡（設計文件第 9、10 節），維持規則型策略並標為 `HEURISTIC`。
- **街道**：河牌、轉牌即時求解；翻牌用深度限制的即時求解（翻牌、轉牌有下注，河牌直接發完比牌）。
- **翻牌前**：短籌碼用 M4 的 push/fold 解（`SOLVER_APPROX`）；深籌碼目前沒有已求解的資料，用規則型策略並標為 `HEURISTIC`。

## 架構

```
             PlayerView（只有自己看得到的資訊）
                          │
                ┌─────────▼─────────┐
                │  RangeTracker     │  雙方範圍（1,326 維）隨每個動作做貝氏更新
                └─────────┬─────────┘
                          │ 目前節點 + 雙方範圍
                ┌─────────▼─────────┐
                │  RealtimeSolver   │  以目前狀態為根建子賽局，GPU 批次 DCFR，時間預算內求解
                └───┬───────────┬───┘
         成功（附 exploitability）│   │ 超時／記憶體不足／不支援
                    │           ┌▼──────────────┐
                    │           │ Policy Network │  備援（NEURAL_APPROX）
                    │           └┬──────────────┘
                ┌───▼────────────▼───┐
                │  Decision          │  依自己這手牌的機率抽樣；附來源、機率、原因
                └────────────────────┘
```

### 1. RangeTracker（雙方範圍）

- **起點**：翻前範圍。短籌碼時由 push/fold 策略依實際動作做貝氏更新（精確）；深籌碼時用規則型策略的翻前門檻推出範圍（近似，標示清楚）。
- **自己的範圍**：每次自己行動後，乘上「自己在這個節點、每組底牌選這個動作的機率」。這些機率就是即時求解（或網路）實際用的策略，所以是精確的。
- **對手的範圍**：乘上「假設對手在這個節點的策略」選到觀察動作的機率。沒有對手資料時用我方求解得到的對手均衡策略（self-play 假設）。有 M8 的對手模型時（`backend/opponent/ranges.py`）：
  - **翻後**：把均衡策略混入 2% 均勻（容許失誤），再做最小 KL 的「傾斜」，讓範圍加權後各類動作（棄牌／過牌／跟注／下注／加注）的頻率等於目標頻率；同一類裡各尺寸的比例、以及「哪些牌做什麼」的順序都保留。目標頻率 = 對手在這個情境（例如轉牌被下注）的觀察次數，以均衡自己的頻率為先驗中心、20 個虛擬觀察收縮（「在 GTO 基準上依信心程度調整」）；沒有觀察時就是均衡。
  - **翻前**：範圍寬度由對手最後一個翻前動作所在情境的頻率決定（加注 = 前 P(加注)；跟注 = 前 P(跟注)+P(加注)，其中最前面 P(加注) 的牌權重降為 0.3），同樣以啟發式寬度為先驗中心收縮；沒有觀察時與啟發式完全相同。
  - Agent 的 `hand_over(view)` 在每手結束時用公開資訊更新對手模型；網頁對戰為每個即時求解 AI 座位保留模型。Arena 每手重建 Agent，所以 Arena 裡不會跨手學習。
- **牌張移除**：與公共牌、自己手牌衝突的組合一律歸零。

M9 開啟且有參考模型時，頻率造成的範圍調整也受剝削證據限制：只有 VPIP 的 95% 區間明確寬於參考模型時才調整翻前範圍；翻後則只調整寬範圍對手，或有足夠觀察支持的過度棄牌情境。過度棄牌情境還要通過實際下注尺寸的損益兩平棄牌率檢查，並保留原本跟注／加注的比例。這避免把依強牌決策的高跟注頻率套到所有手牌。`exploit=False` 時，M8 仍可單獨使用原本的範圍模型。完整啟用條件與限制見 README「M9 驗收結果」。

### 2. RealtimeSolver（即時求解）

- **以實際狀態為根**：子賽局從目前的底池、籌碼、這條街已下的注開始，所以對手用樹外的下注尺寸時，直接以實際金額重新求解，不需要動作轉換（action translation）。需要擴充 `BatchedBuilder`，讓根節點可以是「街道中途、面對某個下注」。
- **時間預算**：每條街設定上限（例如河牌 2 秒、轉牌 5 秒），或提早達到 exploitability 目標就停；回傳實際達到的 exploitability，寫進決策原因。
- **快取**：同一手牌中，雙方範圍與公共狀態相同的請求直接重用（request hash）。
- **安全性**（C 階段）：預設用 safe re-solving：re-solve gadget（Brown & Sandholm 2017，以對手先驗範圍加權），對手每組牌可選「離開並拿前一次求解給他的反事實值」或「進入子賽局」，所以重新求解不能讓他任何一組牌拿得比前一次求解多。目標值用前一次求解的期望值（不是最佳回應值：後者理論上安全，但實測太保守、幾乎沒有改善）。前一次求解的樹找不到目前節點時退回 unsafe 並標示。實測：範圍正確時 unsafe 略好，範圍估錯時 unsafe 常比不重新求解還差，safe 從未比藍圖差（README「結果（C 階段）」）。

### 3. 網路的角色

1. **備援**：超時、樹太大（例如翻牌多尺寸）、記憶體不足時，用網路的策略，標為 `NEURAL_APPROX`，原因寫明為什麼沒有求解。
2. **熱啟動**（B 階段實測，**不採用**）：用網路策略初始化 solver（Brown & Sandholm 式 regret 熱啟動，或只初始化平均策略），轉牌測試題 12 題上，相同迭代次數時都比冷啟動差，而且在每個節點跑網路要 3～21 秒（`scripts/experiments/warm_start_turn.py`）。
3. **翻牌**：網路沒有翻牌訓練資料，翻牌改用深度限制求解；求解失敗時退回規則型。

### 4. 決策與標示

| 情況 | strategy_source | 原因欄位 |
|---|---|---|
| 即時求解完成 | `SOLVER_APPROX` | 子賽局、迭代次數、達到的 exploitability（% 底池） |
| 網路備援 | `NEURAL_APPROX` | 為什麼沒有求解（超時／不支援／記憶體） |
| 翻前短籌碼 push/fold | `SOLVER_APPROX` | 資料集與深度 |
| 翻前深籌碼、多人牌局 | `HEURISTIC` | 沒有已求解資料 |

`VERIFIED_EQUILIBRIUM` 只用在 Kuhn/Leduc 這類可精確驗證的遊戲，不會出現在德州撲克的決策中。

## 評估方式

1. **每個決策的 exploitability**：即時求解在根節點達到的值，統計中位數與分布。
2. **Arena**：單挑 100 BB、鏡像牌局，比較「即時求解 Agent」vs「純網路 Agent」vs「規則型」，回報 bb/100 與 95% 信賴區間。預期即時求解 > 純網路。
3. **整體 exploitability 估計**：local best response（設計文件第 10 節）對 Agent 實際打法的估計，作為後續工作。
4. **延遲**：每條街的決策時間分布；網頁對戰的 AI 思考時間可接受 1～5 秒，Arena 大量模擬則需要較短預算或快取。

## 分階段

- **A**：中途根節點的建樹、RangeTracker、單挑河牌與轉牌的即時求解 Agent、網路備援、標示；測試包含貝氏更新正確性、即時求解與離線求解一致、決策合法性、延遲；Arena 實測。
- **B**（已完成）：翻牌深度限制求解（深度 1 偏差太大，改用深度 2；偏差評估 `python -m backend.policy depth-bias`）、勝負矩陣建置加速、網路熱啟動實驗（不採用）。
- **C**：safe re-solving（已完成，`resolving="safe"` 為預設）；接上 M8 的對手模型（已完成，見「對手的範圍」）。對手模型給出的範圍可能是錯的，safe re-solving 正好限制這種錯誤的代價。

## 已決定的設定

- 互動對戰每個決策的時間預算為 10 秒；M9 Arena 驗收使用每步 2 秒。
- 深籌碼翻前先用規則型範圍，標示為 `HEURISTIC`。
- 第一版只做單挑。
