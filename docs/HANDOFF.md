# 交接：目前進度與下一步（2026-10-24，已移至新電腦續跑）

## 新電腦續跑（2026-10-24 21:50 CST；本節優先於下方暫停檢查點）

- 新電腦：WSL Ubuntu 26.04、32 核、RTX 4060 Ti 16 GB。`.venv` 以 uv 建立（Python 3.12，系統缺 python3-venv），含 `.[web,sim,solver,dev,oracle]` 與 torch 2.11.0+cu128。out 資料自 tar 還原；`backend/opponent/m9.py` 與 `out/m9/m9-gates.sha256` 相符。
- **A6 Arena v7 完成並通過固定 gate**（`acceptance.passed = true`，2–9-max 完整覆蓋）。7–9-max station 組依下方指令重跑；parallel-5…9 經 seed／table SHA／config 斷言後合併到 `out/web-realtime/a6-arena-v7-3000.json`，gate 未修改。合併前備份在 `out/web-realtime/backup-20261024/`。單位 bb/100、95% CI：

| 人數 | vs rule | vs station |
|---:|---:|---:|
| 2 | +41.65 [+28.35,+54.95] | +379.39 [+308.98,+449.80] |
| 3 | +19.06 [+7.69,+30.43] | +1085.04 [+1013.13,+1156.96] |
| 4 | +24.48 [+12.98,+35.98] | +1363.25 [+1274.86,+1451.64] |
| 5 | +19.06 [+5.89,+32.23] | +1588.74 [+1488.16,+1689.31] |
| 6 | +7.61 [−3.53,+18.74] | +1621.33 [+1516.85,+1725.81] |
| 7 | +8.07 [−2.57,+18.72] | +1605.52 [+1501.41,+1709.64] |
| 8 | +10.85 [+0.69,+21.01] | +1645.15 [+1540.29,+1750.02] |
| 9 | +5.87 [−3.60,+15.35] | +1650.22 [+1543.05,+1757.38] |

- 6、7、9-max 對 rule 的 CI 跨 0：gate 只要求「不顯著為負」，不代表對 rule 有確定優勢。逐位置非退化棄牌與 cheap gate 全部通過。
- 注意：manifest 記錄的 `backend/agents/web_realtime.py`、`scripts/experiments/web_realtime_accept.py` SHA 與續跑時不同（差異為之後加入、預設關閉的 A7 `multiway_bluff`，Arena 未開啟）；7–9-max station 組在較快 CPU 上以 2 秒預算重跑。
- 下一步：A7 與 A2–A5 的 2–4-max 配對以 CPU 在 `~/poker` 背景續跑；翻前 v8（2–9 人、深度分桶、vs 3bet／4bet／all-in）在獨立 worktree 開發，不影響這些實驗。network_opponents.py 與全專案 pytest 留到 v8 程式定案後再跑。

### 翻前 v8（worktree `~/poker-v8`，branch `preflop-v8`）

- 實作與測試完成，內容見 README「翻前 v8」。`~/poker-v8/out` 是指向 `~/poker/out` 的 symlink；`~/poker-v8/.venv` 獨立。
- **v8 驗收門檻（2026-10-24 22:15 CST，使用者確認、執行前訂定，不事後調整）**：
  1. **100 BB**：`web_realtime_accept.py --phase arena --deals 3000 --seconds 2`，seed 20261029，2–9-max × rule／station 共 16 組，沿用 A6 的固定 `gate(rows, "arena")` 並要求 2–9-max 完整覆蓋。2-max 用 GPU、3–9-max 隱藏 GPU，各桌型一個 worker，輸出 `out/web-realtime/v8-arena-3000-parallel-{n}.json`，合併到 `v8-arena-3000.json`。
  2. **40 BB 與 150 BB**：`web_policy_ab.py --baseline web_preflop_v7.json --candidate web_preflop_v8.json.gz --stack-bb {40,150} --deals 3000 --seed 20261029 --seconds 2`，2–9-max × rule／station；32 個配對 v8−v7 的 95% CI 上界都必須 ≥ 0（不顯著為負）。輸出 `out/web-realtime/v8-paired-{40,150}bb-{n}-3000.json`。註：32 個比較中即使 v8 與 v7 相同，也可能因多重比較有約 0.8 個偽失敗，結果照實記錄、不改門檻。
  3. 1 與 2 都通過才算 v8 通過；不通過就先分析，不調整門檻。
- 執行：等 `~/poker` 的 A2–A7 背景配對（web_policy_ab.py）全部結束後自動啟動，避免 CPU 爭用影響 2 秒預算；啟動時把相關檔案 SHA 寫入 `out/web-realtime/v8-eval.manifest.txt`。
- **v8 驗收結果（2026-10-25 03:59–13:34）：未通過。** 依門檻 3，runtime 不升級，維持 v7 判斷前先分析。
  1. 100 BB Arena：**通過**固定 gate（2–9-max 完整、逐位置全部通過）。vs rule 2–9-max：+41.34、+18.86、+27.86、+22.28、+6.31、+10.39、+14.68、+10.26 bb/100（6-max CI 跨 0）；vs station 全部顯著為正。報告 `out/web-realtime/v8-arena-3000.json`（由 parallel-2…9 合併，seed／table SHA／config 已斷言）。
  2. 150 BB 配對：16 組全部不顯著為負（8-max vs rule +7.26 [+0.53,+13.98]、9-max vs rule +8.64 [+0.99,+16.29] 顯著為正）。
  3. **40 BB 配對：未通過**。vs rule 8 組都不顯著為負；**vs station 3–9-max 全部顯著為負**：−45.41、−41.88、−40.21、−51.65、−46.78、−44.88、−52.50 bb/100（2-max −12.43 [−30.32,+5.47]）。報告 `out/web-realtime/v8-paired-{40,150}bb-{n}-3000.json`。
  - 初步診斷（6-max vs station 40 BB）：v8 的 VPIP／PFR 只比 v7 低約 0.01–0.04，但每個位置的配對 EV 都為負（CO −81.7、HJ −74.7）；150 BB 時兩者頻率相同、差距不顯著。主要嫌疑是 20–60 BB 的 open 尺寸上限（40 BB 為 2.2 BB，v7 為 2.5／3.0 BB）：對只會跟注的 station，較小的尺寸少拿價值。單一變因診斷見 `out/web-realtime/v8-diag/`（只作診斷，不是驗收）。
  - 診斷結果（6-max vs station、40 BB、1,000 deals、seed 20261101，相對 v8）：只把 20–60 BB open 尺寸改回 v7 → +17.23 [+7.86,+26.60] bb/100；open 尺寸加上 unopened／vs-open 的深度範圍係數都改回 v7 → +48.63 [+22.86,+74.41]。兩者合計約等於 v8 對 v7 的損失（−51.65），所以損失來自這兩項淺籌碼調整（範圍收窄約占 2/3、較小 open 約占 1/3），不是新增的 vs 3bet／4bet／all-in 節點。對只會跟注的 station，較寬範圍與較大尺寸本來就賺更多；這不代表對會反擊的對手也一樣（40 BB vs rule 這兩項沒有造成顯著差異）。
  - 若要再驗收，候選 v8.1 = v8 但 unopened／vs-open 的深度係數與 open 尺寸回到 v7（保留 vs 3bet／4bet／5bet／all-in 節點與其深度係數）；必須重新訂門檻後完整重跑，不能沿用本次診斷宣稱通過。

### 翻前 v8.1（使用者要求建立並完整重跑；runtime 預設已指向 v8.1，驗收前不得當作已通過）

- 表：`backend/agents/data/web_preflop_v8_1.json.gz`（version `web-v8.1-depth-allin`），由 `python -m scripts.experiments.web_preflop_v8_tables` 與 v8 一起產生；v8 表重新產生後位元相同。open／limp／option、vs open（含 SB／BB 分流）在每個深度都與 100 BB 表相同（= v7 的寬度與尺寸）；只有 vs 3bet／cold 3bet／vs 4bet／vs 5bet 與 all-in 節點依深度變化。測試 94 passed（v8.1／v8 29、v7 53、API 12）。
- **v8.1 驗收門檻（2026-10-25 16:30 CST，執行前訂定，不事後調整）**：
  1. **100 BB**：與 A6／v8 相同的固定 Arena gate（`web_realtime_accept.py --phase arena --deals 3000 --seconds 2`，固定 seed 20261029，2–9-max × rule／station 16 組，`gate(rows, "arena")`，完整覆蓋）。輸出 `out/web-realtime/v81-arena-3000-parallel-{n}.json` → `v81-arena-3000.json`。
  2. **40 BB 與 150 BB**：`web_policy_ab.py --baseline web_preflop_v7.json --candidate web_preflop_v8_1.json.gz --stack-bb {40,150} --deals 3000 --seconds 2`，2–9-max × rule／station；32 個配對 v8.1−v7 的 95% CI 上界都 ≥ 0。**種子改用新的 20261102**：v8.1 的設計參考了 v8 在 seed 20261029 的結果（診斷用 20261101），沿用同一種子會有選擇偏差。輸出 `out/web-realtime/v81-paired-{40,150}bb-{n}-3000.json`。多重比較下即使無差異也可能約有 0.8 個偽失敗，照實記錄、不改門檻。
  3. 1 與 2 都通過才算 v8.1 通過並保留為 runtime；不通過則 runtime 回到 v7 並先分析。
- 執行：以 Windows 隱藏 `wsl.exe` 程序獨立執行（不受 Claude 工作階段結束影響），2-max Arena 用 GPU，其餘隱藏 GPU；manifest `out/web-realtime/v81-eval.manifest.txt`，進度 `out/web-realtime/v81-eval.out`。
- **v8.1 驗收結果（2026-10-25 16:28 – 10-26 01:58）：通過**，runtime 保留 v8.1。
  1. 100 BB Arena：通過固定 gate（2–9-max 完整、逐位置全部通過）。vs rule 2–9-max：+40.57、+18.86、+27.86、+22.28、+6.31、+10.39、+14.68、+10.26 bb/100（6-max CI 跨 0）；vs station 全部顯著為正。3–9-max 與 v8 完全相同（100 BB 起手時兩者的表相同、CPU 路徑確定性）；2-max 為 GPU 求解，數字略有不同。報告 `out/web-realtime/v81-arena-3000.json`。
  2. 40 BB 與 150 BB 配對（seed 20261102）：32 組全部不顯著為負。顯著為正：40 BB 8-max vs rule +3.33 [+0.91,+5.74]、9-max vs rule +2.69 [+0.38,+5.01]、7-max vs station +7.65 [+0.76,+14.54]；150 BB 9-max vs rule +6.75 [+0.74,+12.76]。v8 失敗的 40 BB vs station 3–9-max 現為 +0.94、−3.74、+0.55、+3.69、+7.65、−3.61、−3.22，都不顯著為負。報告 `out/web-realtime/v81-paired-{40,150}bb-{n}-3000.json`。
  - 限制：仍是 HEURISTIC 表（非 GTO）；通過只代表對 rule／station 不比 v7 差；20–30 BB 與 50–80 BB 桶尚未做 EV 配對驗收；頻率稽核、pytest、network_opponents 與 A2–A7 分析見下方「v8.1 後續檢查」。
- 移到原本電腦開網頁：在本機 commit 到 `preflop-v8` 後由使用者 `git push -u origin preflop-v8`；原電腦 `git fetch && git switch preflop-v8 && pip install -e ".[web,sim,solver]"`，再照 README「網頁牌桌（M2）」建置前端並啟動 uvicorn。網頁的 `realtime` 座位會自動讀 v8 表；v8 表與 equity 矩陣都在 git 內，不需要 out 資料。

### v8.1 後續檢查（2026-10-26）

- **翻前頻率稽核**（只模擬翻前；seed 20261103；每組 3,000 手；2–9 人 × 25／40／60／100／150 BB；自我對打與對 rule；結果 `out/web-realtime/v81-preflop-audit-3000.json`，稽核腳本未入庫）：
  - 頻率隨深度的方向合理：vs 3bet 的 raise 從 20 BB 的 23% 降到 150 BB 的 10%；open／vs open 在每個深度都相同（= v7）。20／40 BB 的 4bet 會因 `jam_fraction` 0.40 變成全押，所以 vs 4bet 節點只在 ≥ 60 BB 出現。
  - 167 個「fold 0% 或 100%」旗標中，131 個是 BB 面對 limp（可以免費過牌，fold 0% 是正常的）。
  - **漏洞 1：limp 後被加注就全部棄牌**（從 v7 就有，v8.1 沿用）。2-max BTN 用 37 類最弱的牌 limp（約 12%）。被 BB 加注到 2.5–4 BB 時，vs open 表對這些牌全部棄牌（每個深度 63–85 次決策，全是 fold）。3–5-max 的 BTN／CO／HJ limp 後遇到 SB／BB 加注也一樣。對「每次 limp 都加注」的對手，這些牌比直接棄牌多輸 0.5 BB；2-max 的損失上限約 3 bb/100（以全部手數計）。
  - **漏洞 2：平跟後遇到 squeeze 就全部棄牌**。cold_vs_3bet 也用在已經平跟 open 的玩家（例如 4–6-max BTN 平跟後，SB／BB 3bet）。這張表 fold 96–97%，有數格在 30–102 次決策中全部棄牌，會被頻繁 squeeze 剝削。
  - 兩個漏洞都只在表的設計層面確認，尚未用剝削型對手量化。修正會改變表，等同 v8.2，必須先訂門檻再完整驗收。
- **對手直接 open 全押**（`vs_allin_*_unopened`，Arena 與自我對打都沒出現過）以探針確認：key 正確（2-max 為 late，6／9-max 第一位為 early）；跟注範圍隨深度收緊、越後位越寬。100 BB 只跟 AA–KK（BB 再加 AKo，0.5–2.6%），40 BB 為 1–5%，20 BB 為 1–17%（2-max BB 36%）。假設的全押範圍很緊，所以面對亂推的對手會少賺；但跟注率仍足以讓任意兩張全押不賺錢（9-max 100 BB 合計約 10% 被跟，被跟時平均輸約 60 BB），不構成可剝削的漏洞。
- **全專案 pytest**（新電腦首次）：280 passed、2 skipped（前端未 build）、**1 failed**：`tests/test_cfr_games.py::test_leduc_value_and_convergence`。300 次迭代的 NashConv 為 0.01251 > 0.01（期望值 −0.0855 正常）。
  - 這個結果在任何 PYTHONHASHSEED、numpy 2.1.3／2.2.6／2.3.4／2.5.4 下都相同；`handoff/transfer-20261024` 分支同樣失敗，CFR 程式自 M5 起沒改過，所以不是 v8 造成的。
  - 舊電腦上一次全套是 249 passed，差異原因未查明。測試與門檻都沒有修改。log 在 `out/full-tests-20261026.log`。
- **network_opponents.py**（`--deals 2000`，GPU；10-26 03:22–03:43，exit 0）：八個模型的 KL、EV regret、river／turn exploitability 與 vs rule／vs net-mlp 的 bb/100 和 CI，都與舊電腦的 `out/policy/network_opponents.json` 完全相同；只有 CPU p50 延遲較低（1–3 ms，舊電腦 4–10 ms）。結果 `out/policy/network_opponents-20261026.{json,log}`。
- **A2–A5 的 2–4-max 配對**（seed 20261029、3,000 deals、rule／station；bb/100、95% CI；2-max 各組都不顯著）：
  - A2−A1、A4−A3：都不顯著。
  - A3−A2（v3 的 SB／BB 分流）：vs rule 不顯著；**vs station 3-max −62.18 [−84.51,−39.85]、4-max −45.01 [−68.35,−21.68]**，主要來自 BB（3-max −115.7、4-max −87.0）與 4-max CO（−74.1）。v7／v8.1 用的就是 A3 這套分流。
  - A5（同一張 v3 表，比較 public 與 uniform 對手範圍）：vs rule 3-max +48.41 [+35.83,+61.00]、4-max +54.37 [+39.71,+69.03]；**vs station 3-max −111.21 [−148.67,−73.75]、4-max −120.92 [−166.46,−75.39]**（SB −222.4、−354.8）。runtime 用的是 public range。
  - 解讀：A3 分流與 public range 對 rule 有利、對 station 不利，表示它們只在「對手範圍會隨行動收窄」的前提下才賺錢；面對只會跟注的對手應改用寬範圍。v7 Arena 對 station 的絕對值仍大幅為正（+379 到 +1650），所以這是少賺，不是輸錢。依觀察到的風格切換 range source，可以交給 M8 對手模型處理，尚未實作。
- **A7 逐位置**：對 rule 的正值集中在後位（BTN +2.1 到 +8.0、CO +2.2 到 +3.4、HJ +2.4 到 +3.8）；對 nit 在 3–9-max 都為正（7-max 不顯著）。對 station 3-max −1.41 [−2.79,−0.03]、tag 5-max HJ −3.3，對 maniac 與 station 大多為 0。維持預設關閉。

### 翻前 v8.2（修 limp 後全棄與平跟後遇 squeeze 全棄；2026-10-26 驗收未通過，runtime 維持 v8.1）

- 表：`backend/agents/data/web_preflop_v8_2.json.gz`（version `web-v8.2-capped-defence`），由 `python -m scripts.experiments.web_preflop_v8_tables` 與 v8／v8.1 一起產生（v8、v8.1 重新產生後位元相同）。除了新增的兩種 `:mdf` 節點，其他 10,780 張表與 v8.1 完全相同。
  - `limp_vs_raise`：自己 limp 後面對一次加注（不含全押）。
  - `flat_vs_squeeze`：自己平跟 open 後面對 3bet。
  - 兩者都存每手牌在「到達這個節點的範圍」中的百分位（open-limp 範圍，或各 vs-open 表平跟範圍的平均；依對隨機手牌的 equity 排序）。runtime 以 `mdf_scale × MDF × 0.75^(其他仍在局的對手數)` 決定繼續的比例，MDF = 1 − 跟注額／底池；limp 節點 `mdf_scale` 0.75，squeeze 節點 0.60。最強約 2.5% 的牌 re-raise。
  - 例：HU limp 後被加注到 4 BB，繼續 limp 範圍中最好的 30%（v8.1 為 0%）；6-max BTN 平跟後被 squeeze 到 10 BB，繼續約 22% 的平跟範圍（77、88、ATo、AJo 等），AA 4bet。
  - 只有表含 `defend_groups` 時才用新節點，所以 v8.1 表在同一份程式下行為不變。
- 剝削探針 `scripts/experiments/preflop_exploiters.py`（已註冊到 `web_policy_ab.py --opponents`）：`iso` 對每個 limp 用任意兩張牌加注到 4 BB（每多一個 limper 加 1 BB），`squeeze` 對每個被平跟的 open 用任意兩張牌 3bet 到 3 倍 open（每個跟注者再加 1 倍）。其餘行動與翻後都是 RuleBasedAgent。Arena 的 `OPPONENTS` 與 gate 未修改。
- 測試：`tests/test_web_preflop_v8.py`、`test_web_realtime.py`、`test_api.py` 共 102 passed。
- **v8.2 驗收門檻（2026-10-26 03:56 CST，執行前訂定，不事後調整）**：
  1. **100 BB Arena**：與 A6／v8／v8.1 相同的固定 gate（`web_realtime_accept.py --phase arena --deals 3000 --seconds 2`，固定 seed 20261029，2–9-max × rule／station 16 組，`gate(rows, "arena")`，完整覆蓋）。輸出 `out/web-realtime/v82-arena-3000-parallel-{n}.json` → `v82-arena-3000.json`。2-max 用 GPU，等 network_opponents 結束後才開始（一次只跑一個 GPU 工作）。
  2. **對 rule 配對**：`web_policy_ab.py --baseline web_preflop_v8_1.json.gz --candidate web_preflop_v8_2.json.gz --opponents rule --stack-bb {40,100,150} --deals 3000 --seconds 2 --seed 20261104`，2–9-max；24 組 v8.2−v8.1 的 95% CI 上界都 ≥ 0。station 不納入配對：station 從不加注，新節點不會觸發，兩版行為相同；station 由第 1 項 Arena 檢查。
  3. **剝削探針配對**：同上但 `--opponents iso squeeze`、100 BB、seed 20261104，2–9-max 共 16 組：每組 v8.2−v8.1 的 95% CI 上界都 ≥ 0，**而且 6-max vs squeeze 的 95% CI 下界 > 0（必須顯著改善），2-max vs iso 的點估計 > 0（方向正確）**。
     - 修訂紀錄：03:56 初版要求 2-max vs iso 也顯著為正；03:58 在啟動任何 seed 20261104 的執行之前，依檢定力估算改成只要求方向。理由：HU 的 limp 範圍是最弱的牌（Q7o、K2o 一類），面對任意兩張 iso，跟注只比棄牌每次多約 0.6 BB，3,000 deals 預估約 +1 bb/100；而 v8.1 的 2-max 配對 CI 寬約 ±2.2 bb/100，要顯著需約 3 倍手數（2-max 約多 30 小時）。HU 漏洞較根本的修法是不要只用最弱的牌 limp（改 open 表），本版未做。squeeze 節點的平跟範圍是中等牌，每次跟注預估多 4 BB 以上，維持顯著要求。
  4. 1–3 都通過，v8.2 才保留為 runtime；任一項不通過，runtime 回到 v8.1 並先分析，不調整門檻。第 2、3 項共 40 組，即使兩版真的無差異，也可能有約 1 組偽失敗；照實記錄，不改門檻。
  - 種子 20261104 未用於設計：v8.2 的參數只看過靜態探針（固定牌局的 lookup），沒有看過任何模擬結果。冒煙測試（只確認流程）用另一個種子 20261105，結果不採計。
- 執行（10-26 03:59 起）：`run_v82_eval.sh` 以 Windows 隱藏 `wsl.exe` 獨立執行，40 個工作、同時最多 24 個，2-max 先跑；manifest `out/web-realtime/v82-eval.manifest.txt`，進度 `out/web-realtime/v82-eval.out`。輸出：`v82-arena-3000-parallel-{n}.json`、`v82-rule-{40,100,150}bb-{n}-3000.json`、`v82-probes-100bb-{n}-3000.json`。
- **更正（04:02，啟動後、看任何結果前）**：03:58 修訂的理由「2-max 配對手數加 3 倍要多約 30 小時」是錯的。`web_policy_ab.py` 用 network 模式，2-max 配對約 2 分鐘就跑完；慢的只有用 solver 的 2-max Arena。工作已啟動，所以門檻不再更動（仍以 03:58 版為準）。另跑一組**不採計**的補充檢定：2-max vs iso、12,000 deals、新種子 20261106，輸出 `out/web-realtime/v82-supp-iso-2-12000.json`，只作 HU limp 修正效果的參考。
- **v8.2 驗收結果（10-26 03:59–04:15）：未通過**。依門檻第 4 項，runtime 改回 v8.1（`web_preflop.TABLE_PATH`），v8.2 的表、程式與測試保留為未通過的候選（同 v8）。判定腳本輸出見下；第 2、3 項已確定不通過，第 1 項不可能改變結論，所以 04:15 停掉尚未跑完的 2-max、5–9-max Arena（3、4-max 已完成，報告留在 `v82-arena-3000-parallel-{3,4}.json`，另有中斷的 5–9 檔案，不採計）。
  1. 100 BB Arena：未完成（已停止）。
  2. **對 rule 配對：未通過**。7-max 40 BB −1.76 [−3.04,−0.48]、100 BB −2.36 [−4.25,−0.47] 顯著為負；其餘 22 組不顯著，但點估計幾乎全部為負（−0.2 到 −2.6）。rule 只用強牌加注，對它防守較寬會小輸。
  3. **剝削探針：未通過**。
     - 2-max vs iso −7.95 [−17.39,+1.48]（需點估計 > 0）。
     - 6-max vs squeeze +0.12 [−4.53,+4.78]（需顯著為正）。
     - 其他：iso 3–9-max 都為正（+1.3 到 +2.3），9-max +2.29 [+0.31,+4.27] 顯著；squeeze 8-max +4.61 [+0.06,+9.16] 顯著，3、4、7-max 為負但不顯著。
     - 補充檢定（不採計）：2-max vs iso 12,000 deals +1.36 [−1.16,+3.88]。
  - 分析：
    1. 多人桌的 limp 修正方向正確（對 iso 普遍為正）。
    2. HU 的 limp 範圍太弱，跟注與棄牌差不多，看不出改善。
    3. squeeze 節點沒有預期的大幅改善。可能原因：多人翻後是 heuristic，3bet 底池打得不好；或平跟範圍面對任意兩張 squeeze 的實際 equity 實現率比假設低。這點尚未驗證。
    4. 對 rule 的小幅損失，代表「固定 MDF 防守」對只用強牌加注的對手是錯的。
  - 可能的下一步（未做）：
    - 用 M8 對手模型決定是否啟用防守：只在觀察到對手 iso／squeeze 頻率偏高時才防守，其餘維持 v8.1。
    - HU 改 open 表，不要只用最弱的牌 limp。
    - 新一輪都要重新訂門檻與種子。

### A8 依對手風格切換（opt-in，`WebRealtimeAgent(adaptive=True)`；2026-10-26 驗收未通過，預設關閉，runtime 維持 v8.1）

- 目的：A5 顯示 public 對手範圍對 station 少賺很多，v8.2 顯示固定的 capped-range 防守對 rule 小輸。改成依觀察到的對手風格決定，只在對手像 station 或常 iso／squeeze 時才改變行為。
- 實作（`backend/agents/web_realtime.py`；M8／M9 程式未改）：
  - **統計**：`hand_over` 除了更新 M8 模型，也記錄每個座位的 iso 機會／次數（第一次行動時前面有 limper 且無人加注）和 squeeze 機會／次數（第一次行動時有一個 open 加上至少一個跟注者）。記在同一個 `opponent_models` dict 的 `"preflop_style"` 鍵，網頁 session 本來就會跨手保存並 deep copy 這個 dict。
  - **station-like**：M8 VPIP 觀察 ≥ 30 手、VPIP ≥ 0.40、PFR ≤ 0.08。多人翻後估 equity 時，這些座位用均勻範圍（A5 的 uniform），其餘座位照舊用 public range。
  - **常 iso／squeeze**：機會 ≥ 8 次，且 (次數 + 1)／(機會 + 2) ≥ 0.50。只有在加注者被判定為常 iso（limp 後被加注）或常 squeeze（平跟後被 3bet）時，才用 v8.2 的 capped-range 防守節點；否則與 v8.1 完全相同。
  - **沒有對手資料時與 v8.1 完全相同**（測試 `test_adaptive_without_history_plays_exactly_v81`）。
- 評估工具：`simulate(..., persistent_models=True)`、`web_policy_ab.py --persistent-models --candidate-adaptive`。英雄在整個 run 保留同一份對手模型（每個對手座位都是同一種對手），像網頁 session 一樣；兩邊都用這個 harness。原本的 Arena／配對每手都重建 agent，模型永遠是空的，A8 不會觸發。
- 門檻設計前的分類檢查（設計種子 20261108，只看對手行為統計、沒看輸贏；`classify_check.py`，150 deals）：
  - station 的座位都被判為 station-like，其他類型都沒有。
  - iso／squeeze 探針分別 100% iso／squeeze，會被判定。
  - rule、nit 不觸發；tag 只有 3-max 一個座位觸發 iso；lag、maniac 部分座位被判為常 iso／squeeze（它們確實常這樣打）。
- 測試：`tests/test_web_preflop_v8.py`、`test_web_realtime.py`、`test_api.py`、`test_opponent.py` 共 122 passed。
- **A8 驗收門檻（2026-10-26 17:22 CST，執行前訂定，不事後調整）**：
  1. `web_policy_ab.py --baseline web_preflop_v8_1.json.gz --candidate web_preflop_v8_2.json.gz --baseline-label v8.1 --candidate-label A8 --persistent-models --candidate-adaptive --deals 3000 --seconds 2 --seed 20261107`，100 BB，2–9-max × {rule, station, iso, squeeze, nit, tag, lag, maniac} 共 64 組：每組 A8−v8.1 的 95% CI 上界都 ≥ 0。
  2. station 3-max 與 4-max 的 95% CI 下界 > 0（A5 在這兩個桌型 public 比 uniform 少賺 111／121 bb/100，A8 必須顯著改善）。
  3. iso／squeeze 不要求顯著改善（v8.2 已顯示效果小），結果照實記錄。
  4. 1、2 都通過才算 A8 通過。通過後 runtime 改為 v8.2 表加 `adaptive=True`（網頁 session）；並改成只有明確要求時才用防守節點，避免沒有 adaptive 的呼叫端誤用 v8.2 的防守。不通過則維持 v8.1、預設不啟用，不調整門檻。
  5. 不重跑固定 Arena：Arena 每手重建 agent、沒有對手資料，A8 與 v8.1 決策完全相同（上述測試），A6／v8.1 的 Arena 結果直接適用。
  - 限制：只驗 100 BB；64 組中即使無差異也可能有約 1–2 組偽失敗（多數組 A8 不觸發、差異為 0，實際更少），照實記錄。
  - 輸出 `out/web-realtime/a8-{opponent}-{n}-3000.json`，manifest `a8-eval.manifest.txt`，進度 `a8-eval.out`。
- **A8 驗收結果（10-26 17:22–18:40，64 組全部完成）：未通過**。判定腳本 `a8_gate.py`（scratchpad）。A8 維持 opt-in、預設關閉，runtime 不變（v8.1，網頁未啟用 adaptive）。
  - **station：未通過**。6-max −78.76 [−130.75,−26.76] 顯著為負；3-max +24.80 [−6.71,+56.31]、4-max −21.71 [−59.11,+15.68] 都沒有達到「顯著為正」。其他桌型：2-max +7.57、5-max −10.53、7-max −29.89、8-max −19.67、9-max +79.45 [+26.33,+132.58]（顯著為正）。BTN／CO 幾乎每個桌型都為負（6-max BTN −327.9、8-max BTN −299.1）。所以多人翻後對 station 改用均勻範圍，並沒有重現 A5 的改善。A5 是對所有座位都改 uniform，而且用的是 v3 表；兩者不能直接比較，原因尚未查明。
  - rule：2–5-max 差異正好是 0（分類器沒有觸發），6–9-max 不顯著（−0.94 到 +0.32）。nit、tag 都不顯著。
  - 對積極的對手大多是正的：
    - iso：8 組點估計全部為正，8-max +2.55 [+0.57,+4.52] 顯著。
    - squeeze：不顯著，−1.18 到 +2.89。
    - lag：5-max +4.62 [+0.52,+8.72]、7-max +4.08 [+0.60,+7.55] 顯著。
    - maniac：2-max +15.34 [+3.42,+27.26] 顯著。
  - 解讀：只對常 iso／squeeze 的對手才啟用 v8.2 防守，方向看起來是對的，而且對 rule 不再造成損失（v8.2 的問題）。失敗來自 station 的範圍切換。
  - 可能的下一步（未做，需新門檻與新種子）：
    - 拿掉 station 範圍切換，只保留「依對手啟用防守」重新驗收。
    - 另外查 station 範圍切換為何讓 BTN／CO 大輸，可能是均勻範圍讓英雄 value bet 太寬或跟注太多。
  - **更正（10-26 21:25，A8.1 後發現）**：上面「失敗來自 station 範圍切換」的解讀不成立。A8 和 A8.1 都有下面 A8.1 段落記錄的範圍估計 bug，A8 對 station 的結果同時混了兩個因素，無法分開。

### A8.1 只依對手啟用防守（opt-in，`WebRealtimeAgent(adaptive=True)`；2026-10-26 驗收未通過，預設關閉，runtime 維持 v8.1）

- 改動：`adaptive=True` 不再做 station 範圍切換，只保留「加注者被判定為常 iso／squeeze 時才用 v8.2 防守節點」；分類門檻、統計方式都與 A8 相同，未調整。
  - A8 的 station 範圍切換移到另一個開關 `station_ranges=True`（預設關閉，只用來重現 A8；`web_policy_ab.py --candidate-station-ranges`）。
  - 測試 `test_station_ranges_only_when_explicitly_enabled`：只開 `adaptive` 時不會用均勻範圍。四個測試檔共 122 passed。
- 設計依據（注意：下面的門檻是看過 A8 種子 20261107 的結果後訂的，所以 A8.1 一律換新種子 20261109，A8 的數字不算進驗收）：
  - A8 對 iso／squeeze 16 組合併（各組 bb/100 平均、各組獨立的 SE）為 +1.13 [+0.30,+1.96]；iso 單獨 +1.52 [+0.28,+2.76]，squeeze 單獨 +0.74 [−0.36,+1.84]。
  - iso／squeeze 每組改用 6000 deals，SE 約縮小為 1/√2。若真實效果與 A8 相同，合併檢定的檢定力約 97%；A8 的估計可能偏高（挑選效應），若真實效果只有 +0.7，檢定力約 65%。
  - station 不再有特別要求：A8.1 只在加注者常 iso／squeeze 時才改變行為，分類檢查顯示 station 不會被判定，預期差異接近 0。
- **A8.1 驗收門檻（2026-10-26 20:02 CST，執行前訂定，不事後調整）**：
  1. `web_policy_ab.py --baseline web_preflop_v8_1.json.gz --candidate web_preflop_v8_2.json.gz --baseline-label v8.1 --candidate-label A8.1 --persistent-models --candidate-adaptive --seconds 2 --seed 20261109`，100 BB，2–9-max × {station, rule, iso, squeeze, nit, tag, lag, maniac} 共 64 組；iso、squeeze 用 `--deals 6000`，其他用 `--deals 3000`。每組 A8.1−v8.1 的 95% CI 上界都 ≥ 0。
  2. iso＋squeeze 16 組合併：各組 A8.1−v8.1 bb/100 的平均，SE = √(Σ 各組 SE²)/16（各組 SE = CI 寬度／(2×1.96)），95% CI 下界 > 0。這是要證明依對手啟用防守確實有用，而不只是沒有傷害。
  3. lag、maniac 等其他對手不要求顯著改善，結果照實記錄。
  4. 1、2 都通過才算 A8.1 通過。通過後 runtime 改為 v8.2 表加 `adaptive=True`（網頁 session），並把 `lookup` 改成只有 `defend=True` 時才用防守節點，避免沒有 adaptive 的呼叫端用到 v8.2 防守。不通過則維持 v8.1、預設不啟用，不調整門檻。
  5. 不重跑固定 Arena：理由同 A8（每手重建 agent、沒有對手資料，與 v8.1 決策完全相同）。
  - 限制：只驗 100 BB；只驗這些模擬對手，真人的 iso／squeeze 頻率可能不同；64 組中即使無差異也可能有 1–2 組偽失敗，照實記錄。
  - 輸出 `out/web-realtime/a81-{opponent}-{n}-{deals}.json`，manifest `a81-eval.manifest.txt`，進度 `a81-eval.out`，判定腳本 `a81_gate.py`（scratchpad）。
- **A8.1 驗收結果（10-26 20:02–21:17，64 組全部完成）：未通過**。A8.1 維持 opt-in、預設關閉，runtime 不變。
  - 條件 1 **未通過**：station 9-max −92.07 [−138.90,−45.24]（BTN −316.3 顯著）。其餘 63 組 CI 上界都 ≥ 0；station 8-max −42.87 [−87.54,+1.80]、2-max −30.02 [−61.98,+1.95] 也接近失敗。
  - 條件 2 通過：iso＋squeeze 合併 +1.35 [+0.73,+1.97]。iso 3／4／5／8／9-max 單組顯著為正（+1.94 到 +4.16），squeeze 4-max +3.14 [+0.67,+5.62]。
  - rule：2–5-max、8-max 正好是 0，6／7／9-max 不顯著。nit、tag 都不顯著。lag 3-max +14.08、9-max +3.11，maniac 3-max +13.43 顯著為正。
  - **原因（驗收後診斷，未改程式）**：A8.1 對 station 的翻前行動和 v8.1 一樣，但候選載入的是 v8.2 表。英雄估計對手範圍的 `web_preflop.probabilities()`（`ranges()` 和多人翻後的 `_public_ranges` 都用它）呼叫 `lookup(view, legal)` 沒有傳 `defend`，預設會用 v8.2 的防守節點。所以只要有人「limp 後跟加注」或「平跟後跟 squeeze」，英雄對那個人的範圍估計就換成 v8.2 的 capped 防守範圍，而不是 v8.1 的範圍。station 正好最常這樣打。
    - 檢查（`a81_range_check.py`，scratchpad）：limp 後跟 iso 的範圍，v8.1 與 v8.2 的 total variation 在 3／6／9-max 分別是 0.777／0.491／0.453。
    - 這是設計錯誤：A8 門檻第 4 點已經寫了「改成只有明確要求時才用防守節點」，但我排在通過之後才做，應該在驗收前就做。A8 也受同一個問題影響。
  - 可能的下一步（未做，需新門檻與新種子）：讓 `lookup` 只有 `defend=True` 時才用防守節點（範圍估計固定用 v8.1 節點），測試確認 adaptive 沒觸發時翻前與翻後都與 v8.1 完全相同（包括範圍估計），再重新驗收。

### A8.2 修正範圍估計後重新驗收（opt-in，`WebRealtimeAgent(adaptive=True)`；2026-10-27 驗收未通過，預設關閉，runtime 維持 v8.1）

- 改動（只改 A8.1 找到的 bug，分類門檻與防守節點都不變）：
  - `web_preflop.lookup` 與 `web_preflop_v8.lookup` 的 `defend` 預設改成 `False`，只有 `defend=True`（adaptive agent 自己判定要防守時）才用 v8.2 防守節點。範圍估計（`probabilities()`／`ranges()`／`_public_ranges`）和其他所有呼叫端都讀 v8.1 節點。
  - 因此沒有 adaptive 時，載入 v8.2 表也與 v8.1 完全相同；v8.2 單獨驗收時「一律防守」的行為要回到 commit 0ff1fee 才能重現。
  - 測試：新增 `test_v82_range_estimates_match_v81`（limp 後跟 iso、平跟後跟 squeeze 兩個情境，v8.1／v8.2 的 `ranges()` 逐一相等）；`test_v82_preflop_is_legal_against_probes` 改成強制每個防守節點都防守。四個測試檔共 123 passed。
  - A8.1 診斷腳本 `a81_range_check.py` 的 total variation 從 0.453–0.777 變成 0.000。
  - 實作冒煙（種子 20261105、120 deals，非驗收）：station 9-max、station 3-max、rule 6-max 的 A8.2−v8.1 都正好是 0（含翻後）。
- **A8.2 驗收門檻（2026-10-26 22:42 CST，執行前訂定，不事後調整）**：與 A8.1 相同，只換種子與標籤。
  1. `web_policy_ab.py --baseline web_preflop_v8_1.json.gz --candidate web_preflop_v8_2.json.gz --baseline-label v8.1 --candidate-label A8.2 --persistent-models --candidate-adaptive --seconds 2 --seed 20261110`，100 BB，2–9-max × {station, rule, iso, squeeze, nit, tag, lag, maniac} 共 64 組；iso、squeeze 用 `--deals 6000`，其他用 `--deals 3000`。每組 A8.2−v8.1 的 95% CI 上界都 ≥ 0。
  2. iso＋squeeze 16 組合併：各組 bb/100 平均，SE = √(Σ 各組 SE²)/16，95% CI 下界 > 0。
  3. 其他對手不要求顯著改善，結果照實記錄。
  4. 1、2 都通過才算 A8.2 通過。通過後 runtime 改為 v8.2 表（`web_preflop.TABLE_PATH`）加網頁 session 的 `adaptive=True`；沒有 adaptive 的呼叫端行為與 v8.1 相同（上述改動）。不通過則維持 v8.1、預設不啟用，不調整門檻。
  5. 不重跑固定 Arena：理由同 A8（每手重建 agent、沒有對手資料，與 v8.1 決策完全相同）。
  - 設計依據與限制：門檻沿用 A8.1，A8.1 的 iso＋squeeze 合併結果（+1.35 [+0.73,+1.97]）已看過，所以換新種子 20261110，A8.1 的數字不算進驗收。只驗 100 BB 與這些模擬對手；64 組中即使無差異也可能有 1–2 組偽失敗，照實記錄。
  - 輸出 `out/web-realtime/a82-{opponent}-{n}-{deals}.json`，manifest `a82-eval.manifest.txt`，進度 `a82-eval.out`，判定腳本 `a82_gate.py`（scratchpad）。
- **A8.2 驗收結果（10-26 22:42–23:57，64 組全部完成）：未通過**。A8.2 維持 opt-in、預設關閉，runtime 不變（v8.1）。
  - 條件 1 **未通過**：lag 6-max −0.15 [−0.25,−0.05]（HJ −0.2 顯著）。其餘 63 組 CI 上界都 ≥ 0。
  - 條件 2 通過：iso＋squeeze 合併 +1.67 [+1.06,+2.28]。iso 3／5／6／8-max、squeeze 9-max 單組顯著為正。
  - 範圍估計修正生效：station、nit 16 組全部正好是 0；rule 只有 6／9-max 有極小差異（+0.06、+0.03，不顯著）；tag 只有 7／9-max −0.01（不顯著）。
  - lag、maniac：除 lag 6-max 外都不顯著；lag 7-max +3.14、maniac 4-max +9.18 接近顯著為正。
  - **lag 6-max 診斷（用同種子重跑，結果完全相同；`a82_lag6_diag.py`，scratchpad；不改變判定）**：3000 deals 中只有 14 個 deal 有差異，13 個都是 −0.25 BB、1 個 −1.2 BB，合計 −4.45 BB。不是隨機雜訊，而是某個固定情境每次小輸；差異太少使配對 CI 很窄。
  - **情境（10-27 診斷，`a82_lag6_spots.py`，scratchpad，記錄兩邊英雄每個決策）**：14 個 deal 全部是同一種情境。
    - 英雄在 LJ／HJ／CO／BTN 用 v8.1 的範圍 open-limp 弱的 offsuit 牌（A3o–A9o、K8o–KTo、Q8o、Q8s），lag 在後面 iso（lag 已被判定為常 iso），通常還有 1–3 人跟注。
    - v8.1 在這裡全部棄牌；A8.2 進入 `limp_vs_raise:mdf` 跟注 150。
    - 13 手進入翻後（12 手 3–5 人、1 手 2 人）都沒中，多人翻後 heuristic 不詐唬、equity 不夠就棄牌，全部在翻牌或轉牌棄牌；另 1 手跟注後被別人再加注，翻前棄牌。14 手沒有一手到攤牌或贏。每手多輸 1.5 BB（一個 deal 6 個座位平均就是 −0.25 BB），一手跟了翻牌注再棄多輸 7.2 BB。
  - **原因**：防守節點用 MDF 的想法（加注者對 capped 範圍加注，要守住一定比例），假設跟注的牌在翻後能實現 equity。但面對 lag（翻後持續下注）加上多人底池，英雄翻後策略只在有足夠 equity 時繼續，弱 offsuit 牌幾乎實現不了 equity。`OTHER_DEFENDER_DECAY`（每多一個其他對手 ×0.75）不足以把這些牌排除。
  - 證據有限：只有 14 手，且 14 手全輸（若中牌率約 30%，全輸的機率約 0.7%，所以也有運氣成分）。對 iso 探針與其他 lag 桌型，防守點估計多為正。

### A8.3 只在沒人跟注加注時防守（2026-10-27 驗收通過；目前 runtime：v8.2 表＋網頁 session `adaptive=True`）

- 改動（只針對 A8.2 lag 6-max 的情境）：`WebRealtimeAgent._defend` 多一個條件，最後一次加注之後若已有人跟注（多人底池），就不用防守節點，與 v8.1 相同。其餘（分類門檻、防守節點、範圍估計用 v8.1 節點）都不變。
  - 測試 `test_adaptive_defends_only_while_nobody_called_the_raise`：6-max CO limp、BTN（常 iso）iso，SB 棄牌時防守，SB 跟注時不防守。四個測試檔共 124 passed。
  - A8.2 lag 6-max 的 14 手中有 13 手在英雄行動前已有人跟注加注，會被排除；只剩 deal 993（只有英雄與加注者）仍會防守。
- 設計依據與限制：這個條件是看過 A8.2（種子 20261110）結果才加的，所以換新種子 20261111，A8.2 的數字不算進驗收。拿掉多人防守後，對 iso／squeeze 的增益可能變小；A8.2 合併 +1.67、SE 約 0.31，若增益減半（約 +0.8），6000 deals 的檢定力約 74%，所以 iso、squeeze 改用 9000 deals（SE 約 0.25，檢定力約 89%）。
- **A8.3 驗收門檻（2026-10-27 00:42 CST，執行前訂定，不事後調整）**：
  1. `web_policy_ab.py --baseline web_preflop_v8_1.json.gz --candidate web_preflop_v8_2.json.gz --baseline-label v8.1 --candidate-label A8.3 --persistent-models --candidate-adaptive --seconds 2 --seed 20261111`，100 BB，2–9-max × {station, rule, iso, squeeze, nit, tag, lag, maniac} 共 64 組；iso、squeeze 用 `--deals 9000`，其他用 `--deals 3000`。每組 A8.3−v8.1 的 95% CI 上界都 ≥ 0。
  2. iso＋squeeze 16 組合併：各組 bb/100 平均，SE = √(Σ 各組 SE²)/16，95% CI 下界 > 0。
  3. 其他對手不要求顯著改善，結果照實記錄。
  4. 1、2 都通過才算 A8.3 通過。通過後 runtime 改為 v8.2 表（`web_preflop.TABLE_PATH`）加網頁 session 的 `adaptive=True`；沒有 adaptive 的呼叫端行為與 v8.1 相同。不通過則維持 v8.1、預設不啟用，不調整門檻。
  5. 不重跑固定 Arena：理由同 A8。
  - 限制：只驗 100 BB 與這些模擬對手；64 組中即使無差異也可能有 1–2 組偽失敗；差異很少的組（像 A8.2 lag 6-max 只有 14 個 deal 不同）CI 很窄，只要固定情境小輸就會失敗，照實記錄。
  - 輸出 `out/web-realtime/a83-{opponent}-{n}-{deals}.json`，manifest `a83-eval.manifest.txt`，進度 `a83-eval.out`，判定腳本 `a83_gate.py`（scratchpad）。
- **A8.3 驗收結果（10-27 00:42–01:58，64 組全部完成）：通過**。
  - 條件 1 通過：64 組的 CI 上界都 ≥ 0。
  - 條件 2 通過：iso＋squeeze 合併 +2.06 [+1.65,+2.48]。
    - iso（9000 deals）：3–9-max 全部顯著為正，+6.03、+5.45、+4.82、+3.19、+2.22、+2.43、+2.10；2-max −1.67 [−6.39,+3.06] 不顯著。
    - squeeze（9000 deals）：4–9-max 全部顯著為正，+0.75 到 +2.65；3-max +0.40 不顯著；2-max 是 0（HU 沒有 squeeze）。
  - station、rule、nit 24 組全部正好是 0；tag 只有 3／4-max −0.03／−0.05（不顯著），其他是 0。
  - lag 全部不顯著（−0.03 到 +2.87），A8.2 失敗的 6-max 現在 +0.05 [−0.18,+0.29]。maniac 2-max +19.99 [+6.62,+33.36] 顯著為正，其他不顯著。
  - 比 A8.2 好的原因（推測）：拿掉多人防守後，對 iso／squeeze 的合併增益沒有變小反而變大（+1.67 → +2.06），表示多人底池的防守本來就是虧的。
- **runtime 已切換（照門檻第 4 點）**：
  - `web_preflop.TABLE_PATH` 改為 `web_preflop_v8_2.json.gz`。
  - `backend/api/sessions.py` 的 `_realtime` 改為 `WebRealtimeAgent(..., adaptive=True)`（`station_ranges` 維持關閉）。
  - 沒有 adaptive 的呼叫端（Arena、舊實驗）因為 `lookup` 預設不用防守節點，行為與 v8.1 完全相同。
  - 測試 `test_v82_only_adds_defence_nodes_and_is_the_runtime` 檢查以上設定。四個測試檔 124 passed；全專案 pytest 294 passed、2 skipped、1 failed（舊的 Leduc 收斂測試 `test_leduc_value_and_convergence`，NashConv 0.0125 > 0.01，切換前就已失敗，與 v8／A8 無關）。
- 限制：只驗 100 BB 與這些模擬對手；防守只在同一個網頁 session 中、對手至少 8 次 iso／squeeze 機會且比例 ≥ 50% 後才啟用，對真人的效果未驗證。固定 Arena 沒有重跑（理由見門檻第 5 點）。
  - 可能的下一步（未做，需新門檻與新種子）：找出 lag 6-max 那 14 個 deal 的情境並修正後再驗收；或接受目前結果，維持 v8.1。

### 翻前 v8.3 面對加注節點看實際賠率、頂端牌型不棄牌（2026-10-27 驗收通過；目前 runtime：v8.3 表＋網頁 session `adaptive=True`）

- 起因（使用者回報，網頁第 29 手）：6-max HJ 用 QQ 開到 300，BTN 跟，BB 最小再加注到 500；QQ 只需補 200 進 1,350 的底池（price 0.148）卻棄牌。
  - 節點 `6/100/HJ/vs_3bet_late_small:cheap` 的 QQ 列是 fold 0.052／call 0.216／raise 0.732，這次抽到 fold。
  - 原因 1：產生器用 `smooth(rank, defend, .025)`，而 vs 3bet／cold 3bet／vs 4bet／vs 5bet 的繼續寬度只有 2–9%，與平滑寬度同量級，連 AA 都到不了 100% 繼續。v8.2 全表 AA 平均棄牌率：vs 3bet 4.6%、cold 3bet 24%、vs 4bet 15%、vs 5bet 21%（QQ 分別 6.4%、32%、20%、28%）。
  - 原因 2：v8 寬度不分價格（`cheap`／`normal` 同一張表），最小 3bet 和 3 倍 3bet 用同一列。
- 改動（只動加注節點；表 `backend/agents/data/web_preflop_v8_3.json.gz`，version `web-v8.3-priced-raises`，與 v8／v8.1／v8.2 一起產生，舊三張重新產生後位元相同）：
  1. 加注節點中 AA、KK、QQ，以及繼續範圍較強的一半（rank ≤ 繼續寬度／2）一律不棄牌（`PREMIUM_SHARE`）。
  2. 加注節點另存每手牌對「假設的加注者範圍」的 equity（`RAISER_100`：vs 3bet 8%、cold 3bet 6%、vs 4bet 3.5%、vs 5bet 2.5%，乘上加注者所在節點的深度加注係數，後位再 ×1.35）。runtime（`priced_probs`）在 `0.70 × equity ≥ price + 0.03 × 後面還沒行動的人數` 時也繼續（平滑同 all-in 節點），多出的繼續一律是跟注；只會放寬、不會比表更緊。
  - 其他 7,000 多張表與 v8.2 完全相同；沒有 `priced_raises` 的表（v8.2）行為不變。
  - 靜態結果：第 29 手 QQ 棄牌 0；面對最小 3bet，HJ 的 open 範圍全部繼續（v8.2 只繼續 33%）；3 倍 3bet 繼續 41%（v8.2 33%）；5 倍 3bet 28%（v8.2 26%）。
  - 剝削探針 `minraise`（`preflop_exploiters.py`）：每手第一次面對 open 或 3bet 時，用任意兩張牌最小再加注，其餘照 RuleBasedAgent。
  - `web_policy_ab.py --baseline-adaptive`：baseline 也用 adaptive，用來對照目前網頁 runtime（v8.2＋A8.3）。
  - 測試：四個測試檔（`test_web_preflop_v8.py`、`test_web_realtime.py`、`test_api.py`、`test_api_ai.py`）128 passed。
- **v8.3 驗收門檻（2026-10-27 20:40 CST，執行前訂定，不事後調整）**：
  1. **100 BB**：`web_policy_ab.py --baseline backend/agents/data/web_preflop_v8_2.json.gz --candidate backend/agents/data/web_preflop_v8_3.json.gz --baseline-label A8.3 --candidate-label v8.3 --persistent-models --baseline-adaptive --candidate-adaptive --deals 3000 --seconds 2 --seed 20261112`，2–9-max × {station, rule, iso, squeeze, nit, tag, lag, maniac, minraise} 共 72 組：每組 v8.3−A8.3 的 95% CI 上界都 ≥ 0。
  2. **40 BB 與 150 BB**：同上加 `--stack-bb {40,150}`，2–9-max × {rule, lag} 共 32 組：每組 95% CI 上界都 ≥ 0。
  3. **minraise 必須顯著改善**：100 BB minraise 8 組合併（各組 bb/100 平均，SE = √(Σ 各組 SE²)/8），95% CI 下界 > 0。
  4. 1–3 都通過，runtime 才改為 v8.3 表（`web_preflop.TABLE_PATH`，網頁 session 仍是 `adaptive=True`）；任一項不通過，runtime 維持 v8.2＋A8.3 並先分析，不調整門檻。
  5. 不重跑固定 Arena：station 從不加注，所以 v8.3 對 station 與 v8.2 完全相同；對 rule 的影響由第 1、2 項配對檢查。這是限制，照實記錄。
  - 種子 20261112 未用於設計：v8.3 的參數只看過靜態探針（第 29 手與固定牌局的 lookup），沒有看過任何模擬結果。冒煙測試（只確認流程）用種子 20261113，結果不採計。
  - 104 組中即使兩版無差異，也可能有 1–2 組偽失敗；station 等不觸發加注節點的組會正好是 0。照實記錄，不改門檻。
  - 輸出 `out/web-realtime/v83-{opponent}-{stack}bb-{n}-3000.json`，manifest `v83-eval.manifest.txt`，進度 `v83-eval.out`，判定腳本 `v83_gate.py`（scratchpad）。
- **v8.3 驗收結果（10-27 20:41–21:58，104 組全部完成、全部 exit 0）：通過**。單位 bb/100、95% CI。
  - 條件 1 通過：100 BB 72 組的 CI 上界都 ≥ 0，沒有任何一組顯著為負。
    - station 8 組正好是 0；rule、iso、squeeze 的 2-max 也是 0，3–9-max 全部不顯著（−1.22 到 +3.10）。
    - nit：8-max +11.81 [+4.96,+18.66]、9-max +8.43 [+2.10,+14.76] 顯著為正，其他不顯著。
    - tag：6–9-max 顯著為正（+12.81、+17.79、+27.94、+16.64），2–5-max 不顯著。
    - lag：5–9-max 顯著為正（+22.05 到 +48.13），2–4-max 不顯著。
    - maniac：3–9-max 顯著為正（+22.83 到 +63.57），2-max +1.03 不顯著。
  - 條件 2 通過：40／150 BB 32 組的 CI 上界都 ≥ 0。
    - rule 16 組全部不顯著（40 BB −0.97 到 +0.39；150 BB −2.05 到 +3.26）。逐位置只有 40 BB 9-max UTG+2 −4.8 顯著為負，整組 +0.08 [−1.11,+1.26]。
    - lag：40 BB 6–9-max 顯著為正（+10.62 到 +20.92）；150 BB 3、5–9-max 顯著為正（+23.29 到 +58.41）；其他不顯著。
  - 條件 3 通過：minraise 合併 +59.66 [+55.02,+64.29]；8 組各自都顯著為正（+52.63 到 +68.61）。
  - 解讀：增益集中在會 3bet／4bet 的對手（minraise、maniac、lag、tag），符合「v8.2 面對加注棄太多」的診斷。對不常再加注的 rule 沒有可量測的差異。
- **runtime 已切換（照門檻第 4 點）**：
  - `web_preflop.TABLE_PATH` 改為 `web_preflop_v8_3.json.gz`；網頁 session 仍是 `adaptive=True`（A8.3 防守照舊）。
  - 範圍估計（`ranges`）也走同一個 `lookup`，所以對手面對加注時的範圍估計同樣看價格；這已包含在驗收的配對中。
  - 測試 `test_v83_with_a83_is_the_runtime` 檢查以上設定（原 `test_v82_only_adds_defence_nodes_and_is_the_runtime` 改名為 `test_v82_only_adds_defence_nodes`）。四個測試檔 129 passed；全專案 pytest 307 passed、10 skipped、1 failed（同上，舊的 Leduc 收斂測試，與 v8.3 無關）。
- 限制：
  - 只驗模擬對手。`REALIZATION = 0.70` 與假設的加注者範圍是捷徑，不是求解結果。
  - 沒有 adaptive 的呼叫端（Arena、舊實驗）現在也讀 v8.3 的加注節點，所以不再與 v8.1 完全相同；對 station 仍相同（station 不加注）。
  - 固定 Arena 沒有重跑（理由見門檻第 5 點）。

### 翻前 v8.4 面對 3bet 放寬（2026-10-28 驗收未通過；runtime 維持 v8.3）

- 起因（使用者回報「面對 3bet 和 4bet 還是太緊」）。用 runtime 的 `lookup` 實際建牌局做靜態稽核（100 BB、其他人都棄牌，算繼續部分佔 open 範圍的比例）：
  - v8.3 面對 3bet 大多只繼續 22–45%。例如 6-max BTN 被 BB 3bet 到 4 倍時只繼續 27%，2-max BTN 是 29%。
  - BB 用任意兩張牌 3bet 到 10（4 倍）：多投 9 搶 4，棄牌率超過 69% 就賺錢，v8.3 棄 73%。
  - 原因：繼續寬度是 `max(.055, open 寬度 × .24)`；超過 3.5 倍就算 large，再乘 0.8，所以盲注的標準 4–4.5 倍 3bet 落在最緊的那格；價格放寬用 0.70 的實現率，加上很窄的假設 3bet 範圍，所以只救得了最小 3bet。
  - 面對 4bet（2.2–2.5 倍）會繼續自己 3bet 範圍的 54–90%，所以這次不改。
- 改動（表 `backend/agents/data/web_preflop_v8_4.json.gz`，version `web-v8.4-wider-vs-3bet`，與 v8–v8.3 一起產生；舊四張重新產生後內容相同）：
  1. vs 3bet 節點的繼續寬度 = open-raise 寬度 × `VS_3BET_SHARE`（3bettor 在後位 small .55／large .42，在前位 .50／.38），再乘原本的深度係數；取它與 v8.3 寬度的最大值（BB 的「open 寬度」不是 open 範圍，所以要保底）。4bet 寬度維持 v8.3，所以多出來的繼續全部是跟注。
  2. 3bettor 在 open 者之後才行動（out of position，通常是盲注）時，4.5 倍以內都算 small（表欄位 `small_3bet_oop`）；in position 的 3bettor 維持 3.5 倍。
  3. 價格放寬的實現率依位置設定：in position 0.85、out of position 0.70（表欄位 `realization`；判斷方式是翻後其他還沒棄牌的人是否都比自己先行動）。
  - 沒有這兩個欄位的表（v8.3 以前）行為不變。範圍估計同樣走 `lookup`，所以會一起變。
  - 靜態結果（100 BB，v8.3 → v8.4，繼續部分佔 open 範圍的比例）：6-max BTN 被 BB 3bet 到 4 倍 27% → 57%；CO 被 BTN 3bet 到 3 倍 40% → 58%；2-max BTN 被 BB 3bet 到 4 倍 29% → 55%；9-max UTG 被 CO 3bet 到 4 倍 43% → 48%。全部 480 個組合（3–5 倍）在 40／100／150 BB 介於 37–84%。
  - 新剝削探針 `threebet`（`preflop_exploiters.py`）：第一次面對 open 且自己還沒行動時，用任意兩張牌 3bet 到 open 的 4 倍，每多一個跟注者再加 1 倍；其餘照 RuleBasedAgent。
- **v8.4 驗收門檻（2026-10-28 01:40 CST，執行前訂定，不事後調整）**：
  1. **100 BB**：`web_policy_ab.py --baseline backend/agents/data/web_preflop_v8_3.json.gz --candidate backend/agents/data/web_preflop_v8_4.json.gz --baseline-label v8.3 --candidate-label v8.4 --persistent-models --baseline-adaptive --candidate-adaptive --deals 3000 --seconds 2 --seed 20261114`，2–9-max × {station, rule, iso, squeeze, nit, tag, lag, maniac, minraise, threebet} 共 80 組：每組 v8.4−v8.3 的 95% CI 上界都 ≥ 0。
  2. **40 BB 與 150 BB**：同上加 `--stack-bb {40,150}`，2–9-max × {rule, lag, threebet} 共 48 組：每組 95% CI 上界都 ≥ 0。
  3. **threebet 必須顯著改善**：100 BB threebet 8 組合併（各組 bb/100 平均，SE = √(Σ 各組 SE²)/8），95% CI 下界 > 0。
  4. 1–3 都通過，runtime 才改為 v8.4 表（`web_preflop.TABLE_PATH`，網頁 session 仍是 `adaptive=True`）；任一項不通過，runtime 維持 v8.3 並先分析，不調整門檻。
  5. 不重跑固定 Arena：station 從不加注，所以 v8.4 對 station 與 v8.3 完全相同；對 rule 的影響由第 1、2 項配對檢查。這是限制，照實記錄。
  - 種子 20261114 未用於設計：v8.4 的參數只看過靜態稽核，沒有看過任何模擬結果。冒煙測試（只確認流程）用種子 20261115、少量 deals，結果不採計。
  - 128 組中即使兩版無差異，也可能有 1–3 組偽失敗；station 等不觸發 vs 3bet 節點的組會正好是 0。照實記錄，不改門檻。
  - 執行：`out/web-realtime/v84_eval.py`（每組一個 process、隱藏 GPU、每個 process 1 個 BLAS thread、20 個 worker）。輸出 `out/web-realtime/v84-{opponent}-{stack}bb-{n}-3000.json`，manifest `v84-eval.manifest.json`（表與程式 SHA），進度 `v84-eval.out`，log `v84-logs/`，判定 `out/web-realtime/v84_gate.py`。
- 冒煙（種子 20261115、每組 20 deals、結果不採計）：128 組全部 exit 0，判定腳本可讀全部報告。
- **正式驗收已啟動（2026-10-28 01:55 CST）**：runner PID 259998（`setsid nohup`，不受 Claude 工作階段結束影響），20 個 worker。表 SHA-256 v8.4 `af07e262f685…`，全部見 manifest。
  - 時間估計：9-max 對 station 單獨就要約 5.7 秒／deal（兩邊合計；多人翻後的 heuristic `equity` 占大半），所以單組約 5 小時；全部約 200 CPU 小時，預計 10–12 小時完成。
  - 中斷後重跑同一行指令會跳過已完成的組（`web_policy_ab.py` 依 config 續跑），但同一組做到一半的進度會重來。
  - 完成後執行 `.venv/bin/python out/web-realtime/v84_gate.py`，照門檻第 4 點決定是否切換 runtime。
  - **2026-10-28 02:43 CST 依使用者要求暫停**：runner 259998 與 20 個 worker 都以 SIGSTOP 停住（狀態 T），記憶體中的進度保留。繼續：`kill -CONT 259998 $(pgrep -P 259998)`。若改為終止，已完成的組會保留，做到一半的組重跑時從頭開始。
  - 2026-10-28 03:29 CST 依使用者要求以 SIGCONT 繼續。
- **v8.4 驗收結果（2026-10-28 06:28 判定；123／128 組完成）：未通過**。單位 bb/100、95% CI；判定 `out/web-realtime/v84_gate.py`。
  - 條件 1、2 **不通過**：10 組顯著為負，剩下 5 組（station 5–9-max）不論結果都改變不了判定。
    - rule 100 BB：4-max −4.14 [−7.18,−1.11]、5-max −3.40；iso（rule＋iso 探針）4-max −4.14、5-max −3.41；squeeze 5-max −4.40 [−8.75,−0.05]。
    - rule 40 BB：3-max −1.98 [−3.93,−0.03]、4-max −2.89 [−4.63,−1.14]；rule 150 BB：3-max −4.97 [−9.37,−0.58]、4-max −3.51 [−6.69,−0.32]。
    - maniac 100 BB 8-max −0.12 [−0.23,−0.02]（只有 UTG −0.8 顯著）。
    - 逐位置：損失集中在 BTN／SB（例如 rule 100 BB 4-max BTN −3.9、SB −7.4；5-max BTN −8.9；squeeze 5-max BTN −10.8，都顯著）。rule 的盲注只用強牌 3bet，v8.4 放寬的跟注正好對上它的價值範圍。
    - station 2–4-max 正好是 0（station 不加注）。
  - 條件 3 通過：threebet 100 BB 合併 +22.30 [+18.57,+26.03]；threebet 24 組（三種深度）全部顯著為正（+13.61 到 +41.85）。
  - 其他：nit 5–8-max、tag 5–6-max 顯著為正；lag 40／150 BB 多組顯著為正，100 BB 全部不顯著；minraise 3–9-max 顯著為正。
  - 解讀：放寬確實修好了「任意兩張牌 3bet 有利可圖」的漏洞，也讓對 tag／nit 更好；但對只用價值牌 3bet 的對手變差。v8.4 這種固定寬度沒辦法同時兼顧兩者。
- **runtime 維持 v8.3**（照門檻第 4 點）；`web_preflop.TABLE_PATH`、`test_v83_with_a83_is_the_runtime` 都沒改。v8.4 表、探針與測試保留在程式中（沒有 `small_3bet_oop`／`realization` 欄位的表行為不變）。`tests/test_web_preflop_v8.py` 63 passed、`test_web_realtime.py`＋`test_api.py`＋`test_api_ai.py` 82 passed。
- 可能的下一步（未做，需新門檻與新種子）：比照 A8.3，只對被觀察到 3bet 頻率高的對手（同一 session、足夠樣本）才用 v8.4 的 vs 3bet 節點，其他情況維持 v8.3。

## 暫停檢查點（2026-10-24 19:43 CST；歷史紀錄）

- 使用者要把計算移到另一台電腦；目前評估程序已按 PID 停止。SIGINT 一度被背景程序忽略，之後對各工作 PID 發 SIGTERM。已確認沒有 web_policy_ab.py、web_realtime_accept.py、Arena runner、網路評估、網站 supervisor 或 vite/uvicorn 程序仍在跑。
- RTX 4060 為 0% 使用，約 639 MiB / 8188 MiB、43°C。網站 supervisor PID 123459 已停止，網站維持停止。
- 所有變更仍未 commit；不要 push。換機時需要帶未提交的工作樹，以及 git 忽略的 out 資料，至少包含 out/web-realtime、out/m9、out/policy、out/datasets 中本任務要用到的報告與模型。

### A6 Arena v7 暫停點

- 固定驗收設定：3,000 deals × 16 組、seed 20261029、table SHA-256 fc4df14c9a056b3edc0a0b323770d84f878f4ce8c3aecb8bf051bc7b1a259d53。M9 rule、門檻、種子均未修改。
- 主報告 out/web-realtime/a6-arena-v7-3000.json 只有 2–4-max 的 rule/station。平行輸出 5、6-max 已各完成 rule/station；7、8、9-max 只有 rule 組已落盤。station 組中斷時 log 分別到 7-max 2900/3000、8-max 2280/3000、9-max 1920/3000；這些未完成組沒有寫入 JSON，續跑時會從該 station 組的 0 開始。輸出檔保留的 rule 組會由 web_realtime_accept.py 跳過。
- 在新電腦執行下列 7–9-max worker；5、6-max 不必重跑。每個命令沿用同一個輸出檔，程式會依 config、seed、table hash 重用已落盤的對手組。

```bash
cd ~/poker
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u scripts/experiments/web_realtime_accept.py --phase arena --deals 3000 --seconds 2 --players 7 --out out/web-realtime/a6-arena-v7-3000-parallel-7.json
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u scripts/experiments/web_realtime_accept.py --phase arena --deals 3000 --seconds 2 --players 8 --out out/web-realtime/a6-arena-v7-3000-parallel-8.json
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u scripts/experiments/web_realtime_accept.py --phase arena --deals 3000 --seconds 2 --players 9 --out out/web-realtime/a6-arena-v7-3000-parallel-9.json
```

- 三個 worker 完成後，把 parallel-5 至 parallel-9 各自 rows[桌型] 合併到主報告 rows，先斷言每份 seed 與 table_sha256 相同，再用 scripts.experiments.web_realtime_accept.gate(rows, "arena") 重算 acceptance，並驗證 2–9-max 完整覆蓋。不要修改 gate 或 m9.py。

### A2–A5 配對評估暫停點

- 四個輸出 JSON 尚不存在；四個 log 顯示工作都在第一組 2-max vs rule，停止時進度為 A2 2640/3000、A3 2620/3000、A4 2640/3000、A5-range 2280/3000。這些進度只在 log，未達完整配對組，所以換機後四個 2-max vs rule 組都要從頭跑；完整組落盤後，web_policy_ab.py 會跳過已完成的 opponent/player pairs。

```bash
cd ~/poker
env CUDA_VISIBLE_DEVICES= .venv/bin/python -u scripts/experiments/web_policy_ab.py --baseline backend/agents/data/web_preflop_v1.json --candidate backend/agents/data/web_preflop_v2.json --baseline-label A1 --candidate-label A2 --players 2 3 4 --opponents rule station --deals 3000 --seed 20261029 --seconds 2 --out out/web-realtime/a2-paired-rule-station-2-4-3000.json > out/web-realtime/a2-paired-rule-station-2-4-3000.log 2>&1 &
env CUDA_VISIBLE_DEVICES= .venv/bin/python -u scripts/experiments/web_policy_ab.py --baseline backend/agents/data/web_preflop_v2.json --candidate backend/agents/data/web_preflop_v3.json --baseline-label A2 --candidate-label A3 --players 2 3 4 --opponents rule station --deals 3000 --seed 20261029 --seconds 2 --out out/web-realtime/a3-paired-rule-station-2-4-3000.json > out/web-realtime/a3-paired-rule-station-2-4-3000.log 2>&1 &
env CUDA_VISIBLE_DEVICES= .venv/bin/python -u scripts/experiments/web_policy_ab.py --baseline backend/agents/data/web_preflop_v3.json --candidate backend/agents/data/web_preflop_v4.json --baseline-label A3 --candidate-label A4 --players 2 3 4 --opponents rule station --deals 3000 --seed 20261029 --seconds 2 --out out/web-realtime/a4-paired-rule-station-2-4-3000.json > out/web-realtime/a4-paired-rule-station-2-4-3000.log 2>&1 &
env CUDA_VISIBLE_DEVICES= .venv/bin/python -u scripts/experiments/web_policy_ab.py --baseline backend/agents/data/web_preflop_v3.json --candidate backend/agents/data/web_preflop_v3.json --baseline-label A4-uniform --candidate-label A5-range --baseline-range-source uniform --candidate-range-source public --players 2 3 4 --opponents rule station --deals 3000 --seed 20261029 --seconds 2 --out out/web-realtime/a5-range-paired-rule-station-2-4-3000.json > out/web-realtime/a5-range-paired-rule-station-2-4-3000.log 2>&1 &
wait
```

### A7 配對評估暫停點

- 既有 JSON 可續跑。a7-bluff-paired-3000-14groups.json 已完成 rule 對手 3–8-max；a7-bluff-paired-3000-style-28groups.json 已完成 nit 對手 3–6-max。中斷中的組未寫入 JSON，重跑同一命令會跳過已完成組並重做未完成組。

```bash
cd ~/poker
CUDA_VISIBLE_DEVICES= .venv/bin/python -u scripts/experiments/web_policy_ab.py --baseline backend/agents/data/web_preflop_v7.json --candidate backend/agents/data/web_preflop_v7.json --baseline-label A6 --candidate-label A7 --players 3 4 5 6 7 8 9 --opponents rule station --deals 3000 --seed 20261029 --candidate-multiway-bluff --out out/web-realtime/a7-bluff-paired-3000-14groups.json
CUDA_VISIBLE_DEVICES= .venv/bin/python -u scripts/experiments/web_policy_ab.py --baseline backend/agents/data/web_preflop_v7.json --candidate backend/agents/data/web_preflop_v7.json --baseline-label A6 --candidate-label A7 --players 3 4 5 6 7 8 9 --opponents nit tag lag maniac --deals 3000 --seed 20261029 --candidate-multiway-bluff --out out/web-realtime/a7-bluff-paired-3000-style-28groups.json
```

### 續跑順序

1. 完成並檢查 A6 Arena 的固定 gate；不通過就先分析，不跑 Arena 後續或調整門檻。
2. 完成 A2–A5 缺少的 2–4-max 配對與 A7；逐位置看頻率、配對 EV 與區間。
3. Arena 通過後執行 scripts/experiments/network_opponents.py，使用新的 --out；GPU 一次只跑一個工作。
4. 其他 GPU 工作結束後跑全專案 .venv/bin/pytest -q，更新 README 與 HANDOFF，再依原計畫重啟網站並檢查 HTTP 200。
5. 列出結果、限制與尚未 solver-backed 的項目，再詢問使用者是否 commit。git push 由使用者自行執行。

舊段落中的 PID 與「正在執行」狀態是歷史紀錄；續跑時以本暫停檢查點及實際輸出檔為準。

## 本輪最新狀態

- A6 配對 CPU 診斷已完成：兩組同用 v7 frozen table（SHA-256 fc4df14c…259d53）、public range、rule 對手、seed 20261031，每個 5–9-max 各 1,000 deals。比較組是合成的「價格 ≤25% 時把 fold 改成 call」控制組，不是歷史版本。差值：5 +9.77 [+2.57,+16.97]、6 +6.03 [−3.12,+15.19]、7 +16.70 [+9.24,+24.16]、8 +12.64 [+8.00,+17.29]、9 +10.85 [+5.97,+15.73] bb/100；6-max CI 跨 0。對 rule 絕對結果 5 +23.37 [+2.67,+44.08]、6 +4.88 [−12.64,+22.39]、7 +11.10 [−6.37,+28.57]、8 +16.94 [+0.89,+32.98]、9 −1.90 [−20.30,+16.49]。逐位置便宜價格機會有低於 30 的格子，須由 Arena 合併對手樣本驗收。完整報告：out/web-realtime/a6-floor-control-paired-rule-1000.json。
- 曾啟動每桌 3,000 deals 的擴樣，估計超過 30 分鐘後依等待限制停止 PID 116970；只保留 partial log out/web-realtime/a6-floor-control-paired-rule-3000.log，沒有把它當成完成報告。
- A7 前全專案測試最新結果：.venv/bin/pytest -q 249 passed in 652.34s。A7 後 `tests/test_web_realtime.py` 已重跑，52 passed；Arena 結束前尚未重跑全套。
- A1 補測：`out/web-realtime/a1-paired-3-4-rule-station-3000.json` 已有 3-max、4-max vs rule；A1−A0 分別 +0.18 [−3.22,+3.58]、+1.91 [−0.79,+4.61] bb/100，兩者 CI 都跨 0。PID 131294 的 3-max station 配對已完成：A1−A0 −30.72 [−53.89,−7.55] bb/100，顯著為負；A1 絕對 +1092.98 [1021.11,1164.85]。負向差異集中於 SB 的 `limped:cheap`，A1 fold 38.5%、call 30.4%，A0 fold 27.8%、call 41.2%，raise 均 31.1%；各位置 EV 差 CI 都跨 0。PID 131294 的 3–4-max station 配對已完成。4-max A1−A0 −91.99 [−122.15,−61.82] bb/100，顯著為負；CO −85.59 [−149.51,−21.66]、BTN −63.04 [−124.68,−1.40]、SB −180.63 [−253.32,−107.94] 均顯著為負，BB −38.68 [−80.04,+2.68] CI 跨 0。4-max 只改變 SB limped:cheap 的 fold/call：A0 28.1%/41.1%，A1 39.7%/29.5%，raise 均 30.8%。PID 132702 的 5-max vs station 配對已完成：A1−A0 −144.33 [−182.15,−106.50] bb/100，顯著為負；A0 +1756.44 [1652.84,1860.04]，A1 +1612.12 [1511.68,1712.55] bb/100。逐位置差 HJ −16.18 [−61.73,+29.36]（CI 跨 0）、CO −156.65 [−237.38,−75.91]、BTN −290.65 [−407.57,−173.74]、SB −160.74 [−250.87,−70.60]、BB −97.40 [−178.38,−16.43]。A1 只改變 BTN/SB 的 limped:cheap：BTN fold/call 24.3%/37.6%→33.7%/28.2%，SB 31.6%/42.3%→45.5%/28.4%，raise 不變；負向差異跨到其他位置的整手損益，不能解讀成所有位置都該放寬。6-max station A0 baseline 已完成 3000/3000；A1 candidate 5–9-max station 配對已完成。9-max A0 +2099.44 [1985.21,2213.67]、A1 +1700.45 [1593.22,1807.67]、A1−A0 −398.99 [−454.47,−343.52] bb/100；UTG CI 跨 0，其餘位置顯著為負。9-max preflop_contexts 顯示僅 UTG+2/LJ/HJ/CO/BTN/SB 的 limped:cheap 改變；fold 46.6/45.1/44.6/40.2/29.9/33.6%→68.7/66.7/64.6/58.8/42.6/48.9%，call 38.4/39.7/38.4/39.2/37.6/41.9%→16.3/18.0/18.5/20.6/24.9/26.7%，raise 不變；每節點 n=3000。八個固定 shard 以 seed 20261029 完成，逐筆同 deal index，結果已合併至 paired JSON。輸出 `out/web-realtime/a1-paired-station-5-9-3000.json`。5–9-max vs rule 已有報告：A1−A0 在 5-max +5.41 [+1.16,+9.66]，6–9-max CI 跨 0；但 A1 對 rule 絕對 bb/100 在 5–9 桌型都顯著為負，例如 6-max −52.17 [−71.62,−32.72]。表格檢查 148,720 個機率列皆加總為 1；`72o` 便宜跟注情境由 A0 35% 繼續降到 A1 約 1%。PID 133318 的 2-max CPU 配對已完成。對 rule：A1 +36.53 [+24.20,+48.86]、A1−A0 +0.34 [−6.47,+7.16] bb/100；對 station：A1 +106.19 [+68.38,+144.01]、A1−A0 +13.35 [−1.34,+28.03] bb/100。兩個配對差都未顯著，逐位置差的 CI 也都跨 0。A1 的 2-max 覆蓋完成；3–9-max station 配對已完成。配對程式已核對：A0/A1 共用 `derive_seed(seed, n, deal)` 發牌，且每位 hero／對手的 seed 也相同。配對腳本已增加 A0/A1 log 前綴；分片結果已按固定 deal index 合併。
- A1 表格產生器檢查：scripts/experiments/web_preflop_tables.py 已不含逐手 35% 繼續下限，但仍用手牌排名 sigmoid 配合人工 WIDTHS／BB_DEFENDS 產生各格頻率；這是位置／情境校準的 heuristic，不是每節點 EV 或 solver 推導。A1 可診斷移除原硬性下限的影響，不能稱為 GTO；後續版本仍須以已驗證 EV／solver 頻率替換或保留清楚的 heuristic 標示。
- A1 station 配對新增 6–8-max：A1−A0 分別為 −235.08 [−282.62,−187.55]、−258.92 [−306.74,−211.11]、−312.38 [−363.38,−261.38] bb/100，三者 95% CI 都低於 0；A1 絕對成績分別為 +1607.27 [+1505.03,+1709.52]、+1643.25 [+1539.69,+1746.81]、+1693.98 [+1588.97,+1798.99] bb/100。位置差集中在後位：6-max LJ CI 跨 0，其餘五個位置為負；7-max UTG 跨 0，其餘為負；8-max UTG/UTG+1 跨 0，其餘為負。頻率變動只出現在 limped:cheap，受影響位置的 call 轉為 fold、raise 不變。已完成的 station paired JSON 含 5–9-max；A1 station 配對已完成。
- runtime 稽核發現 A4 opener split 沒有套用在目前 WebRealtimeAgent：`context()` 會產生 opener＋size key，但 `web_preflop_v7.json` 是 A3 表，3,168 keys 中沒有 exact `bb_vs_open_single_LJ_medium`；`lookup()` 會退回 early/late A3 key。v4–v6 有 A4 keys，但目前 runtime 指向 v7。A1 結束後須用配對評估確認 A4 表再決定 runtime 表與測試。
- 完整翻前節點尚未達成：目前 context key 只粗分人數、Hero 位置、raise/caller、部分 opener/size 與 cheap/normal；除 5–20 BB 的既有 push/fold 分流外，沒有 deep-stack effective-stack、ante/rake、底池尺寸、尚未行動人數或 call 是否關閉行動欄位。SB 表也沒有明確建模 BB 尚未行動的參與風險。A1 完成後，A2/A3 需要依此補足並逐位置驗證。
- A1 對 `rule` 的 2–9-max 結果已完整，單位 bb/100，95% CI：

| 人數 | A1 絕對結果 | A1−A0 配對差 |
|---:|---:|---:|
| 2 | +36.53 [+24.20,+48.86] | +0.34 [−6.47,+7.16] |
| 3 | +18.46 [+7.21,+29.70] | +0.18 [−3.22,+3.58] |
| 4 | +23.13 [+10.89,+35.37] | +1.91 [−0.79,+4.61] |
| 5 | −26.04 [−45.73,−6.35] | +5.41 [+1.16,+9.66] |
| 6 | −52.17 [−71.62,−32.72] | +4.17 [−0.23,+8.57] |
| 7 | −35.70 [−52.11,−19.28] | +3.22 [−1.89,+8.32] |
| 8 | −34.56 [−51.39,−17.73] | +2.23 [−2.38,+6.83] |
| 9 | −46.48 [−61.26,−31.71] | +2.13 [−2.97,+7.23] |

只有 5-max 的 A1−A0 顯著為正；A1 絕對結果在 5–9-max 全部顯著為負。A1 尚不能視為完成策略改善驗收，station 對照仍在跑。A1 vs rule 的位置頻率審核共 44 個位置：位置 fold rate 為 22.95–81.93%，沒有 0%／100%；各位置 cheap 樣本均至少 30，cheap fold 最高 43.6%（4-max BB）。BB defense 合計最低 24.0%（9-max，368 次），該總數混合 opener／caller 情境，需用分流後指標判斷。
- 絕對成績交叉檢查：A2–A4 的 5–9-max vs rule 結果仍全部顯著為負；A3−A2 雖在 9-max 有 +3.54 [+1.06,+6.03] bb/100 的配對改善，A3 絕對值仍 −42.03 [−56.48,−27.57]。Range-source 單變因診斷以同一張 A3 v3 翻前表比較 uniform/public range；public range 對 rule 的 5–9-max 絕對點估計依序 +20.00、+7.65、+8.69、+11.31、+4.39 bb/100，只有 5-max、8-max 的 95% CI 高於 0。這是 range heuristic 診斷，不能當作完整 A5、solver 或 GTO 驗收；仍缺 2–4-max 與其他對手。
- 情境診斷：新 A1 2–4-max vs rule 報告中，3-max BB 面對 open 無 caller n=524、fold 68.1%；已有 caller n=56、fold 62.5%。4-max 分別 n=715、fold 60.4%，以及 n=164、fold 64.6%，方向不一致，不能用全域 BB 放寬／收緊概括。SB 面對 open 且前面無 caller 時，3-max fold 71.6%（n=348）、4-max 72.8%（n=591）；此時 BB 仍未行動。4-max SB 面對 open+caller fold 80.0%（n=60）。舊 5–8-max A1/A2 報告沒有 preflop_contexts，無法還原這些節點；新完成的 9-max station paired row 已含 preflop_contexts，可直接核對 limped:cheap 頻率。
- A3 vs rule 的 5–9-max 情境頻率中，BB multiway fold 71.2–82.9%，均高於 single-open 的 64.2–76.5%；SB 面對 open+caller fold 79.3–85.7%，高於前面無 caller 的 74.8–82.8%。A3−A2 的 9-max BB 配對 EV +31.89 [+9.51,+54.28] bb/100，其他桌型 BB 區間跨 0。此結果方向符合按單挑／多人底池分流的假設，但目前只有 rule、5–9-max，尚不足以通過全桌驗收。
- v7 Arena gate 固定 3,000 deals × 16 組，seed 20261029、table SHA fc4df14c9a056b3edc0a0b323770d84f878f4ce8c3aecb8bf051bc7b1a259d53；M9 rule、門檻與種子未修改。原序列 PID 121067 已寫妥 2–4-max 結果後由 PID 終止；CPU-only worker PID 154728–154730 正跑 7–9-max，均用 `CUDA_VISIBLE_DEVICES=""`，seed／table hash 不變。19:15 checkpoint：7-max station 2600/3000、8-max 2060、9-max 1720；5–6-max station 已完成；每個 worker 的 rule 組已完成。PID 154693 等所有結果後合併至 `out/web-realtime/a6-arena-v7-3000.json`；19:35 recheck：7–9-max Arena workers 仍在執行；GPU 0%、639/8188 MiB、44°C、2.07 W，nvidia-smi 未列出 compute process；WSL MemAvailable 約 10.2 GB。PyTorch CUDA 可用，CUDA 12.8，device NVIDIA GeForce RTX 4060 Laptop GPU；network_opponents.py cache 現以模型 SHA-256、spots 與 backend source SHA 命名；舊 cache 保留但不採用。下次會為八個模型建立新 cache，於 CUDA 重算 40 spots exploitability；cache key 測試 1 passed。首組 2-max vs rule +41.65 bb/100，95% CI [+28.35,+54.95]；BTN/SB VPIP/PFR 79.8%/66.7%，BB 對單開池 defense 67.1%、cheap fold 57.7%（111 次）；翻後 p95 6,076.96 ms，2 秒 solver budget。此項不是完整 Arena gate。
- 舊 out/web-realtime/arena-v2-3000.json 的 acceptance 對 5–9-max rule 顯著為負，但其 table SHA 不等於目前 v7，不能當成本輪候選的驗收結果。
- 網站 PID 116075 已依使用者授權停止，供 Arena 與 GPU 回歸使用；supervisor PID 123459 會等 Arena 結束、GPU 記憶體低於 1 GB，再等全專案測試完成後的 `out/web-realtime/restart-web-after-tests.allow` 標記才啟動新版並檢查 HTTP 200。程式、log、PID 檔分別在 `out/web-realtime/restart_web_after_arena.py`、`out/web-realtime/restart-web-after-arena.log`、`out/web-realtime/restart-web-after-arena.pid`。標記建立前網站維持停止。
- backend/opponent/m9.py 的 SHA-256 與 out/m9/m9-gates.sha256 相符。M9 rule、門檻、種子未修改。新增 simulate(..., deal_start=0) 以保留固定 deal seed 做分片；test_simulation_deal_shards_replay_the_same_seeded_deals targeted test 1 passed in 2.88s。
- 169 格表仍是 heuristic 資料格式。Deep-stack solver-derived 頻率／A7 未完成；要以 CFR/MCCFR 或可驗證參考解產生，保留抽象、迭代、EV 與 exploitability／誤差界，不宣稱 heuristic 表是 GTO。現有 tabular CFR+ 僅 Kuhn/Leduc、MCCFR 是短籌碼 push/fold、GPU 子賽局是 HU 翻後；三人以上要另定義多人一般和的均衡與 best-response 驗收，不能直接沿用雙人零和 NashConv。
- A7 多人 bluff 候選目前預設關閉。最新 20-deal 3-max CPU 冒煙配對 v5 有 20 個候選點、11 個模型估計下注優於 check、實際抽到 1 次 bluff；A7−A6 = −7.17 [−21.21,+6.88] bb/100，沒有收益結論。已修正 chips→BB 單位，並輸出 raise／check／差值 EV、估計 profitable 點數與抽樣頻率；這些 EV 仍是未校準 proxy。報告 `out/web-realtime/a7-bluff-smoke-3max-20-v5.json`；v2–v4 是先前診斷快照，勿當目前候選結果。
- A7 完整配對 CPU 評估已啟動：PID 128742；3–9 人對 `rule`／`station` 各 3,000 deals，固定 seed 20261029，A6/A7 共用 v7 表與 public range，只有 candidate 開啟 `multiway_bluff`。輸出 `out/web-realtime/a7-bluff-paired-3000-14groups.json`、log/PID 同名。此路徑只跑 3–9 人 CPU heuristic；A1 完成後已恢復；Arena 剩 3 個 CPU worker，均隱藏 GPU。A7 配對報告已完成 rule 對手的 3-max 與 4-max：3-max A7−A6 +3.13 [+0.04,+6.21] bb/100（BTN +3.95、SB −1.83、BB +7.27）；4-max +2.08 [+0.15,+4.00]（CO +3.38、BTN +2.24、SB +0.70、BB +2.00）。3-max 選中 293 次 bluff，估計 raise edge +0.63 BB、其牌局總淨結果 −0.30 BB；4-max 選中 346 次，估計 edge +0.70 BB、牌局總淨結果 +0.41 BB。後兩者是含整手變化的觀察值，不是單一動作 EV；5-max A7−A6 +1.31 [−0.48,+3.10] bb/100（不顯著）；位置差 HJ +2.45、CO +1.21、BTN +3.05、SB −1.52、BB +1.36。此組抽到 297 次 bluff，平均策略頻率 8.8%，全部機會平均估計 raise edge −0.93 BB、選中者 +0.89 BB，而選中 bluff 牌局淨結果 +0.01 BB；後者不是單一動作 EV。6-max A7−A6 +2.93 [+0.82,+5.04] bb/100，位置差 LJ +0.79、HJ +2.98、CO +3.03、BTN +8.04、SB +1.55、BB +1.18。抽到 282 次 bluff，平均策略頻率 6.7%，全部機會平均估計 raise edge −0.99 BB、選中者 +1.01 BB，選中 bluff 牌局總淨結果 +1.72 BB（不是單一動作 EV）。目前 JSON 完成 rule 對手 3–6-max；19:15 的 8-max vs rule log 到 1300/3000。PID 128742 已於 19:05 以 SIGCONT 恢復；仍需看其他桌型及 station／風格對手，並考慮多組比較。
- 另有 A7 風格對手配對 CPU 評估：PID 129771，對 `nit`／`tag`／`lag`／`maniac`，3–9 人桌各 3,000 deals，seed 20261029；baseline A6、candidate A7，只有 candidate 開啟 bluff，輸出 `out/web-realtime/a7-bluff-paired-3000-style-28groups.json`。3-max vs nit 已完成：A7−A6 +5.88 [+3.01,+8.75] bb/100（BTN +10.60、SB +3.41、BB +3.64）。4-max vs nit 亦完成：+8.67 [+4.83,+12.51]（CO +6.65、BTN +4.27、SB +10.92、BB +12.83）。4-max 抽到 401 次 bluff，全部機會平均估計 raise edge −0.75 BB、選中者 +1.08 BB，選中 bluff 牌局淨結果 +3.45 BB；同樣不是單一動作 EV。6-max vs nit 已完成：A7−A6 +1.88 [+0.04,+3.71] bb/100；7-max vs nit log 到 680/3000（19:15）。PID 129771 已於 19:05 以 SIGCONT 恢復；此組為 CPU 工作，與 Arena 7–9-max 並行。
- 版本控制：全部未 commit；GPU 驗收與文件更新完成後再詢問是否 commit。git push 由使用者自行執行。

## 下一步（依序；目前已暫停）

1. A1 2–9-max 對 rule/station 配對已完成，讀 paired JSON 按 A1−A0、絕對 bb/100 與逐位置頻率判斷。9-max 對 station A1−A0 −398.99 [−454.47,−343.52] bb/100，顯著為負；負差主要伴隨 UTG+2 至 SB 的 VPIP 下滑、PFR 不變。保留 A1 的失敗結果，不修改最低繼續率或門檻。
2. 完成 v7 Arena，固定 3,000 deals × 16 組；PID 121067 已在 2–4-max checkpoint 後結束。PID 154726–154730 以 CUDA 隱藏環境並行跑 5–9-max，PID 154693 負責依 seed／table hash 驗證後合併至原報告；讀完整組別及最終 `acceptance`，門檻、seed 20261029 與 table SHA 不可改。
3. A2–A5 的 5–9-max vs rule 舊報告已核對；2–4-max 缺口已於 19:09 啟動四個 CPU-only paired jobs（PID 158229–158232），均 3,000 deals、rule/station、seed 20261029、2 秒。A5 固定 v3 表並只比較 uniform/public range。19:36 checkpoint：四個 CPU 評估都在第一組 2-max vs rule 的候選執行，A2 1480/3000、A3 1440/3000、A4 1480/3000、A5 1100/3000；四個程序仍活躍。輸出為 out/web-realtime/a2-paired-rule-station-2-4-3000.json、a3-paired-rule-station-2-4-3000.json、a4-paired-rule-station-2-4-3000.json、a5-range-paired-rule-station-2-4-3000.json；各版本凍結表 SHA 與舊報告一致。
4. A1 階段已完成，PID 128742、129771 已恢復 A7 配對 CPU 工作；逐位置分析 rule、station、nit、tag、lag、maniac，估計 bluff EV 仍只能標為 proxy，A7 預設關閉直到完成驗證。
5. Arena 完成且 GPU 空閒後，跑 `scripts/experiments/network_opponents.py`，使用新 `--out`，保留既有報告。
6. GPU 工作結束後跑 `.venv/bin/pytest -q`。完成驗證後統一更新 README 與 HANDOFF，再建立 `out/web-realtime/restart-web-after-tests.allow`，確認網站 HTTP 200。
7. 列出各版本結果、未解問題與 solver 限制，詢問使用者是否 commit；不要執行 git push。
A5 rule 評估命令（已完成）：
.venv/bin/python -u scripts/experiments/web_policy_ab.py --baseline backend/agents/data/web_preflop_v3.json --candidate backend/agents/data/web_preflop_v3.json --baseline-label A4-uniform --candidate-label A5-range --baseline-range-source uniform --candidate-range-source public --players 5 6 7 8 9 --opponents rule --deals 3000 --seed 20261029 --out out/web-realtime/a5-range-paired-rule-3000.json

## M9：設計與歷史評估

**做法**（README「M9 驗收結果」）：`Exploit`（`backend/solver/subgame_gpu.py`，data-biased response）：對手在每個決策點以機率 p 照模型、1 − p 自由行動，我方 CFR 對混合體求最佳策略；p = 0.8 × n/(n+50)。`backend/opponent/exploit.py`：`exploit_from_model`、精確評估 `evaluate_spot`、`style_policy`、`history_model`。`RealtimeAgent(exploit=True)` 預設開啟，但**沒有參考模型就不剝削**（`DEFAULT_REFERENCE = out/m9/models/gto.json`）。

**第一版（絕對頻率）被否決**：精確評估 `out/m9/spots_absolute.json`：對 5 種風格 Agent 多贏 +5～+46% 底池，但對 6 種網路 Agent 全部**變差** −2～−10% 底池。原因：模型記的是情境的整體頻率，依手牌／範圍決策的對手在單一局面的頻率不同，硬套整體平均會誤導。

**第二版（相對傾向）也未過關**：30 題完整結果對 `net-mlp-small` −2.948 ± 0.914、`net-mlp` −0.153 ± 0.061、`net-rnn` −0.114 ± 0.069% 底池，三者仍顯著為負。其他五種網路沒有顯著負值；五種風格都為正，station +30.07 ± 9.56% 底池。參考模型的河牌 `unbet` 只有 12 次、`vs_bet` 只有 1 次觀察。兩個額外診斷也已否決：只要求參考情境至少 20 次觀察會消除風格收益；固定局面 GTO 的倍率模型仍讓 MLP 虧損。診斷資料在 `out/m9/guard_probe.*`、`fixed_probe.*`，固定模型診斷已用 PID 63435 停止。

**目前版本（證據限制）**：倍率公式、p 與 Arena 門檻不變。新增 `wide_range_evidence` 與 `overfold_evidence`：VPIP 雙方至少 20 次觀察、95% 區間不重疊且對手較寬時，才完整使用倍率；其餘只在 `*_vs_bet`、至少 20 次觀察、參考有實際觀察、棄牌平均高於參考、棄牌 95% 下界 > 1/3 時處理過度棄牌。solver 再逐節點檢查實際下注尺寸的損益兩平棄牌率；只調整棄牌倍率，跟注／加注倍率維持 1。M9 的 M8 範圍調整也套用相同證據條件，且過度棄牌時保留跟注／加注比例、不降低棄牌頻率。`exploit=False` 的獨立 M8 行為不變。

這個版本目前會停用八種網路對手與 style-nit 的轉牌／河牌調整，保留其他四種風格的完整倍率；因此網路的零差距代表沒有啟用剝削，不能宣稱成功剝削網路。其共同規則型翻牌的過度棄牌仍可能通過證據條件，必須由 Arena 驗證。對其他寬範圍、依手牌決策的對手仍未驗證。針對性測試 13 項通過；完整測試首次為 237 passed、1 個舊 E2E 底牌保密斷言失敗，改成符合結算後揭露規格後，桌面 E2E 單項重跑通過；該 M9 工作交接當時尚未重跑完整套件；本輪 Web A1–A6 的全專案 pytest 已通過 248 項。精確評估在沒有啟用節點時直接重用 GTO 策略，避免重複求解造成接近零的浮點差異。

E2 用的 GTO 對手模型 `gto-opponent.json` 要用 seed 10，不能和參考模型同一批資料。Arena 的 E1／E2 門檻未修改；`out/m9/m9-gates.sha256` 可核對 `backend/opponent/m9.py`。

另外修正了決策說明的數值標示：`cfr.exploitability()` 的最佳回應沒有套用 Exploit 約束，因此原本的 `restricted-game gap` 不正確，已改成 `unrestricted exploitability`，README 與 GPU 測試的文字同步。只修正標示，數值與策略、驗收門檻均未變動。

**M9 狀態（歷史交接）**：

1. 證據限制版未通過前置精確評估，因此沒有進入 M9 Arena；E1／E2 門檻與 M9 rule 保持不變。
2. 網路對手完整評估已完成。當時全專案測試曾有 237 passed、1 個舊 E2E 斷言失敗；之後該斷言修正，桌面 E2E 單項通過。本輪全專案 pytest 最新結果為 248 passed。
3. 本文件頂端的 Web A1–A6 狀態與下一步已取代本節當時的活動工作清單。git push 由使用者在終端機執行。

## 網路對手群（使用者要求：各種架構的網路當對手）

`backend/policy/model.py`：`build_model(config)`，架構 `mlp`／`linear`／`resmlp`／`cnn`（牌面 4×13 影像 + 13×13 起手牌範圍圖）／`rnn`（GRU）／`lstm`／`transformer`。`python -m backend.policy train ... --arch <name>`。Arena kind `net-<name>`（`backend/arena/agents.py` 的 `NETWORK_OPPONENTS`；只有轉牌／河牌由網路決策，翻前與翻牌是共用的規則型邏輯）。測試集（全部 V1+V2、12 epoch、批次 8192）：

| 模型 | 目錄 | 參數 | KL | EV regret % |
|---|---|---|---|---|
| linear | `out/policy/arch-linear` | 2.8 萬 | 0.387 | 1.87 |
| mlp-small | `arch-mlp-small` | 22 萬 | 0.128 | 0.24 |
| mlp（M7 主模型） | `run-v1v2` | 85 萬 | 0.093 | 0.19 |
| resmlp | `arch-resmlp` | 183 萬 | 0.093 | 0.10 |
| cnn | `arch-cnn` | 206 萬 | 0.139 | 0.21 |
| rnn（GRU） | `arch-rnn` | 118 萬 | 0.106 | 0.15 |
| lstm | `arch-lstm` | 153 萬 | 0.105 | 0.14 |
| transformer | `arch-transformer` | 75 萬 | 0.105 | 0.14 |

已知：同一組 40 題上，linear 的可被剝削量（河牌中位數 6.8%、轉牌 18.7%）遠低於 M7 MLP（12.5%、53.5%）——模仿越準不代表越難被剝削。完整評估已完成，詳細數值見 README「對手池」；各 Arena 信賴區間都涵蓋 0，未證明任一架構對這兩種對手有顯著優勢。主模型仍是 M7 MLP，沒有更換。

## 已決定的事（使用者確認過）

- 即時求解每個決策時間預算 **10 秒**；第一版只做**單挑**；深籌碼翻前用規則型範圍並標示。
- 驗收門檻在實驗前訂好，不事後調整；「是否接近 GTO」以 exploitability 為準。
- M8：同意改用門檻版本 2（G2′）並用新種子重跑（已通過）。
- 對手模型接進即時求解 Agent（已完成）；之後直接做 M9（進行中）。
- 用各種網路架構訓練對手（含 CNN、RNN、LSTM，使用者指定）。

## 環境注意事項與這次學到的教訓

- **Python**：`.venv/bin/python`（3.13）；**GPU**：RTX 4060 Laptop 8 GB；**Node.js** 22（`~/.local/opt`）。
- **GPU 一次只跑一個工作**：訓練資料約 2.9 GB 放在 GPU，CNN／Transformer／GRU／LSTM 每步再 1.8～2.4 GB。和其他 GPU 工作同時跑時超過 8 GB，WSL 會溢出到系統記憶體、速度掉好幾個數量級（GRU 曾因此 7.5 小時跑不完 1 個 epoch；單獨跑只要 9 分鐘）。程序結束後 WSL 釋放記憶體有延遲，下一個工作開始前先等 `nvidia-smi` 的使用量降到 1 GB 以下。
- **輸出緩衝**：背景執行並導向檔案時 `print` 會整塊緩衝，看起來像「卡住」。我曾因此誤殺兩次正常的 LSTM 訓練。判斷進度看 checkpoint 檔時間，或用 `python -u`（`train()` 的紀錄已改成即時輸出）。
- **`pgrep -f`／`pkill -f` 會殺到自己**：同一個指令裡含有要比對的字串時會 exit 144（這次又踩到一次，連佇列一起殺掉）。用 PID，或用輸出檔判斷狀態。
- **資料（不在 git 裡，可重建）**：`out/datasets/gto-postflop-hu-v1`、`-v2`；`out/policy/v1v2-features.npz`；各模型 `out/policy/*/model.pt`；M8／M9 報告在 `out/opponent/`、`out/m9/`。
- **git push**：從 Claude 的工作階段推送會卡在 Windows 的 Git Credential Manager；請使用者自己執行 `cd ~/poker && git push`。

### 快照 API 實戰調整（2026-11-08）

`POST /api/ai/sessions/{sid}/snapshot-decide` 現況（向下相容：`view.history` 為空時行為同前，只有 `label` 改為依實際來源顯示）：

- **輸入**：`view.history` 可傳完整事件（codec 六欄，含 post_ante／post_blind／post_live／post_dead、action、deal_board）；新增選填 `player_ids`（seat → player_id）。事件格式不對回 422；格式對但籌碼或牌面與快照對不上時不報錯，當作沒給紀錄並在回應 `history` 寫 `ignored: 原因`。
- **翻後**：有紀錄走 `WebRealtimeAgent`（session 的 mode／seconds／adaptive／multiway_bluff 生效，同手以 `hand_id` 保留記憶）；沒紀錄或代理失敗才退回 RuleBasedAgent，`approximations` 含 `rule_based_fallback`。
- **翻前**：有紀錄用真實紀錄，並可用 adaptive 防守節點；沒紀錄維持由下注額推算（`history_inferred_from_bets`）。
- **回應新增**：`node`（查到的節點）、`action_probs`、`history`、`approximations`（`dead_button_shifted`、`forced_post_as_limp`、`pushfold_table_has_no_ante`、`chart_has_no_ante` 等，可用來統計頻率）、`in_hand_memory`。
- **修正**：`web_preflop.ranges` 重播時漏傳 ante，前注桌的短全下會被判不合法。
- **推棄表前注**：求解器支援 `--ante`；48 組**尚未重算**。若資料放到 `data/preflop/pushfold_ante_0.5/`，`PushFoldAgent` 會在桌面前注落在 0.375–0.625 BB 且該人數有資料時使用（重啟 API 才載入），否則維持無前注表。
- **已知限制**：單挑求解有最少迭代數，`seconds: 0.5` 實測單手 1–6 秒，程序冷啟動第一次約 36 秒。

pkui 要配合的項目：移除 `poker_ai_api.decide_snapshot` 對 `history != []` 的拒絕並送出完整事件與 `player_ids`；`_normalize_decision` 帶出 `node`／`approximations`／`history` 寫進 trace；每手結束 `POST /sessions/{sid}/hands` 寫回牌局（第 8 項）。

後續（2026-11-09）：
- **死小盲**：`_snapshot_preflop` 在按鈕與大盲之間放一個已棄牌的空位，查多一人的表（`dead_small_blind_empty_chair`），不再退回 RuleBasedAgent。9 人桌或按鈕也空著時仍退回。
- **極短籌碼**：推棄表補算 0.5 BB 前注的 2、3、4 BB（4 BB 不能省，否則 3.75–4 BB 會落在 3 BB 與 5 BB 的缺口），涵蓋到有效籌碼約 1.5 BB。更短的仍查翻前表。
- **補盲**：維持當 limp。實戰約 10.6% 的牌局有補盲（179 手可判讀中 19 手）；要加節點需新的表與驗收，尚未做。

未做：策略內近似（優先度低）。

### 0.5 BB 前注翻前表（A9）驗收門檻（2026-11-09 00:13 CST，使用者訂定、執行前寫下，不事後調整）

- 模型：以 v8.3 為基礎，只改 open、limped、option、vs_open（含 SB／BB）、bb_limped。範圍寬度（跟注與加注）× ((1.5 + 0.5n) / 1.5)^k，n 是人數；k 候選 0.15、0.25、0.35。跟注範圍上限 85%（原本已超過的不縮小），面對開池的加注範圍維持 13% 上限。下注尺寸不變。產表：`scripts/experiments/web_preflop_ante_tables.py`。
- 線上啟用條件（通過後才接）：前注 ÷ 大盲在 0.4～0.6 才用新表，否則用 v8.3。
- 比較：`web_policy_ab.py --baseline web_preflop_v8_3 --candidate 新表 --ante-bb 0.5 --deals 3000 --seconds 2`，同牌配對。
- 靜態：未修改的節點和尺寸完全相同；修改的節點中沒有任何牌的繼續（跟注＋加注）或加注機率比 v8.3 低。
- 100 BB 不變差：2–9 人 × 10 種對手（rule、station、nit、tag、lag、maniac、iso、squeeze、minraise、threebet），80 組，每組 95% CI 上界 ≥ 0。（修訂：原訂 9 種 72 組；工具實際有 10 種，使用者於正式驗收啟動前、看任何正式結果前決定全部納入。）
- 40／150 BB 不變差：2–9 人 × {rule, lag}，32 組，每組 95% CI 上界 ≥ 0。
- 一定要改善：6–8 人 × {rule, tag, nit}，100 BB，9 組合併後 95% CI 下界 > 0。
- 選 k：設計種子 20261120 跑三個候選值，取 9 組合併成績最高者，平手取較小的 k。正式驗收用種子 20261121。
- 「9 組合併」的算法：9 組平均值的平均，標準誤 = √(Σ各組標準誤²) / 9（各組視為獨立）。
- 沒通過就不上線：維持 v8.3 並先分析，不事後調整門檻。112 組裡即使沒有差異也可能有 1～2 組偽失敗，結果照實記錄。
- 上線後：看前 200 手的 VPIP、PFR，以及有沒有查表失敗。

#### A9 結果（2026-11-09）：沒通過，線上維持 v8.3，新表未接線

- 選 k（設計種子 20261120，9 組合併）：k=0.15 +24.02 [+20.53, +27.50]；k=0.25 +35.80 [+31.49, +40.12]；k=0.35 +51.74 [+46.86, +56.62]。依規則選 k=0.35（候選上限，成績隨 k 單調上升）。
- 正式驗收（種子 20261121，k=0.35，112 組全跑完）：靜態檢查通過；「一定要改善」通過，+44.26 bb/100，CI [+39.51, +49.02]；「不變差」15 組失敗，全部在 100 BB：minraise 3–9 人共 7 組（-17.49～-39.83），threebet 2–9 人共 8 組（-17.52～-36.34）。40／150 BB 的 32 組沒有失敗。判定腳本輸出 A9 NOT PASSED。
- 這 15 組不是偽失敗：集中在兩種對手、CI 上界都明顯小於 0，threebet 8/8、minraise 7/8 同方向。
- 100 BB 各對手 2–9 人平均：rule +25.6、station +430.8、nit +48.9、tag +36.5、lag +22.7、maniac -1.7、iso +29.7、squeeze +19.4、minraise -23.9、threebet -29.1 bb/100。station 的數字反映模擬對手什麼都跟注，不代表實戰。
- 失敗原因：開池放寬後（6 人 raise 29%→42%），vs_3bet／vs_4bet 依規格沒有放寬。threebet 6 人的 vs_3bet 決策 3880→5220 次，棄牌率 74%→81%；minraise 6 人的再加注節點 5458→7090 次，棄牌率 69%→72%。多開的牌大多在被 3-bet 後棄掉，付出開池金額；threebet 組以 BTN、SB 損失最大（6 人 -58、-48），minraise 組損失較分散。這正是 v8.4 門檻要測的漏洞（面對 3-bet 棄掉約 3/4 的開池範圍），放寬開池把它放大了。
- 設計上的盲點：選 k 只看 rule、tag、nit，這三種對手都不攻擊開池範圍，所以必然選最寬的候選。
- 候選表 `web_preflop_v8_3_ante05_k{15,25,35}.json.gz` 留在 `backend/agents/data/`，程式沒有引用。要再試需另訂新候選與新種子，不能沿用 20261120／20261121 的結果。

### 0.5 BB 前注翻前表第二輪（A9d）驗收門檻（2026-11-09，使用者要求開新一輪；執行前寫下，不事後調整）

- 候選：在 A9 的修改之外，再放寬多開的牌之後會遇到的節點，讓棄牌比例不因開池變寬而上升。`vs_3bet_*`（normal、cheap）與 `vs_4bet_early`／`vs_4bet_late`（只有 cheap）的**繼續寬度** × 同一個 ((1.5 + 0.5n) / 1.5)^k，上限 85%；這些節點的加注寬度和尺寸都不變。k 候選 0.15、0.25、0.35。表 `web_preflop_v8_3_ante05d_k{15,25,35}.json.gz`。
- cold_vs_3bet、vs_4bet_cold、vs_5bet、vs_4bet:normal、all-in 節點完全不動。
- 靜態：未修改的節點和尺寸完全相同；修改的節點中沒有任何牌的繼續或加注機率比 v8.3 低。
- 選 k（設計種子 20261122，每組 3000 手，100 BB，每個 k 30 組）：2–9 人 × {rule, minraise, threebet} 24 組算「不變差」組（95% CI 上界 < 0 為失敗），加上 6–8 人 × {tag, nit} 6 組。取失敗組數最少者；同數取 9 組（6–8 人 × rule、tag、nit，算法同 A9）合併成績最高者；再同取較小的 k。
- 提前停止：被選中的 k 在這 24 組有 3 組以上失敗，就不跑正式驗收，線上維持 v8.3。24 組裡最多預期 1～2 組偽失敗，3 組以上視為真問題。
- 正式驗收（種子 20261123，只跑被選中的 k）：門檻與 A9 完全相同。100 BB 2–9 人 × 10 種對手 80 組、40／150 BB 2–9 人 × {rule, lag} 32 組，共 112 組，每組 95% CI 上界 ≥ 0；6–8 人 × {rule, tag, nit} 100 BB 9 組合併後 95% CI 下界 > 0。
- 線上啟用條件和上線後檢查與 A9 相同。沒通過就維持 v8.3，照實記錄，不事後調整門檻。
- 已知風險：v8.4 放寬 vs 3-bet 跟注時對 rule 3–5 人顯著變差（-2～-5 bb/100，無前注），所以 rule 2–9 人放進設計階段的不變差組。

#### A9d 結果（2026-11-09）：設計階段提前停止，沒跑正式驗收，線上維持 v8.3，新表未接線

- 設計種子 20261122，24 組不變差中的失敗數與 9 組合併成績（bb/100）：k=0.15 8 組，+20.24 [+16.69, +23.79]；k=0.25 10 組，+36.84 [+32.54, +41.14]；k=0.35 10 組，+51.61 [+46.70, +56.52]。依規則選 k=0.15，失敗 8 組 ≥ 3，停止；種子 20261123 未使用。
- k=0.15 的失敗：minraise 5–9 人（-10～-21）、threebet 7–9 人（-6～-14）。rule 2–9 人三個 k 都沒有失敗，v8.4 的風險沒有出現。
- 防守放寬有作用但不夠：vs_3bet 棄牌率回到約 v8.3 的水準（threebet 7 人 k=0.15：74%→75%），第一輪 k=0.35 的 threebet 損失 -18～-36 降到 +1～-17（種子不同，只能粗略比較）；minraise 仍 0～-32。
- 剩下的損失不是棄牌率造成，而是多開的牌本身：繼續比例不變時，多出來的弱牌仍有約 3/4 在被任意牌再加注後棄掉，付出開池金額。
- 結論：固定放寬開池是取捨。對不攻擊開池的對手賺（k=0.15 +20、k=0.35 +52），對任意牌再加注的探針賠，k 再小也還是有多組失敗。要同時過兩邊，得依對手被觀察到的再加注頻率決定是否放寬（與 v8.4 的「可能的下一步」同方向），這需要新門檻與新種子。
- 候選表 `web_preflop_v8_3_ante05d_k{15,25,35}.json.gz` 留在 `backend/agents/data/`，程式沒有引用。

### 快照路徑的對手資料（2026-11-09）

- `CreateSession.opponent_pool`：同名 session 共用同一份對手模型，存在 `<POKER_AI_DIR>/pools/<name>.jsonl`（不含 Hero 底牌），重啟時重建。pkui 用 `qqpk`，`snapshot-decide` 帶 `player_ids`；讀不到名字的座位不送。沒有 `opponent_pool` 的 session 行為不變。
- 只記每手 Hero 第一次翻前決策之前的動作。Hero 後面的玩家只有在有人加注時才會再出現，把後面的動作也算進去會高估他們的 3-bet 率。所以翻後統計（c-bet 等）目前沒有資料，攤牌牌也沒有。
- 重放失敗的手牌略過，不影響決策。實際日誌 177 個第一次決策有 172 個可重放，5 個略過。
- 目前線上真正會用到的只有 A8.3 的 iso／squeeze 防守（至少 8 次機會且頻率 ≥ 50%）。死小盲補空椅時會重新編號座位，這種手牌不使用對手模型。
- OCR 名字不做模糊合併，同一玩家被讀成不同名字時資料會分開累積。
- M9 的 `Exploit` 求解沒有接上：只有 2 人翻後求解器能用（`WebRealtimeAgent` 傳 `exploit=False`），證據限制版也沒通過前置精確評估。

### 0.5 BB 前注翻前表第三輪：依對手切換（A9m）驗收門檻（2026-11-09，使用者選定方向；執行前寫下，不事後調整）

- 候選：A9d 的 `web_preflop_v8_3_ante05d_k{15,25,35}` 只在以下條件都成立時使用，否則該次決策用 v8.3：前注 ÷ 大盲在 0.4～0.6；session 是 adaptive；桌上沒棄牌的對手裡，沒有人被判定為常再加注。
- 判定常再加注：`OpponentModel` 的 `three_bet` 至少 8 次機會，且 (次數 + 1) / (機會 + 2) ≥ 50%（沿用 A8.3 的 iso／squeeze 門檻）。
- 對手範圍估計（`ranges`）維持 v8.3 表，只有 Hero 自己的翻前決策換表。
- 比較：`web_policy_ab.py --persistent-models --baseline-adaptive --candidate-adaptive --candidate-ante-switch --observe-first-decision --ante-bb 0.5 --deals 3000 --seconds 2`，同牌配對。基準是 v8.3，兩邊都累積對手資料，而且都只用 Hero 第一次翻前決策之前的動作，和線上一致。
- 選 k（設計種子 20261124，每組 3000 手，100 BB，每個 k 30 組）：規則與 A9d 相同。2–9 人 × {rule, minraise, threebet} 24 組看失敗數，加 6–8 人 × {tag, nit} 6 組；取失敗最少者，同數取 9 組合併最高者，再同取較小的 k。被選中的 k 有 3 組以上失敗就不跑正式驗收，線上維持 v8.3。
- 正式驗收（種子 20261125，只跑被選中的 k）：門檻與 A9 完全相同。100 BB 2–9 人 × 10 種對手 80 組、40／150 BB 2–9 人 × {rule, lag} 32 組，共 112 組，每組 95% CI 上界 ≥ 0；6–8 人 × {rule, tag, nit} 100 BB 9 組合併後 95% CI 下界 > 0。
- 通過才設定 `web_preflop.ANTE_TABLE_PATH`；沒通過就維持 v8.3，照實記錄，不事後調整門檻。
- 已知限制：2 人桌對手的 3-bet 機會都發生在 Hero 決策之後，這種資料收集方式看不到，所以 2 人桌永遠不會切回 v8.3。冒煙測試用種子 20261126、少量手數，結果不採計。
- 上線後：看前 200 手的 VPIP、PFR、查表失敗，以及有多少決策切回 v8.3。
- 執行：`out/web-realtime/a9m_run.sh`（`setsid nohup`，輸出 `a9m-run.out`，每組的 log 在 `a9m-logs/`，判定 `a9m_gate.py`）。中斷後直接重跑這支腳本，已完成的組會自動跳過，結果只由種子決定。2026-11-09 第一次執行因 WSL 重開機中斷（完成 5/90），已從中斷處接續。

#### A9m 結果（2026-11-10）：未通過（112 組有 2 組失敗），線上維持 v8.3，`ANTE_TABLE_PATH` 仍是 None

- 選 k（種子 20261124）：三個 k 都是 0/24 失敗；合併 k=0.15 +14.79、k=0.25 +28.01、k=0.35 +41.52 [+37.26, +45.78]，依規則選 k=0.35，沒有提前停止。
- 正式驗收（種子 20261125，k=0.35，112 組）：靜態檢查通過。「一定要改善」通過，+43.25 bb/100，CI [+39.16, +47.34]。「不變差」失敗 2 組：100 BB maniac 5 人 -0.14 [-0.242, -0.028]、maniac 7 人 -0.05 [-0.100, -0.005]。判定腳本輸出 NOT PASSED，門檻不改。
- 100 BB 各對手 2–9 人平均（bb/100）：rule +36.2、station +348.4、nit +40.9、tag +28.5、lag +21.0、maniac -1.3、iso +30.5、squeeze +30.8、minraise +3.1、threebet +0.9。40／150 BB 的 32 組沒有失敗。第一、二輪失敗的 minraise、threebet 這輪 16 組全部沒有失敗。
- 失敗的兩組幾乎是同一套決策：maniac 5–9 人兩邊的開池決策數和加注率完全相同，翻前決策數只差 1–5 次（約 1.7 萬～2.8 萬次中）。差異只來自極少數手牌，配對差幾乎全是 0，所以 CI 很窄；這種稀疏差值下常態近似的 CI 也不可靠。這可能屬於規格預期的 1～2 組偽失敗，但規格同時規定沒通過就不上線，所以照實記為未通過。
- 要上線有兩條路，都由使用者決定：明確破例（記錄為破例），或另開一輪，事先訂好不劣性邊界與新種子。不能重跑同一輪直到通過。
- 線上資料接入已實際寫入：2026-11-10 02:25 `pools/qqpk.jsonl` 有 21 筆。

### 0.5 BB 前注翻前表第四輪（A9r4）驗收門檻（2026-11-10，使用者訂定；執行前寫下，不事後調整）

- 候選：A9m 的依對手切換，固定 k=0.35（`web_preflop_v8_3_ante05d_k35.json.gz`），不再選 k，也不改程式。比較方式與 A9m 相同（`out/web-realtime/a9m_ab.sh`：v8.3 基準，兩邊都累積對手資料，都只用 Hero 第一次翻前決策之前的動作，每組 3000 手，前注 0.5 BB）。
- 種子 20261127（20261120～20261126 都已用過）。
- 組別：與 A9m 相同的 112 組（100 BB 2–9 人 × 10 種對手 80 組，40／150 BB 2–9 人 × {rule, lag} 32 組），另做「一定要改善」的合併檢查。
- 不變差：每組 95% CI 上界 ≥ -1 bb/100（A9m 是 ≥ 0）。
- 一定要改善：6–8 人 × {rule, tag, nit}，100 BB，9 組合併後 95% CI 下界 > 0（算法同 A9）。
- 停止規則：通過才設定 `web_preflop.ANTE_TABLE_PATH`；沒通過就放棄這個方向，線上維持 v8.3，不再開第五輪。
- 執行：`out/web-realtime/a9r4_run.sh`，判定 `a9r4_gate.py`，輸出 `a9r4-run.out`。

#### A9r4 結果（2026-11-11）：通過，已接上線

- 112 組全部完成。不變差（上界 ≥ -1）0 組失敗；一定要改善 +42.70 bb/100，CI [+38.48, +46.91]。判定腳本輸出 PASSED。
- 100 BB 各對手 2–9 人平均（bb/100）：rule +32.3、station +331.5、nit +42.1、tag +25.2、lag +14.5、maniac +0.3、iso +31.8、squeeze +26.7、minraise +2.0、threebet +2.0。40 BB：rule +32.8、lag +10.3；150 BB：rule +33.6、lag +17.4。
- 照實記錄：若用 A9m 的「上界 ≥ 0」，仍有 1 組會失敗（maniac 5 人 -0.06 [-0.11, -0.00]）。單組最低平均 lag 100 BB -5.4 bb/100，但 CI 上界 ≥ 0。station 的數字反映模擬對手什麼都跟注，不代表實戰。
- 已接線：`web_preflop.ANTE_TABLE_PATH` = `web_preflop_v8_3_ante05d_k35.json.gz`。快照 API 與網頁 agent 在前注 0.4–0.6 BB、adaptive、桌上沒有已知常再加注的對手時用這張表，否則用 v8.3；`approximations` 會出現 `ante_table`。設成 None 即可關閉。
- `web_policy_ab.py` 結束時改為還原 `ANTE_TABLE_PATH` 的原值（原本會清成 None）。
- 上線後檢查：前 200 手的 VPIP、PFR、查表失敗，以及多少決策切回 v8.3（`approximations` 有 `chart_has_no_ante` 而非 `ante_table`）。

### pkui 找不到牌桌（2026-11-11）

- QQPK 新版的牌桌視窗標題改成「德州扑克 | 常规 | $0.01/$0.02($0.01) | 桌 970」，原本記錄器只抓標題完全等於 `QQPK` 的視窗，所以偵測不到牌桌。
- `poker_logger.window_matches`：標題完全相同，或在預設標題 `QQPK` 時，視窗屬於 `QQPK.exe`。瀏覽器裡標題含 QQPK 的頁面仍會被排除。修正後牌桌 970 立即被偵測到。
- 從 WSL 啟動 UI 時，用 PowerShell `Start-Process` 把輸出導到 `logs/ui-stdout-*.log`／`ui-stderr-*.log`，否則看不到記錄器訊息。
- 面對全下沒跟（2026-11-11）：對手下注超過 Hero 籌碼時，畫面只剩「棄牌」和「All in」，原本沒有任何地方認得 All in，等不到跟注按鈕就超時棄牌。現在 `poker_logger.is_allin_label`（全下／全压／All in／ALL-IN）、`state["call_is_allin"]`（輪到 Hero、沒有過牌與跟注按鈕、有 All in）→ 確認器把跟注額定為 Hero 全部籌碼 → `ui.api_decision` 允許跟注 → `operation.button_for`／`calls_allin` 把跟注對到 All in 按鈕，送出前的金額核對也接受這種情況。有跟注按鈕時 All in 不會被當成跟注。按鈕實際文字未從畫面確認，兩種語言都收。pkui 離線測試 363 項通過。
- 第二桌仍抓不到：視窗放大到 1506×1053 時，QQPK 自己畫的標題列字體不跟著放大，OCR 讀不到桌號（縮到 998 或 1200 寬就讀得到）。新增 `poker_logger.apply_window_title`：畫面讀不到桌號時，從系統視窗標題補上桌號、級別與前注，畫面讀到的值優先。修正後 970、831 兩桌都被偵測到。pkui 離線測試 357 項通過。

- 自動操作可加入其他桌（2026-11-12）：介面原本啟用自動後就鎖住牌桌下拉選單，只能全部停止再重來。現在選單一直可選：選到不在自動清單的桌，按鈕是「加入此桌」；選到已在清單的桌，按鈕是「移出此桌」；另有「全部停止」。後端 `/api/auto` 收到 `{"enabled": false, "hwnd", "table"}` 只移出那一桌（`OperationController.set_auto(False, target=...)`），最後一桌移出才整個停用；不帶 hwnd/table 仍是全部停止。
- 自動縮牌桌視窗（2026-11-12）：記錄器每看到一個標題含「| 桌 N」的視窗，就檢查一次；畫面（client 區）比 998×699 大超過 20 px 時縮成 998×699（左上角不動、不搶焦點，最大化的先還原）。每個視窗只縮一次，之後手動放大不會被改回；大廳（標題 `QQPK`）不會被縮。關閉用 `--no-fit-window`。重開後 942 被自動縮小，970 原本就是這大小沒動。pkui 離線測試 369 項通過。

### 對手資料回填驗收門檻（2026-11-12，使用者選定「從記錄補資料」；執行比對前寫下，不事後調整）

- 目的：線上 qqpk 資料池只有 53 手、49 人，沒人的 3-bet 機會超過 2 次（換表判斷要 8 次），對手模型實際上沒作用。用 pkui 從 10/25 起的記錄（約 2,300 手）補資料。
- 方法：把 `D:\pkui\logs\session-*.jsonl` 的每張畫面依序餵給線上同一套 `HandConfirmer`（含 `build_history`），在每手 Hero 第一次翻前輪到時取 `decision_snapshot`，和線上資料池記錄的時機與內容一致（只算 Hero 第一次決策前的動作）。換表門檻（8 次機會、50%）與資料池程式都不改。
- 驗證集：線上資料池的 53 手，全部找得到對應的記錄檔與手牌編號。
- 通過條件（全部要成立）：
  1. 重現率：53 手中至少 45 手回放後產生快照。
  2. 一致率：重現的手中，至少 90% 的手，每位玩家的觀察（`observe_opening` 產生的所有統計與 iso／squeeze 次數）和線上完全相同。
  3. 3-bet 與 iso／squeeze：重現的手加總後，（機會、採取）和線上的差距合計不超過 1。
- 通過：備份 `qqpk.jsonl` 後把回填的手（排除已在資料池的手，hand_id 以 `backfill:` 開頭）附加進去，重開 API 載入。沒通過：不寫入，資料池維持原樣，記錄原因。
- 結果（2026-11-12）：**通過**。53 手全部重現；48 手完全一致（90.6%，門檻 90%）；3-bet 與 iso／squeeze 差距 0。5 手不一致全是回放端「不記錄這手」（Hero 第一次輪到時重建不出歷史），沒有任何一手回放算出不同的數字；其中 4 手線上本來也沒有觀察。
- 回放 167 個記錄檔得到 1,379 個 Hero 第一次翻前輪到的快照；892 個重建不出動作歷史（籌碼對不上），線上 API 一樣不會記錄；11 個動作無法重播；53 個已在資料池。寫入 428 手（`backfill:<記錄檔>:<手號>`），資料池變成 481 手、234 人。備份：`qqpk.jsonl.bak-20261112-before-backfill`。腳本與回填檔：`out/opponent-backfill/`（`replay_logs.py` 用 WSL python3 跑 pkui 程式碼，`backfill_check.py`／`backfill_make.py` 用 `.venv`）。
- 影響：5 位玩家的 3-bet 機會已達 8 次，但沒有人被判為常再加注（最高 哇哇叫9 1/17、Vladizmaz 0/16）；iso 機會達 8 次的有 7 人，也沒人達到積極門檻（哇哇叫9 iso 5/24）。所以照現行門檻，前注換表與 v8.2 防守都不會因回填而切換，翻前決策不變；資料顯示 QQPK 對手很少 3-bet。API 重開後才會載入。
- 每次呼叫 API 都記錄對手（2026-11-12，使用者要求）：原本只有「Hero 第一次翻前決策、而且帶動作歷史」的那次呼叫會寫入資料池；那次沒歷史（重建失敗）時，同一手之後的呼叫也都不記錄。現在 `snapshot_decide` 只要快照帶歷史就嘗試記錄（`_has_opening` 取代 `_before_first_decision`），`observe_opening` 把歷史切在 Hero 第一次行動之前（起始籌碼仍用完整歷史推算），所以面對 4-bet、翻牌後的呼叫，記到的內容和第一次決策時相同；每手仍只記一次（hand_id 去重）。Hero 行動之後的動作照舊不算，避免高估 3-bet。小偏差：靠後續呼叫補記的手，只有 Hero 還有下一次決策才會出現。舊資料池重新載入結果不變。

### 11/12 實戰檢討後的修正（2026-11-12）

- 檢討報告指出的翻後問題（As5s 在 Qc Qd 5d 判成 strong 加注；3h2c 在 3d 9s 9c 價值下注；Ac2d 頂對弱踢腳跟三條街）全部來自快照 API 的 `rule_based_fallback`：pkui 沒送動作歷史時（今天 844 次呼叫中 808 次）翻後改用 `RuleBasedAgent`，不是解算器。
- `RuleBasedAgent(board_aware=True)`：只給快照 API 的後備使用（`backend/api/ai.py`）。公牌對子上的兩對算成一對（取自己手牌組成的那一對）；頂對踢腳低於 T 為 thin（翻牌價格 ≤0.33 跟，轉牌／河牌只跟 ≤0.2）；轉牌／河牌公牌出現同花色 3 張而自己沒同花時，medium 降為 thin、thin 降為 weak。預設 False：實驗把它當對手與其他 agent 的後備，已通過的驗收依賴舊行為（`test_rule_output_unchanged_on_160_seeded_hands` 雜湊不變）。測試：`tests/test_rule_board_aware.py`。**未做 A/B 驗收**，是針對實際錯誤的規則修正。
- 沒修：沒中牌全部過牌、沒有持續下注、聽牌只過牌或跟注（偏被動）。根本解法是讓 pkui 送得出動作歷史，翻後就會用解算器；後備規則只是保底。
- Hero 籌碼漏讀最前面的「1」（桌 458 12:50～12:57，155↔54.5）：大部分 API 快照走估算路徑、直接用最新畫面籌碼，所以 API 深度跟著跳。`HandConfirmer._fix_dropped_digits`：每張畫面進來就和同一座位上次讀數比，只看 100～199；補上最前面的 1 後剛好接回（扣掉新增下注，差 3BB 或 3% 內），而且從上次讀數到現在底池從沒大到裝得下少掉的籌碼才補；連續補超過 90 秒（`DIGIT_FIX_MAX`）改信讀數。回放 10/25 起全部記錄：補回的畫面 3,387 張之後會再讀回高值（確實是漏讀），108 張沒有讀回（可能是真的輸掉、底池沒讀到，最多影響 90 秒）。桌 458 第 115 手 API 起始籌碼從 51BB 修正為 151BB。12:47 的 444 → 150 不是漏讀一位數的型態，沒動。pkui 離線測試 386 項通過。

### 多人桌剩兩人時用解算器（A10）驗收門檻（2026-11-13，使用者選定規格與手數；執行前寫下，不事後調整）

- 背景：解算器（`_WebHeadsUp`／`RealtimeAgent`）原本只在整桌只坐兩人時使用；6～9 人桌翻後就算只剩兩人，也用多人範圍勝率啟發式（HEURISTIC multiway）。
- 候選：`WebRealtimeAgent(table_heads_up=True)`，翻後剩兩人、而且翻牌後沒有第三人行動時，交給單挑解算器（mode solve、seconds 0.5）：已棄牌玩家的籌碼當底池死錢，OOP＝按鈕之後第一個還在牌局的座位，翻前範圍照舊用網頁翻前表推算（`ranges(view)`）。整桌兩人的行為不變（`replay(view, oop=None)` 照舊用 BB）。基準：現行線上設定（翻後多人啟發式）。
- 共同設定：`web_policy_ab.py --candidate-table-heads-up`，兩邊都 `--ante-bb 0.5 --persistent-models --baseline-adaptive --candidate-adaptive --candidate-ante-switch --observe-first-decision`，100BB，`--deals 1500`，`--seconds 0.5`，種子 20261128（20261120～20261127 都已用過）。
- 組別：6、7、8、9 人 × {rule, station, nit, tag, lag, maniac}，共 24 組。
- 不變差：每組 95% CI 上界 ≥ -1 bb/100。
- 一定要改善：6～8 人 × {rule, tag, nit} 共 9 組合併（SE＝√Σse²／9），95% CI 下界 > 0。
- 延遲（線上可行性）：A/B 之後在沒有其他負載時，用 `hu_timing.py` 的同一種手（6 人桌 UTG 開池、BB 跟注，打到河牌，8 手、預熱後計）量候選的翻後決策時間：p95 ≤ 5.0 秒、最大 ≤ 6.0 秒。理由：使用者說 QQPK 行動時間約 10 秒，通過後 pkui 翻後保底改為 9 秒，扣掉辨識約 2 秒、點擊約 0.7 秒。
- 停止規則：任一條不通過就維持現狀（翻後用多人啟發式），記錄結果與原因；不改門檻重跑。
