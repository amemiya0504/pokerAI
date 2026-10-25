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
  - 限制：仍是 HEURISTIC 表（非 GTO）；通過只代表對 rule／station 不比 v7 差；20–30 BB 與 50–80 BB 桶、v8.1 新節點的逐位置頻率稽核尚未做；全專案 pytest 與 network_opponents.py 尚未在新電腦跑。
- 移到原本電腦開網頁：在本機 commit 到 `preflop-v8` 後由使用者 `git push -u origin preflop-v8`；原電腦 `git fetch && git switch preflop-v8 && pip install -e ".[web,sim,solver]"`，再照 README「網頁牌桌（M2）」建置前端並啟動 uvicorn。網頁的 `realtime` 座位會自動讀 v8 表；v8 表與 equity 矩陣都在 git 內，不需要 out 資料。

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
