# 交接：目前進度與下一步（2026-10-17）

給接手的新對話。先讀這份，再讀 `README.md`（各 Milestone 的驗收結果）與 `docs/realtime-agent.md`（即時求解 Agent 的設計）。原始設計文件是 claude.ai 上的「Poker AI 平台技術設計文件」（Claude Docs）。

## 進度總覽

| 項目 | 狀態 |
|---|---|
| M1 引擎、M2 網頁、M3 Arena、M4 push/fold、M5 solver、M6 資料集 | ✅ 完成 |
| M7 Policy Network | ⚠️ 模仿指標達標，但策略本身高度可被剝削（河牌中位數 9.6%、轉牌 47% 底池） |
| 即時求解 Agent A 階段（單挑轉牌／河牌） | ✅ 完成 |
| **即時求解 Agent B 階段（翻牌）** | **進行中，見下方** |
| M8～M10 | 未開始 |

最新 commit：`2d746b0`。本機比 GitHub 多 4 個 commit（見「環境注意事項」）。完整測試 150 項全過（約 7 分鐘）。

## 進行中：B 階段（翻牌即時求解）

**已完成**（`backend/solver/subgame_gpu.py`、`backend/agents/realtime.py`）：
- 深度限制樹：`BatchedSubgameCFR(..., checkdown=1 或 2)`，前 1 或 2 條街有下注，之後直接發完比牌。
- 葉節點用兩兩勝負矩陣 `EquityMatrix`（與明確展開的子樹完全等價，有測試），每個牌面快取；深度 2 用每張轉牌一個矩陣（`BatchedEquityMatrix`）。
- 兩段式遍歷 `_traverse`：同一次遍歷的所有葉節點一次批次計算。深度 1 翻牌約 2～5 ms／迭代，深度 2 約 55 ms／迭代。
- Agent 的翻牌決策與對手翻牌動作都改用求解（目前是**深度 1**）。

**偏差評估**（已完成，8 題；結果在 `out/policy/flop_depth_limit_bias.log` 與 `out/policy/flop_depth_limit_bias.json`。方法：小型翻牌樹「每街一種 75% 尺寸、不加注、SPR ≤ 4」完整求解，與深度 1／2 的翻牌策略比較；腳本在舊對話的 scratchpad，若要重跑需重寫成正式工具）：

| | 根節點策略差距（TV，平均） | 翻牌改用它、之後照完整解，面對完整解的期望值（平均） | 混合策略的可被剝削量（中位數） |
|---|---|---|---|
| 深度 1 | 0.44 | −2.05% 底池 | 8.4% |
| **深度 2** | **0.10** | **−0.07%** | **2.7%** |
| 完整解 | — | — | 0.14% |

**結論**：深度 1 偏差太大，**Agent 的翻牌應改用深度 2**。

## 下一步（依序）

1. **Agent 翻牌改用深度 2**：`realtime.py` 的 `TREES["flop"]` 加上轉牌尺寸（例如 `{"flop": [0.33, 0.75], "turn": [0.75]}`、`{"flop": [1.0], "turn": [1.0]}`），`_solve` 在翻牌傳 `checkdown=2`；更新決策原因文字（「深度限制：河牌直接發完」）。確認 10 秒預算內的迭代次數與可被剝削量、更新 `tests/test_realtime.py`。
2. **把偏差評估做成正式工具與測試**（類似 `backend/policy/exploit.py`），結果寫進 README。
3. **網路熱啟動實驗**（設計文件 B 階段）：用網路策略初始化 solver，看轉牌收斂是否變快；沒效果就不採用並記錄。
4. 更新 README 的「即時求解 Agent」段落與 `docs/realtime-agent.md` 的狀態。
5. 之後：C 階段（safe re-solving、接 M8 對手模型）或 M8。

## 已決定的事（使用者確認過）

- 即時求解每個決策時間預算 **10 秒**。
- 深籌碼翻前範圍先用規則型推估並標示。
- 第一版只做**單挑**。
- M7 的驗收門檻在訓練前訂好，不事後調整；「是否接近 GTO」以 exploitability 為準，不只看 KL／EV regret。

## 環境注意事項

- **Python**：`/home/ame/poker/.venv`（Python 3.13），所有指令用 `.venv/bin/python`。
- **GPU**：RTX 4060 Laptop 8 GB，PyTorch 2.11 + CUDA 12.8。
- **Node.js**：22.23.3，裝在 `~/.local/opt`（`~/.local/bin` 有連結）。前端改動後 `cd frontend && npm run build`。
- **資料（不在 git 裡，可重建）**：`out/datasets/gto-postflop-hu-v1`、`-v2`；`out/policy/run-v1v2/model.pt`（網路模型，即時求解的備援會用到）。
- **git push**：從 Claude 的工作階段推送會卡在 Windows 端 Git Credential Manager 的登入；請使用者在自己的終端機執行 `cd ~/poker && git push`。repo 已設定 `http.postBuffer`。
- **背景工作的陷阱**：用 `pgrep -f`／`pkill -f` 或 `ps | grep 字串 | xargs kill` 停掉工作時，若同一個指令也包含那個字串，會把自己一起殺掉（exit 144）。停工作與啟動工作要分成不同的指令，或用 PID。
- **長時間工作**（資料集、求解網格、評估）請用背景執行，並把輸出寫進檔案。
