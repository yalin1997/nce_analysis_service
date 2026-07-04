# nce-analysis

[English](README.md) | 繁體中文

一個純 Python 函式庫，用來分析 LITHO 站點的 NCE（Non-Correctable Error，
不可修正誤差）表面平坦度量測資料，偵測晶圓表面上的異常熱點（hotspot），
並將每個熱點回溯歸因到最可能造成問題的上游機台/腔體（CMP、CVD、PVD……）——
同時排除實際上是由 LITHO 機台自身的 Tool/Chuck 造成、而非上游製程造成的異常。

這是一個沒有 API/服務層的函式庫：你只需呼叫一個函式、傳入一個 pandas
`DataFrame`，就能拿到一個型別化的結果物件。

## 安裝

```bash
python3.12 -m pip install --user --break-system-packages -e ".[dev]"
```

（本 repo 沒有內建虛擬環境；請務必使用 `python3.12`——單純的 `python3`
在你的機器上可能會指向別的直譯器。）

## 快速開始

`examples/` 目錄提供了一個可直接執行、自成一體的範例，讓你在串接自己的資料
之前，先看看整條 pipeline 實際跑起來是什麼樣子：

```bash
python3.12 examples/run_example.py
```

這個範例會建立一批 33 片晶圓的合成資料（`examples/sample_data.py`），送進
`nce_analysis.pipeline.run` 執行，並印出：

- 一個被確認的根本原因（`CMP_01`/`ChamberA`，分類為 `CHAMBER_DRIFT`）——
  對應到一個 NCE 隨時間持續惡化的熱點，
- 一個正確歸因給 LITHO 機台自身 chuck 的熱點（`LITHO_CHUCK_ISSUE`），
  而不是誤判成任何上游機台的問題，
- 第三個熱點因為晶圓數量太少、無法有足夠信心分析，被回報在
  `Insufficient_Sample_Hotspots` 中。

建議對照著輸出結果閱讀 `examples/sample_data.py`——它的 module docstring
清楚說明了哪些資料列應該產生哪種結果，同時也是下方「輸入資料格式」的
實例示範。

### CLI 快速開始

同一批範例資料也提供了 CSV/Parquet 版本（已 commit 進 repo），可以直接
透過命令列介面執行，不需要寫任何 Python：

```bash
python3.12 -m nce_analysis --input examples/sample_data.csv \
    --config examples/sample_config.yaml --output /tmp/result.json
cat /tmp/result.json
```

`examples/sample_data.parquet`用法相同（`--input
examples/sample_data.parquet`）。若修改了 `examples/sample_data.py`，可執行
`python3.12 examples/generate_sample_files.py` 重新產生這三個 fixture 檔案。

## 在你自己的資料上使用

```python
import pandas as pd
from nce_analysis import pipeline
from nce_analysis.config import AnalysisConfig

raw_df = pd.DataFrame([...])  # 格式請見下方「輸入資料格式」
config = AnalysisConfig()      # 使用預設值，或覆寫特定欄位

result = pipeline.run(raw_df, config)

print(result.model_dump_json(indent=2))
```

或是不寫 Python，直接用命令列執行：

```bash
python3.12 -m nce_analysis --input your_data.csv --config your_config.yaml --output result.json
```

- `--input`：必填。`.csv` 或 `.parquet` 檔案。CSV 支援兩種格式，依欄位自動
  偵測：**wide**（有 `Measurement_Points` 欄位，內容是每列一筆 JSON 編碼的
  清單，跟記憶體中的形狀完全一致）或 **long**（一列一個量測點，直接有
  `X_Posi`/`Y_Posi`/`NCE_Value` 欄位，並用 `WaferID` 欄位把同一片晶圓的多筆
  量測點重新分組）。Parquet 只支援 wide 格式（巢狀欄位是 Parquet 原生支援
  的，這也是選用 Parquet 的意義所在）。
- `--config`：選填。YAML 檔案，可只列出 `AnalysisConfig` 部分欄位（例如
  `summary_top_n: 3`）；沒列出的欄位沿用預設值。完全不給 `--config` 就是
  全部用預設值執行。
- `--output`：選填。輸出結果 JSON 的檔案路徑；不給的話直接印到 stdout。

### 輸入資料格式

每片晶圓一列資料。歷史紀錄欄位以位置編號展開，格式為 `Pre_<欄位>_<N>`，
其中 `_1` 代表緊鄰在被分析的 LITHO 站點之前的那一站，`_2` 是再往前一站，
以此類推。歷史層級數 `N` 是從實際存在的欄位動態偵測出來的——不需要每片
晶圓都一樣多（若某片晶圓的前置站點數少於這批資料的最大 `N`，較高層級的
欄位就是 `None`/缺值）。

| 欄位 | 型別 | 意義 |
|---|---|---|
| `PartID` | str | 產品/料號代碼 |
| `WaferID` | str | 晶圓代碼 |
| `StageID`, `StepID` | str | 目前正在量測的 LITHO 站點/製程步驟 |
| `ToolID`, `ChuckID` | str | 處理這片晶圓的 LITHO 機台與 chuck |
| `Execute_Time` | str/date | LITHO 量測的時間 |
| `Measurement_Points` | `{X_Posi, Y_Posi, NCE_Value}` 的清單 | 這片晶圓的表面平坦度量測值 |
| `Pre_StageID_i`, `Pre_StepID_i` | str | 往前第 `i` 層的上游站點/步驟 |
| `Pre_ToolID_i`, `Pre_ChamberID_i` | str | 往前第 `i` 層的上游機台/腔體 |
| `Pre_Execute_Time_i` | str/date | 往前第 `i` 層上游製程的執行時間 |

每片晶圓在表面上每個被量測的座標點都會貢獻一列資料（也就是說，真實資料中
一片晶圓的 `Measurement_Points` 通常會有很多筆，對應光罩網格上的每個
(X, Y) 位置）；`examples/sample_data.py` 為了可讀性，每片晶圓只用了一個
量測點。

### 主要設定選項（`AnalysisConfig`）

| 欄位 | 預設值 | 作用 |
|---|---|---|
| `spec_threshold` | `15.0` | `NCE_Value` 超過此值視為異常 |
| `min_wafer_count` | `5` | 某座標點的晶圓數低於此值，會被回報到 `Insufficient_Sample_Hotspots`，不進行分析 |
| `hotspot_ratio_threshold` | `0.05` | 某座標點被判定為熱點所需的最低異常晶圓比例 |
| `noise_filter_majority_threshold` | `0.5` | 某一個 LITHO chuck/機台需佔異常晶圓多少比例，才會被歸咎為 LITHO 自身問題而非上游問題 |
| `root_cause_strategy` | `"both"` | `"statistical"`（卡方/Fisher 檢定）、`"ml"`（用 SHAP 解釋決策樹）、或 `"both"`（交叉驗證；兩者意見不一致時設定 `Requires_Manual_Review`） |
| `root_cause_granularity` | `"chamber"` | `"chamber"` 以 `Pre_ToolID + Pre_ChamberID` 組合作為嫌疑對象；`"tool"` 只用 `Pre_ToolID` |
| `drift_strategy` | `"regression_cusum"` | `"regression_cusum"`（線性趨勢＋變點偵測）或 `"correlation"`（簡單皮爾森相關係數） |
| `alpha` | `0.05` | 統計檢定策略使用的顯著性門檻 |
| `summary_top_n` | `5` | `AnalysisResult.Summary` 保留的嫌疑對象數量上限（依信心分數排序） |

### 輸出（`AnalysisResult`）

- `Summary`：信心分數最高的嫌疑對象清單（依 tool/chamber/step/根本原因類型
  去重、依 `Confidence_Score` 由高到低排序、並截斷至 `summary_top_n` 筆）——
  這是要優先展示給人看的清單。
- `Details`：所有找到的嫌疑對象，每筆對應一組（座標 × 嫌疑對象 × 根本原因
  類型），是尚未經過 Summary 去重/截斷前的完整結果。
- `Insufficient_Sample_Hotspots`：晶圓數量太少、無法分析的座標點；只有原始
  計數，沒有信心分數。
- `Root_Cause_Type` 分類共有：`LITHO_TOOL_ISSUE`、`LITHO_CHUCK_ISSUE`、
  `LITHO_CHUCK_CONTAMINATION`（以上三種都來自 noise filter，代表問題出在
  LITHO 自身設備）、`SPECIFIC_CHAMBER_DEFECT`（上游嫌疑對象，但無時間趨勢）、
  `CHAMBER_DRIFT`、`CHAMBER_SUDDEN_SHIFT`（上游嫌疑對象，且偵測到趨勢/
  變化點）。

## 開發

測試指令、TDD 開發流程與內部 pipeline 架構請參考 `CLAUDE.md`。完整的設計
理念（為什麼分組邏輯是動態的、noise-filter 門檻的由來、統計檢定 fallback
規則等）記錄在
`docs/superpowers/specs/2026-07-04-cross-stage-impact-analysis-design.md`。
