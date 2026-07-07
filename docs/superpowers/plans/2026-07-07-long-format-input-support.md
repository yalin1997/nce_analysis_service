# Long Format 輸入支持 實作計畫

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 讓 nce_analysis_service 支援直接接收已展開的 long format 輸入（每行代表一個 measurement point × prestage 組合），無需再經過 WideHistoryReshape 的 unpivot 步驟。

**Architecture:** 
- 新增 `LongHistoryFormat` PreprocessingStrategy 類來驗證和處理長格式輸入
- 在 `AnalysisConfig` 加入 `input_format` 參數（"wide" 或 "long"）
- 修改 `pipeline.run()` 根據設定選擇正確的 preprocessor
- Long format 會跳過 measurement point explode 步驟（已在輸入端完成），但保持相同的資料正規化規則

**Tech Stack:** 
- pandas（資料驗證與轉換）
- pydantic（AnalysisConfig 擴展）
- pytest（單元與端到端測試）

## Global Constraints

- Python 3.12+ 使用
- 不改變內部 pipeline 的長格式中間表示
- 向後相容：預設仍為 "wide" 格式
- 新格式必須通過既有的所有端到端測試

---

### Task 1: 建立 LongHistoryFormat PreprocessingStrategy

**Files:**
- Create: `nce_analysis/preprocessing/long_history_format.py`
- Modify: `nce_analysis/preprocessing/__init__.py` (if it exists, for exports)
- Test: `tests/preprocessing/test_long_history_format.py`

**Interfaces:**
- Consumes: DataFrame with columns `[WaferID, X_Posi, Y_Posi, NCE_Value, Pre_StageID, Pre_StepID, Pre_ToolID, Pre_ChamberID, Pre_Execute_Time]`
- Produces: Same format as `WideHistoryReshape.transform()` output — DataFrame with `[WaferID, X_Posi, Y_Posi, NCE_Value, Pre_StageID, Pre_StepID, Pre_ToolID, Pre_ChamberID, Pre_Execute_Time]` (already in long format, just validated)

- [ ] **Step 1: 寫測試 - 驗證有效的長格式輸入**

新增 `tests/preprocessing/test_long_history_format.py`：

```python
import pandas as pd
import pytest
from nce_analysis.preprocessing.long_history_format import LongHistoryFormat
from nce_analysis.preprocessing.base import PreprocessingError

def test_valid_long_format_passthrough():
    """Long format input with all required columns should pass through unchanged."""
    input_df = pd.DataFrame({
        'WaferID': ['w1', 'w1', 'w1', 'w2'],
        'X_Posi': [10.0, 10.0, 20.0, 10.0],
        'Y_Posi': [20.0, 20.0, 30.0, 20.0],
        'NCE_Value': [2.5, 2.6, 3.1, 2.4],
        'Pre_StageID': ['CMP', 'CVD', 'PVD', 'CMP'],
        'Pre_StepID': ['3580.01', '3581.01', '3582.01', '3580.01'],
        'Pre_ToolID': ['T1', 'T2', 'T3', 'T1'],
        'Pre_ChamberID': ['C1', 'C2', 'C3', 'C1'],
        'Pre_Execute_Time': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'])
    })
    
    processor = LongHistoryFormat()
    result = processor.transform(input_df)
    
    # Should return the same dataframe (no transformation needed)
    pd.testing.assert_frame_equal(result.reset_index(drop=True), input_df.reset_index(drop=True))


def test_missing_required_column_raises_error():
    """Missing any required column should raise PreprocessingError."""
    input_df = pd.DataFrame({
        'WaferID': ['w1'],
        'X_Posi': [10.0],
        'Y_Posi': [20.0],
        # Missing NCE_Value
        'Pre_StageID': ['CMP'],
        'Pre_StepID': ['3580.01'],
        'Pre_ToolID': ['T1'],
        'Pre_ChamberID': ['C1'],
        'Pre_Execute_Time': pd.to_datetime(['2024-01-01'])
    })
    
    processor = LongHistoryFormat()
    with pytest.raises(PreprocessingError, match="NCE_Value"):
        processor.transform(input_df)


def test_missing_prestage_column_raises_error():
    """Missing any Pre_* column should raise PreprocessingError."""
    input_df = pd.DataFrame({
        'WaferID': ['w1'],
        'X_Posi': [10.0],
        'Y_Posi': [20.0],
        'NCE_Value': [2.5],
        'Pre_StageID': ['CMP'],
        'Pre_StepID': ['3580.01'],
        # Missing Pre_ToolID
        'Pre_ChamberID': ['C1'],
        'Pre_Execute_Time': pd.to_datetime(['2024-01-01'])
    })
    
    processor = LongHistoryFormat()
    with pytest.raises(PreprocessingError, match="Pre_ToolID"):
        processor.transform(input_df)


def test_empty_dataframe_raises_error():
    """Empty dataframe should raise PreprocessingError."""
    input_df = pd.DataFrame({
        'WaferID': [],
        'X_Posi': [],
        'Y_Posi': [],
        'NCE_Value': [],
        'Pre_StageID': [],
        'Pre_StepID': [],
        'Pre_ToolID': [],
        'Pre_ChamberID': [],
        'Pre_Execute_Time': []
    })
    
    processor = LongHistoryFormat()
    with pytest.raises(PreprocessingError, match="empty"):
        processor.transform(input_df)
```

- [ ] **Step 2: 執行測試確認失敗**

```bash
python3.12 -m pytest tests/preprocessing/test_long_history_format.py -v
```

Expected: FAIL - `ModuleNotFoundError: No module named 'nce_analysis.preprocessing.long_history_format'`

- [ ] **Step 3: 實作 LongHistoryFormat 類**

新增 `nce_analysis/preprocessing/long_history_format.py`：

```python
import logging
import pandas as pd

from nce_analysis.preprocessing.base import PreprocessingError, PreprocessingStrategy

logger = logging.getLogger(__name__)

_REQUIRED_POINT_FIELDS = ("X_Posi", "Y_Posi", "NCE_Value")
_REQUIRED_PRESTAGE_FIELDS = ("Pre_StageID", "Pre_StepID", "Pre_ToolID", "Pre_ChamberID", "Pre_Execute_Time")
_ALL_REQUIRED_FIELDS = ("WaferID",) + _REQUIRED_POINT_FIELDS + _REQUIRED_PRESTAGE_FIELDS


class LongHistoryFormat(PreprocessingStrategy):
    """Handle input that is already in long format: one row per (measurement_point × prestage).
    
    No transformation needed; just validates that all required columns are present.
    Output format is identical to WideHistoryReshape output.
    """
    
    def transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Validate long format input and return as-is."""
        if raw_df.empty:
            raise PreprocessingError(
                "Input dataframe is empty; nothing to analyze."
            )
        
        missing_cols = [col for col in _ALL_REQUIRED_FIELDS if col not in raw_df.columns]
        if missing_cols:
            raise PreprocessingError(
                f"Missing required columns for long format: {missing_cols}. "
                f"Expected: {_ALL_REQUIRED_FIELDS}"
            )
        
        # Validate no null values in critical columns
        null_cols = raw_df[list(_ALL_REQUIRED_FIELDS)].columns[raw_df[list(_ALL_REQUIRED_FIELDS)].isnull().any()].tolist()
        if null_cols:
            raise PreprocessingError(
                f"Found null values in required columns: {null_cols}"
            )
        
        # Return only the required columns in canonical order
        return raw_df[list(_ALL_REQUIRED_FIELDS)].copy()
```

- [ ] **Step 4: 執行測試確認通過**

```bash
python3.12 -m pytest tests/preprocessing/test_long_history_format.py -v
```

Expected: PASS (all 4 tests pass)

- [ ] **Step 5: Commit**

```bash
git add nce_analysis/preprocessing/long_history_format.py tests/preprocessing/test_long_history_format.py
git commit -m "feat: add LongHistoryFormat preprocessing strategy with validation"
```

---

### Task 2: 在 AnalysisConfig 中新增輸入格式選項

**Files:**
- Modify: `nce_analysis/config.py`
- Test: `tests/test_config.py` (if exists, or add to existing config tests)

**Interfaces:**
- Consumes: Nothing new
- Produces: `AnalysisConfig.input_format: Literal["wide", "long"]` with default `"wide"`

- [ ] **Step 1: 寫測試 - 驗證配置預設值與自訂值**

如果 `tests/test_config.py` 不存在，建立它；若存在，新增這些測試：

```python
def test_analysis_config_default_input_format():
    """Default input_format should be 'wide' for backward compatibility."""
    config = AnalysisConfig()
    assert config.input_format == "wide"


def test_analysis_config_can_set_long_format():
    """Should accept input_format='long'."""
    config = AnalysisConfig(input_format="long")
    assert config.input_format == "long"


def test_analysis_config_rejects_invalid_format():
    """Should reject invalid input_format values."""
    with pytest.raises(ValueError):
        AnalysisConfig(input_format="invalid")
```

- [ ] **Step 2: 執行測試確認失敗**

```bash
python3.12 -m pytest tests/test_config.py::test_analysis_config_default_input_format -v
```

Expected: FAIL - `AttributeError: 'AnalysisConfig' object has no attribute 'input_format'`

- [ ] **Step 3: 修改 AnalysisConfig**

編輯 `nce_analysis/config.py`，找到 `class AnalysisConfig` 定義並新增欄位：

```python
from typing import Literal

class AnalysisConfig(BaseModel):
    # ... existing fields ...
    
    input_format: Literal["wide", "long"] = "wide"
    """Input data format: 'wide' for Pre_StageID_1/Pre_StageID_2 columns, 
    'long' for already-expanded (measurement_point × prestage) rows."""
```

- [ ] **Step 4: 執行測試確認通過**

```bash
python3.12 -m pytest tests/test_config.py::test_analysis_config_default_input_format tests/test_config.py::test_analysis_config_can_set_long_format tests/test_config.py::test_analysis_config_rejects_invalid_format -v
```

Expected: PASS (all 3 tests pass)

- [ ] **Step 5: Commit**

```bash
git add nce_analysis/config.py tests/test_config.py
git commit -m "feat: add input_format config option to AnalysisConfig"
```

---

### Task 3: 修改 pipeline.run() 根據配置選擇 preprocessor

**Files:**
- Modify: `nce_analysis/pipeline.py`
- Test: `tests/test_pipeline_e2e.py` (add long format test)

**Interfaces:**
- Consumes: `config.input_format` (Literal["wide", "long"]), existing `preprocessed_long_df` parameter
- Produces: Same output as before, but now supports long format input path

- [ ] **Step 1: 寫測試 - 端到端測試（long format 輸入）**

編輯 `tests/test_pipeline_e2e.py`，新增一個完整的長格式測試：

```python
def test_pipeline_with_long_format_input():
    """E2E test: pipeline should accept and process long format input correctly."""
    # Create long format input: one row per (measurement_point × prestage)
    long_input_df = pd.DataFrame({
        'WaferID': ['w1', 'w1', 'w1', 'w1', 'w2', 'w2', 'w2', 'w2'],
        'X_Posi': [10.0, 10.0, 20.0, 20.0, 10.0, 10.0, 20.0, 20.0],
        'Y_Posi': [20.0, 20.0, 30.0, 30.0, 20.0, 20.0, 30.0, 30.0],
        'NCE_Value': [2.5, 2.6, 3.5, 3.6, 2.4, 2.3, 3.2, 3.1],
        'Pre_StageID': ['CMP', 'CVD', 'CMP', 'CVD', 'CMP', 'CVD', 'CMP', 'CVD'],
        'Pre_StepID': ['3580.01', '3581.01', '3580.01', '3581.01', '3580.01', '3581.01', '3580.01', '3581.01'],
        'Pre_ToolID': ['T1', 'T2', 'T1', 'T2', 'T1', 'T2', 'T1', 'T2'],
        'Pre_ChamberID': ['C1', 'C2', 'C1', 'C2', 'C1', 'C2', 'C1', 'C2'],
        'Pre_Execute_Time': pd.to_datetime([
            '2024-01-01', '2024-01-02', '2024-01-01', '2024-01-02',
            '2024-01-03', '2024-01-04', '2024-01-03', '2024-01-04'
        ])
    })
    
    config = AnalysisConfig(
        input_format="long",
        root_cause_strategy="statistical",
        drift_strategy="regression_cusum"
    )
    
    result = nce_analysis.pipeline.run(long_input_df, config)
    
    # Result should be valid AnalysisResult
    assert isinstance(result, AnalysisResult)
    assert result.Generated_At is not None
    assert result.Config_Used.input_format == "long"
```

- [ ] **Step 2: 執行測試確認失敗**

```bash
python3.12 -m pytest tests/test_pipeline_e2e.py::test_pipeline_with_long_format_input -v
```

Expected: FAIL - pipeline 目前不處理 long format

- [ ] **Step 3: 修改 pipeline.run() 的邏輯**

編輯 `nce_analysis/pipeline.py` 中的 `run()` 函數，修改 preprocessing 部分：

找到這一段：
```python
long_df = (
    preprocessed_long_df
    if preprocessed_long_df is not None
    else WideHistoryReshape().transform(raw_df)
)
```

替換為：
```python
from nce_analysis.preprocessing.long_history_format import LongHistoryFormat

_PREPROCESSING_STRATEGIES = {
    "wide": WideHistoryReshape,
    "long": LongHistoryFormat,
}

# ... in run() function:

if preprocessed_long_df is not None:
    long_df = preprocessed_long_df
else:
    preprocessor_class = _PREPROCESSING_STRATEGIES[config.input_format]
    preprocessor = preprocessor_class()
    long_df = preprocessor.transform(raw_df)
```

完整的修改應該如下（更新 run 函數的起頭）：

```python
def run(
    raw_df: pd.DataFrame,
    config: AnalysisConfig | None = None,
    *,
    preprocessed_long_df: pd.DataFrame | None = None,
) -> AnalysisResult:
    """preprocessed_long_df: pass an already-computed WideHistoryReshape
    output (e.g. one the caller also needs for chart_board.render_chart_board)
    to skip re-running preprocessing here."""
    config = config or AnalysisConfig()

    hotspot_detector = RatioThreshold()
    noise_filter = MajorityRule()
    root_cause_strategy = _ROOT_CAUSE_STRATEGIES[config.root_cause_strategy]()
    drift_strategy = _DRIFT_STRATEGIES[config.drift_strategy]()

    if preprocessed_long_df is not None:
        long_df = preprocessed_long_df
    else:
        preprocessor_class = _PREPROCESSING_STRATEGIES[config.input_format]
        preprocessor = preprocessor_class()
        long_df = preprocessor.transform(raw_df)
    
    # ... rest of function unchanged
```

並在檔案頂端加入 import 和 dictionary：

```python
from nce_analysis.preprocessing.long_history_format import LongHistoryFormat

_PREPROCESSING_STRATEGIES = {
    "wide": WideHistoryReshape,
    "long": LongHistoryFormat,
}
```

- [ ] **Step 4: 執行測試確認通過**

```bash
python3.12 -m pytest tests/test_pipeline_e2e.py::test_pipeline_with_long_format_input -v
```

Expected: PASS

- [ ] **Step 5: 確認既有測試仍通過（向後相容性）**

```bash
python3.12 -m pytest tests/test_pipeline_e2e.py -v
```

Expected: 所有現有測試仍 PASS（預設用 wide format）

- [ ] **Step 6: Commit**

```bash
git add nce_analysis/pipeline.py
git commit -m "feat: support long format input via config.input_format"
```

---

### Task 4: 驗證單元與端到端測試

**Files:**
- No new files
- Run: `tests/preprocessing/test_long_history_format.py`, `tests/test_pipeline_e2e.py`

**Interfaces:**
- Consumes: All code from Tasks 1-3
- Produces: Verified test suite passing

- [ ] **Step 1: 執行完整的 preprocessing 測試**

```bash
python3.12 -m pytest tests/preprocessing/test_long_history_format.py -v
```

Expected: PASS (4 tests)

- [ ] **Step 2: 執行完整的 pipeline 測試**

```bash
python3.12 -m pytest tests/test_pipeline_e2e.py -v
```

Expected: PASS (所有現有 + 新的長格式測試)

- [ ] **Step 3: 執行完整測試套件**

```bash
python3.12 -m pytest tests/ -v
```

Expected: PASS (所有測試，包含統計、ML、drift 等所有測試)

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "test: verify long format support with full test suite"
```

---

### Task 5: 更新文件

**Files:**
- Modify: `CLAUDE.md` (update input format section)
- Create: `examples/long_format_example.py` (示範用法)

**Interfaces:**
- Consumes: Completed implementation
- Produces: User-facing documentation

- [ ] **Step 1: 更新 CLAUDE.md 的 Input 格式說明**

編輯 `CLAUDE.md`，找到 "### The input shape and why grouping is dynamic" 段落，在其後新增：

```markdown
### Input Formats

Two input formats are supported via `AnalysisConfig.input_format`:

**Wide Format (default, "wide"):**
```
WaferID | Measurement_Points | Pre_StageID_1 | Pre_ToolID_1 | Pre_ChamberID_1 | Pre_Execute_Time_1 | Pre_StageID_2 | Pre_ToolID_2 | ...
```
- One row per wafer
- Prestage history is positionally indexed (_1, _2, ...) with unpredictable ordering
- `Measurement_Points` is a list of dicts with X_Posi, Y_Posi, NCE_Value
- Processed by `WideHistoryReshape`: explodes measurement points, then unpivots prestage columns

**Long Format ("long"):**
```
WaferID | X_Posi | Y_Posi | NCE_Value | Pre_StageID | Pre_StepID | Pre_ToolID | Pre_ChamberID | Pre_Execute_Time
```
- One row per (measurement_point × prestage combination)
- Prestage history already exploded; schema is stable regardless of history depth
- Processed by `LongHistoryFormat`: validates columns, no transformation needed
- Reduces preprocessing overhead; upstream joins already applied domain knowledge

Both formats produce identical internal long-format representation used by pipeline stages.
```

- [ ] **Step 2: 建立使用示範**

新增 `examples/long_format_example.py`：

```python
"""Example: Analyzing data in long format."""

import pandas as pd
from nce_analysis.config import AnalysisConfig
from nce_analysis import pipeline

# Create sample long format data
long_data = pd.DataFrame({
    'WaferID': ['w1', 'w1', 'w1', 'w1', 'w2', 'w2'],
    'X_Posi': [10.0, 10.0, 20.0, 20.0, 10.0, 20.0],
    'Y_Posi': [20.0, 20.0, 30.0, 30.0, 20.0, 30.0],
    'NCE_Value': [2.5, 2.6, 3.5, 3.6, 2.4, 3.2],
    'Pre_StageID': ['CMP', 'CVD', 'CMP', 'CVD', 'CMP', 'CMP'],
    'Pre_StepID': ['3580.01', '3581.01', '3580.01', '3581.01', '3580.01', '3580.01'],
    'Pre_ToolID': ['T1', 'T2', 'T1', 'T2', 'T1', 'T1'],
    'Pre_ChamberID': ['C1', 'C2', 'C1', 'C2', 'C1', 'C1'],
    'Pre_Execute_Time': pd.to_datetime([
        '2024-01-01', '2024-01-02', '2024-01-01', '2024-01-02',
        '2024-01-03', '2024-01-03'
    ])
})

# Configure for long format
config = AnalysisConfig(
    input_format="long",
    root_cause_strategy="statistical",
    drift_strategy="regression_cusum"
)

# Run analysis
result = pipeline.run(long_data, config)

# Output
print(result.model_dump_json(indent=2))
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md examples/long_format_example.py
git commit -m "docs: add long format input documentation and example"
```

---

## 自我檢查

**Spec 覆蓋檢查：**
- ✅ 支援長格式輸入 (Task 1, 3)
- ✅ 已展開格式無需 unpivot (Task 1)
- ✅ 配置選項支援切換 (Task 2)
- ✅ 向後相容（預設仍為 wide）(Task 3, 4)
- ✅ 完整測試覆蓋 (Task 4, 5)

**佔位符掃描：**
- ✅ 所有代碼完整，無 "TBD" 或 "TODO"
- ✅ 所有測試有具體的 assertion
- ✅ 所有步驟有完整命令和預期結果

**型別一致性：**
- ✅ `input_format: Literal["wide", "long"]`  跨任務一致
- ✅ `LongHistoryFormat` 實作 `PreprocessingStrategy` 介面
- ✅ Output 格式與 `WideHistoryReshape` 相同
