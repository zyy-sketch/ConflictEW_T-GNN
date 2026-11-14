"""ConflictEW 军费冲突早期预警系统 — 核心包。

三层架构（对齐论文）：
  1. 数据层：data_fetch / external / data / build_panel（F01-F04）
  2. 分析引擎：kalman / factors / xgb / shap_analysis / metrics（F05-F09）
  3. 推理与产品：agent / alerts / report（F15-F18）
"""

__version__ = "1.0.0"
