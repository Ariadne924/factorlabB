# Bronze → Silver → Gold 三层数据架构

**上下文**：crypto-factor-lab 项目采用 medallion 架构（bronze → silver → gold）管理数据质量。原骨架只有单一 `data/` 目录，无法表达数据加工的层级关系。

**决策**：

| 层级 | 目录 | 职责 | 数据形态 |
|------|------|------|----------|
| Bronze | `data/bronze/` | 交易所 API 原始数据，不做任何清洗 | 毫秒时间戳（int），原始列名 |
| Silver | `data/silver/` | 标准化清洗：时间戳 → datetime64[ms,UTC]，列名统一 snake_case | 标准化 DataFrame |
| Gold | `data/gold/` | 因子计算结果，评估和可视化的直接数据源 | 因子值 Series（索引为时间戳） |

**理由**：
- Bronze 层保留原始数据，出问题时可以回溯——不需要重新调 API
- Silver 层是"单一真相源"——因子计算、校验、可视化全部从这里读
- Gold 层按因子名索引，增量计算只需追加新因子文件，不需要重算无关因子
- 三层物理隔离保证了数据质量问题不会跨层污染

**废弃的方案**：单一 `data/` 目录 + Pydantic Schema 校验——校验和存储混在一起，原始数据和清洗数据没有物理隔离，无法做增量回溯。
