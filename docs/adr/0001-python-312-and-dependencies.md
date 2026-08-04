# Python 3.12 与依赖锁定

**上下文**：项目启动时需确定 Python 版本和依赖基线。Python 3.12 是当前 LTS 版本，生态支持完整；3.13 部分科学计算包（numpy、pandas）的预编译 wheel 覆盖尚不稳定。

**决策**：锁定 Python 3.12，所有依赖版本在 `requirements.txt` 中明确指定。

**理由**：
- numpy 1.26.4 和 pandas 2.2.3 是 Python 3.12 下最后一批提供所有平台预编译 wheel 的稳定组合
- Pydantic v2（≥ 2.0.0）强制类型校验 + `model_config = {"extra": "forbid"}` 可防御交易所 API 返回字段变更时的静默错误
- 锁定版本避免"在我机器上能跑"——小组 3 人以上协作时版本漂移是最高频故障源

**废弃的方案**：Python 3.13+ —— 部分依赖的 wheel 覆盖不完整，编译安装失败率高，不符合 G0 门禁"10 分钟内可跑通"的要求。
