# Task 3 Binance 测试网接入边界

## 安全默认值

`trading/binance_testnet.py` 只包含以下地址：

- 现货：`https://testnet.binance.vision`
- USD-M 合约：`https://testnet.binancefuture.com`

配置默认 `dry_run=True`。dry-run 只生成将要发送的参数，不访问网络；切换为测试网请求前
必须同时提供以下环境变量：

```bash
cp .env.example .env
# 仅填写 Binance 测试网密钥，禁止填写生产密钥
BINANCE_TESTNET_API_KEY=...
BINANCE_TESTNET_API_SECRET=...
```

`.env` 已被 Git 忽略，配置对象的字符串表示也不会显示密钥。

只检查本地配置与账本（不联网、不下单）：

```bash
python scripts/check_binance_testnet.py
```

访问公开时间接口或读取测试网账户摘要：

```bash
python scripts/check_binance_testnet.py --connect
python scripts/check_binance_testnet.py --account
```

这三个命令都不会提交订单。

## 可靠性基础

- 请求签名使用 HMAC-SHA256。
- 先同步 Binance 服务器时间，再使用 UTC 毫秒时间戳和 `recvWindow`。
- 网络错误、429 和 5xx 有限退避重试。
- `client_order_id` 由策略、资产、信号时间和方向确定性生成，重启后相同信号保持相同 ID。
- SQLite WAL 账本记录订单、成交、持仓快照与事件；订单和成交主键保证幂等。
- 网络异常发生在提交期间时，只记录 `order_submission_uncertain`，不武断标记为失败；恢复后应
  先按 `origClientOrderId` 查询和对账。

## 当前未完成

尚未使用用户测试网密钥进行真实连接、下单或持续前向验证，也没有声称测试网成绩。下一步
需要在用户明确提供测试网权限后，完成账户连通、最小订单、撤单、断线恢复和持续成绩单。
测试网流动性与撮合不同于生产环境，任何测试网结果都不得直接当作真实可交易性结论。
