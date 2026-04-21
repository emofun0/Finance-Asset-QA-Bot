---
name: check-log
description: Inspect backend trace logs for a specific session id in this project. Use when Codex needs to find a session under `backend/logs/traces` or `logs/traces`, list all JSON trace records in that session folder, and review the full conversation/request-response history or a compact summary.
---

# Check Log

## Overview

按 `session id` 查找本项目的 trace 日志目录，并查看该目录下全部 JSON 记录。
优先直接运行脚本，不要手写一长串 `find`、`ls`、`cat` 命令。

## Workflow

1. 先运行脚本查看该 session 是否存在：

```bash
python skills/check-log/scripts/check_session_logs.py <session-id>
```

2. 如果只想快速扫一遍每个请求的摘要，使用：

```bash
python skills/check-log/scripts/check_session_logs.py <session-id> --summary-only
```

3. 默认脚本会依次尝试这两个 trace 根目录：

- `backend/logs/traces`
- `logs/traces`

4. 如果项目以后改了 trace 路径，显式传入：

```bash
python skills/check-log/scripts/check_session_logs.py <session-id> --trace-root custom/logs/traces
```

## Output

脚本会输出：
- 匹配到的 `session_dir`
- 该目录下 JSON 文件数量
- 每个 JSON 的摘要：`request_id`、`route_path`、`status`、`started_at`、`finished_at`、用户消息、事件数、最终回答摘要
- 默认还会打印每个 JSON 的完整内容

## Notes

### scripts/
- `scripts/check_session_logs.py`
  - 输入 `session id`
  - 定位对应 trace 目录
  - 按时间顺序输出全部 JSON 记录
  - 支持 `--summary-only`
