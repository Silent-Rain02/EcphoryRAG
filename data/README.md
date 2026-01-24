# 数据说明（Data Note）

本目录包含用于**快速复现/演示**的少量处理后数据子集（subset），用于验证本仓库代码能够完成索引、检索与评测的完整流程。

## 内容

- `processed_hotpotqa/hotpotqa_subset.jsonl`
- `processed_musique/musique_subset.jsonl`
- `processed_2wiki/2wiki_subset.jsonl`

## 用途与范围

- 这些文件用于开箱即跑（sanity-check），**不等同于**官方完整数据集的再发布版本。
- 完整实验通常需要从各数据集官方渠道下载原始数据，并使用本仓库 `ecphoryrag/data_processing/` 中的处理器生成 `*_processed.jsonl`（详见仓库根目录 `README.md` 的“数据集与许可”部分）。

## 许可与合规

使用者需自行遵守对应数据集及其上游数据源的许可条款（包括但不限于再分发/改写/派生数据的限制）。如某数据集许可不允许在公开仓库中再分发（即使是子集），请删除相关文件并改为仅提供下载与处理脚本。

