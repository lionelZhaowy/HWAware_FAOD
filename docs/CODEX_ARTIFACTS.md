# Codex 中间产物管理

2026-09-14 起，工程文档与本机生成的检查产物分开存放。

| 位置 | 内容 | Git 同步 |
| --- | --- | --- |
| `docs/*.md` | 工程教程、接手记录、运行结论和操作说明 | 是 |
| `codex_artifacts/environment/` | 安装报告、环境清单、兼容性测试日志等快照 | 否 |
| `codex_artifacts/reference/` | 模型形状、权重校验、数据完整性、训练审计与数值记录等 JSON | 否 |
| `requirements/` | 实际安装使用的依赖声明、锁定文件、版本约束 | 是 |
| `logs/`、`wandb/`、`checkpoints/` | 原有本机运行输出、日志和模型二进制 | 否 |

原 `docs/environment/`、`docs/reference/` 已移动到 `codex_artifacts/` 下，文件内容保留。文档中的相对链接已更新；这些链接用于本机查看，其他机器新克隆仓库后没有对应产物属于正常情况，不应假定必须下载或重新生成全部历史报告才能运行项目。

一个例外是原 `docs/environment/pytorch-before.txt`：它是 README 安装命令使用的完整基线版本约束。现迁为 `requirements/pytorch-baseline-constraints.txt`，继续随 Git 同步，安装命令同步修改，约束内容未改变。

后续工作约定：

- 在 Markdown 中记录完成状态、关键结论、配置和验证范围，保证接手不依赖被忽略的文件。
- 新增 JSON、CSV、逐层 trace、机器环境快照等中间产物时，使用 `codex_artifacts/<任务名>/`，不再放入 docs 子目录。
- 手写的项目配置和测试 fixture 仍属于源码，不因扩展名为 JSON 而全局忽略。
- 若一个生成文件成为安装或运行的必要输入，应显式放入 requirements/config/tests 等对应位置，而非令代码依赖本机中间产物。

根 `.gitignore` 已加入 `/codex_artifacts/`。旧目录中原本被 Git 跟踪的文件，移动后会显示旧路径删除；新路径被忽略。这些删除是迁移的预期结果，磁盘上的文件仍保留。本次未执行 commit/push，提交该整理时需要一并提交旧路径删除及文档/约束文件变更。
