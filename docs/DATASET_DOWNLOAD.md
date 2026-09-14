# FAOD 数据集服务器下载

2026-09-14 更新：三个数据包均已完成下载和 CRC 校验，下载任务于 04:56 结束。当前存储迁移与解压进度见 [接手文档](PROJECT_HANDOFF.md)；下文保留原下载操作记录。

2026-09-13 已按用户要求启动后台下载，目标目录：
`/data/lab_dataset/RGB_DVS_DET`。

## 访问方式与实测

README 的三个链接属于 NTU 的 SharePoint / OneDrive for Business。
截图中的 AADSTS50020 表示当前个人 Microsoft 账号不属于资源所在租户；
参考 [Microsoft 的错误说明](https://learn.microsoft.com/en-us/troubleshoot/entra/entra-id/app-integration/error-code-aadsts50020-user-account-identity-provider-does-not-exist)。

但这三个具体共享链接允许匿名下载：保留已有查询参数并添加 `download=1`，
同时在重定向之间保留共享页面发放的匿名 Cookie，即可获得文件。
服务器对三个文件的首部及非零偏移分别进行了 512 字节读取，均得到 HTTP 206。
不保存 Cookie 的 curl 跟随重定向测试返回 403；启用匿名 Cookie 后均通过。
因此当前无需登录账号、安装 OneDrive 客户端或配置 rclone。

| 数据集 | 本地文件名 | 预期字节数 | 大小（十进制 GB） |
| --- | --- | ---: | ---: |
| PKU-DAVIS-SOD | FAOD_PKU_DAVIS_SOD.zip | 54486709692 | 54.49 |
| DSEC-Detection | FAOD_DSEC_Detection.zip | 86801382924 | 86.80 |
| EOD200 | FAOD_EOD200.zip | 47797567175 | 47.80 |

合计 189085659791 字节，即约 189.09 GB / 176.10 GiB。
前两个远端文件都叫 `freq_1_1.zip`，因此使用不同的本地名字。
EOD200 链接指向的远端文件名是 `PKU-HIGH-FREQ.zip`。

探测时项目所在根分区仅余约 57 GiB；目标所在 `/data` 分区余约 1.1 TiB。
当前只下载和校验压缩包，解压空间需另行规划；不改动已有数据集目录。

## 下载与状态

下载程序：[download_faod_datasets.py](../scripts/download_faod_datasets.py)。
使用已有 pytorch 环境中的 requests，无新增依赖。三个数据集依次处理，
后台进程已脱离终端，关闭 SSH 或 VS Code 不会因终端挂断停止该任务。

- 下载期间使用 `.zip.part`，确认长度及所有 ZIP 文件 CRC 后改为 `.zip`。
- 中断后依据文件长度发送 Range 请求；每次重试重新取得匿名共享 Cookie。
- 校验响应状态、Content-Range、文件类型和总大小，避免将登录 HTML 当作数据。
- 不记录匿名 Cookie 或临时签名下载地址。控制目录为 `.faod_download`。
- 同一目录使用进程锁，避免重复启动后同时写同一文件。

在服务器上查看：

```bash
tail -f /data/lab_dataset/RGB_DVS_DET/.faod_download/download.log
cat /data/lab_dataset/RGB_DVS_DET/.faod_download/status.json
```

`queued` 表示排队，`downloading` 表示传输，`verifying_crc` 表示完整性校验，
各数据集的 `complete` 表示压缩包已经通过校验。日志每约 30 秒报告速度与进度。
若进程异常退出，请同时查看日志末尾和进程是否仍存在，不能只依赖最后写入的状态。

需要恢复任务时，在仓库目录的持久会话中运行：

```bash
tmux new -s faod-download
/opt/miniconda3/envs/pytorch/bin/python -u -B scripts/download_faod_datasets.py \
  --destination /data/lab_dataset/RGB_DVS_DET
```

正在运行时不要重复启动；恢复任务会重新检查已完成文件并续传 `.part`。
网络重试耗尽或 CRC 不通过时保留文件，错误写入日志，不自动删除数据。
完整下载和校验结果以后台任务实际完成状态为准。
