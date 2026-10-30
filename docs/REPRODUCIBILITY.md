# 复现与发布说明

## 已保存结果核验

在仓库根目录运行 `python -B scripts/verify_results.py`，按 `README.md` 先安装依赖。脚本离线完成：原始响应 SHA-256、样本归属、训练期标准化、全部五次最佳模型权重恢复、验证和测试逐日预测、持续性基线与线性系数、指标以及五次均值和样本标准差检查。

## 从原始数据重建

依次运行 `scripts/audit_data.py` 和 `scripts/prepare_samples.py`。这两步会更新派生数据和摘要，并保留原始下载响应。然后运行 `scripts/run_stability.py`，依次重新训练种子 42、0、1、2、3；也可用 `scripts/run_experiment.py --seed 42` 单独训练。

原实验全部目录保留，重新训练使用新时间戳目录。当前 `verify_results.py` 核验的是随仓库发布的五次历史运行；如先重建样本，应先完成该核验，因为派生配置文件中的本机路径会改变文件哈希。

## 发布版的必要调整

项目原始位置为 `D:\PYJH`，本仓库由独立发布目录整理而成。`audit_data.py` 和 `prepare_samples.py` 现在支持相对 manifest 路径；`logs/download_status.json` 使用仓库相对路径。`run_stability.py` 默认新建种子 42 后再训练四个种子，也支持 `--baseline` 指定相同数据准备配置的已有种子 42 目录。

模型结构、训练超参数、样本划分和指标定义保持原实验设置。每次历史运行目录保留其当时的两个脚本副本和哈希；执行入口统一为仓库根目录的 `scripts/`。原运行元数据中的 `D:\PYJH` 为记录路径，不要求读者创建该目录。

`.gitattributes` 保留文件原字节，避免 Git 换行转换破坏记录的 SHA-256。`docs/FILE_MANIFEST.json` 保存首次发布各文件的大小与 SHA-256（该清单自身除外）。新增训练结果不会自动加入该清单。

## 环境与复现边界

原实验环境为 Windows x64、Python 3.9.13、PyTorch 2.8.0+cpu、NumPy 2.0.2、pandas 2.3.3、scikit-learn 1.6.1。完整版本见 `requirements-lock.txt`。依赖锁记录原实验环境版本；跨平台或更换软件版本可能出现数值差异。

论文中 CNN 指标为五次独立运行的指标均值与样本标准差，不是对预测取平均后的集成模型指标。种子 42 始终作为示例曲线，没有依据测试结果重新选择示例。
