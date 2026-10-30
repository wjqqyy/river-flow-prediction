# -*- coding: utf-8 -*-
"""Run the fixed persistence / OLS / 1D-CNN comparison on prepared samples."""
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from prepare_samples import FEATURES, SPLITS, REVIEW_DATES, fit_scaler


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=np.float64), np.asarray(predicted, dtype=np.float64)
    if actual.ndim != 1 or predicted.shape != actual.shape or not len(actual):
        raise ValueError('Metric arrays must be matching nonempty vectors')
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError('Nonfinite predictions or observations')
    error = predicted - actual
    denominator = np.square(actual - actual.mean()).sum()
    if denominator <= 0:
        raise ValueError('NSE undefined for constant observations')
    return {'n': len(actual), 'MAE_m3_s': float(np.abs(error).mean()),
            'RMSE_m3_s': float(np.sqrt(np.square(error).mean())),
            'NSE': float(1 - np.square(error).sum() / denominator),
            'negative_predictions': int((predicted < 0).sum())}


def self_check():
    truth = np.array([1., 2., 3.])
    result = evaluate(truth, truth)
    assert result['MAE_m3_s'] == result['RMSE_m3_s'] == 0 and result['NSE'] == 1
    result = evaluate(truth, np.array([2., 2., 2.]))
    assert np.isclose(result['MAE_m3_s'], 2 / 3)
    assert np.isclose(result['RMSE_m3_s'], np.sqrt(2 / 3)) and result['NSE'] == 0


def load_data():
    folder = ROOT / 'data' / 'model'
    config = json.loads((folder / 'sample_config.json').read_text(encoding='utf-8'))
    scaler = json.loads((folder / 'normalization.json').read_text(encoding='utf-8'))
    source = ROOT / 'data' / 'processed' / 'daily_discharge_m3s.csv'
    if not config['model_ready'] or sha256(source) != config['source_sha256']:
        raise ValueError('Prepared data status or source hash changed')
    if config['feature_columns'] != FEATURES or config['scaler'] != scaler:
        raise ValueError('Feature/scaler configuration mismatch')
    daily = pd.read_csv(source, encoding='utf-8-sig', keep_default_na=False,
                        parse_dates=['date']).set_index('date')
    if daily.index.has_duplicates or not daily.index.equals(pd.date_range('2010-01-01', '2024-12-31')):
        raise ValueError('Source calendar changed')
    qualifiers = daily['qualifier_json'].map(json.loads)
    special = qualifiers.map(lambda q: bool(set(q or []) - {'ESTIMATED', 'REVISED'}))
    if set(daily.index[special].strftime('%Y-%m-%d')) != REVIEW_DATES:
        raise ValueError('Special qualifier dates changed')
    daily['unusable_reason'] = ''
    invalid = (~np.isfinite(daily['discharge_m3_s']) | daily['discharge_m3_s'].lt(0)
               | daily['approval_status'].ne('Approved') | special)
    daily.loc[invalid, 'unusable_reason'] = 'excluded'
    if fit_scaler(daily) != scaler:
        raise ValueError('Normalization must be fitted exclusively on valid training days')
    frames = {}
    hashes = {str(source.relative_to(ROOT)): sha256(source)}
    for name, (start, end) in SPLITS.items():
        path = folder / (name + '.csv')
        frame = pd.read_csv(path, encoding='utf-8-sig',
                            parse_dates=['target_date', 'input_start_date', 'input_end_date'])
        dates = pd.DatetimeIndex(frame['target_date'])
        expected = [date for date in daily.loc[start:end].index
                    if date >= daily.index[7] and
                    not invalid.loc[date - pd.Timedelta(days=7):date].any()]
        if dates.tolist() != expected or len(frame) != config['splits'][name]['samples']:
            raise ValueError('Unexpected sample membership: ' + name)
        assert (frame['input_start_date'] == frame['target_date'] - pd.Timedelta(days=7)).all()
        assert (frame['input_end_date'] == frame['target_date'] - pd.Timedelta(days=1)).all()
        for lag, feature in zip(range(7, 0, -1), FEATURES):
            assert np.allclose(frame[feature], daily.loc[dates - pd.Timedelta(days=lag), 'discharge_m3_s'])
        assert np.allclose(frame['target_m3_s'], daily.loc[dates, 'discharge_m3_s'])
        assert np.isfinite(frame[FEATURES + ['target_m3_s']].to_numpy()).all()
        frames[name] = frame
        hashes[str(path.relative_to(ROOT))] = sha256(path)
    for name in ['sample_config.json', 'normalization.json']:
        hashes[str((folder / name).relative_to(ROOT))] = sha256(folder / name)
    return frames, daily, config, scaler, hashes


def make_model():
    return nn.Sequential(nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(),
                         nn.Conv1d(16, 16, 3, padding=1), nn.ReLU(), nn.Flatten(),
                         nn.Linear(112, 16), nn.ReLU(), nn.Linear(16, 1))


def plot_results(out, daily, history, predictions, metrics):
    colors = {'persistence': '#d28b16', 'linear_regression': '#218560', 'cnn': '#b13f51'}
    labels = {'persistence': 'Persistence', 'linear_regression': 'Linear regression', 'cnn': '1D-CNN'}
    fig, ax = plt.subplots(figsize=(12, 4), layout='constrained')
    ax.plot(daily.index, daily['discharge_m3_s'], color='#244b6b', linewidth=.65)
    for (name, (start, end)), color in zip(SPLITS.items(), ['#cde4f1', '#f5e1b1', '#d5ead5']):
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1), color=color, alpha=.45, label=name.title())
    ax.set(title='USGS 01646500 | Daily mean discharge and target-date splits', xlabel='Date', ylabel='Discharge (m³/s)', xlim=(daily.index.min(), daily.index.max()), ylim=(0, None))
    ax.legend(frameon=False)
    ax.grid(alpha=.2)
    fig.savefig(out / 'data_splits.png', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.plot(history['epoch'], history['train_mse_z'], label='Training (during optimization)')
    ax.plot(history['epoch'], history['validation_mse_z'], label='Validation')
    best = int(history.loc[history['validation_mse_z'].idxmin(), 'epoch'])
    ax.axvline(best, color='#777777', linestyle='--', label='Best validation epoch: %d' % best)
    ax.set(title='1D-CNN training | seed 42', xlabel='Epoch', ylabel='MSE (standardized discharge)')
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=.2)
    fig.savefig(out / 'cnn_loss.png', dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), layout='constrained', sharey=True)
    for ax, year in zip(axes, [2022, 2023, 2024]):
        part = predictions.loc[predictions['date'].dt.year.eq(year)]
        ax.plot(part['date'], part['observed_m3_s'], color='#222222', linewidth=1, label='Observed')
        for model in colors:
            ax.plot(part['date'], part[model + '_m3_s'], color=colors[model], linewidth=.8, alpha=.85, label=labels[model])
        ax.set(title=str(year), ylabel='Discharge (m³/s)', xlim=(part['date'].min(), part['date'].max()))
        ax.grid(alpha=.2)
    axes[0].legend(frameon=False, ncol=4, fontsize=8)
    axes[-1].set_xlabel('Target date | rolling next-day prediction')
    fig.suptitle('Test period: observed and predicted daily mean discharge')
    fig.savefig(out / 'test_predictions.png', dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained', sharex=True, sharey=True)
    columns = ['observed_m3_s'] + [name + '_m3_s' for name in colors]
    minimum = min(0, float(predictions[columns].min().min()))
    maximum = float(predictions[columns].max().max()) * 1.04
    for ax, model in zip(axes, colors):
        ax.scatter(predictions['observed_m3_s'], predictions[model + '_m3_s'], s=8, alpha=.35, color=colors[model], rasterized=True)
        ax.plot([minimum, maximum], [minimum, maximum], color='#333333', linestyle='--', linewidth=.8)
        row = metrics.loc[metrics['model'].eq(model)].iloc[0]
        ax.set(title='%s\nRMSE %.2f | NSE %.3f' % (labels[model], row['RMSE_m3_s'], row['NSE']),
               xlabel='Observed (m³/s)', xlim=(minimum, maximum), ylim=(minimum, maximum))
        ax.set_aspect('equal', adjustable='box')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Predicted (m³/s)')
    fig.savefig(out / 'test_scatter.png', dpi=300)
    plt.close(fig)


def main():
    started = time.perf_counter()
    self_check()
    frames, daily, sample_config, scaler, hashes = load_data()
    timestamp = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime('%Y%m%dT%H%M%S')
    out = ROOT / 'outputs' / (timestamp + '_seed42')
    out.mkdir(parents=True, exist_ok=False)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    mean, std = scaler['mean_m3_s'], scaler['std_m3_s']
    arrays = {name: (frame[FEATURES].to_numpy(dtype=np.float64),
                     frame['target_m3_s'].to_numpy(dtype=np.float64)) for name, frame in frames.items()}
    tensors = {name: (torch.tensor((x - mean) / std, dtype=torch.float32).unsqueeze(1),
                      torch.tensor((y - mean) / std, dtype=torch.float32).unsqueeze(1))
               for name, (x, y) in arrays.items()}
    linear = LinearRegression(fit_intercept=True).fit(*arrays['train'])
    linear_params = {'features': FEATURES, 'coefficients': linear.coef_.tolist(),
                     'intercept_m3_s': float(linear.intercept_), 'fit_split': 'train'}
    (out / 'linear_model.json').write_text(json.dumps(linear_params, indent=2), encoding='utf-8')
    loader = DataLoader(TensorDataset(*tensors['train']), batch_size=64, shuffle=True,
                        num_workers=0, generator=torch.Generator().manual_seed(42))
    model = make_model()
    assert model(torch.zeros(2, 1, 7)).shape == (2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    criterion = nn.MSELoss()
    config = {'seed': 42, 'device': 'cpu', 'torch_threads': 1, 'deterministic_algorithms': True,
              'architecture': str(model), 'parameter_count': sum(p.numel() for p in model.parameters()),
              'optimizer': 'Adam', 'learning_rate': .001, 'batch_size': 64,
              'loss': 'MSE on standardized discharge', 'max_epochs': 100,
              'early_stopping_patience': 15, 'early_stopping_min_delta': 0,
              'checkpoint_selection': 'lowest complete-validation-set MSE',
              'training_shuffle': True, 'split_shuffle': False, 'num_workers': 0,
              'postprocessing': 'none; no clipping', 'test_updates': False,
              'input_sha256': hashes, 'sample_preparation': sample_config,
              'training_script_sha256': sha256(Path(__file__))}
    (out / 'experiment_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'normalization.json').write_text(json.dumps(scaler, indent=2), encoding='utf-8')
    versions = {'python': platform.python_version(), 'platform': platform.platform(),
                'packages': {p: importlib.metadata.version(p) for p in ['torch', 'numpy', 'pandas', 'scikit-learn', 'matplotlib', 'scipy']}}
    (out / 'software_versions.json').write_text(json.dumps(versions, indent=2), encoding='utf-8')
    shutil.copyfile(Path(__file__), out / 'run_experiment.py')
    shutil.copyfile(ROOT / 'scripts' / 'prepare_samples.py', out / 'prepare_samples.py')
    print('Output: ' + str(out), flush=True)
    print('Fixed design: seed=42; train/validation/test=3617/731/1096; CNN parameters=%d' % config['parameter_count'], flush=True)
    best_loss, best_epoch, stale = float('inf'), 0, 0
    history = []
    with (out / 'training.log').open('w', encoding='utf-8') as log:
        for epoch in range(1, 101):
            model.train()
            train_sum = 0.
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                loss = criterion(model(x_batch), y_batch)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward()
                optimizer.step()
                train_sum += loss.item() * len(x_batch)
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(tensors['validation'][0]), tensors['validation'][1]).item()
            if not np.isfinite(val_loss):
                raise ValueError('Nonfinite validation loss')
            history.append({'epoch': epoch, 'train_mse_z': train_sum / len(frames['train']),
                            'validation_mse_z': val_loss})
            if val_loss < best_loss:
                best_loss, best_epoch, stale = val_loss, epoch, 0
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch,
                            'validation_mse_z': val_loss, 'seed': 42}, out / 'cnn_best.pt')
            else:
                stale += 1
            line = 'epoch=%d train_mse_z=%.8f validation_mse_z=%.8f best_epoch=%d' % (epoch, history[-1]['train_mse_z'], val_loss, best_epoch)
            log.write(line + '\n')
            log.flush()
            if epoch % 10 == 0 or stale >= 15 or epoch == 100:
                print(line, flush=True)
            if stale >= 15:
                break
    history = pd.DataFrame(history)
    history.to_csv(out / 'training_history.csv', index=False)
    checkpoint = torch.load(out / 'cnn_best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    with torch.no_grad():
        restored_validation_loss = criterion(model(tensors['validation'][0]), tensors['validation'][1]).item()
    assert restored_validation_loss == best_loss
    tables = {}
    for name in ['validation', 'test']:
        x, y = arrays[name]
        with torch.no_grad():
            cnn = model(tensors[name][0]).squeeze(1).numpy().astype(np.float64) * std + mean
        predictions = {'persistence': x[:, -1].copy(), 'linear_regression': linear.predict(x), 'cnn': cnn}
        table = pd.DataFrame([dict(model=key, **evaluate(y, value)) for key, value in predictions.items()])
        table.to_csv(out / ('metrics.csv' if name == 'test' else 'validation_metrics.csv'), index=False, encoding='utf-8-sig')
        dates = frames[name]['target_date']
        saved_predictions = pd.DataFrame({'date': dates, 'observed_m3_s': y,
                                           **{key + '_m3_s': value for key, value in predictions.items()}})
        saved_predictions.to_csv(out / (name + '_predictions.csv'), index=False, encoding='utf-8-sig', date_format='%Y-%m-%d')
        tables[name] = (table, saved_predictions)
    metrics, predictions = tables['test']
    saved = pd.read_csv(out / 'test_predictions.csv', encoding='utf-8-sig')
    for row in metrics.to_dict('records'):
        result = evaluate(saved['observed_m3_s'], saved[row['model'] + '_m3_s'])
        for key, value in result.items():
            assert np.isclose(value, row[key], rtol=1e-12, atol=1e-12)
    assert np.allclose(saved['persistence_m3_s'], arrays['test'][0][:, -1])
    assert np.allclose(saved['linear_regression_m3_s'], arrays['test'][0] @ np.array(linear_params['coefficients']) + linear_params['intercept_m3_s'])
    for path, original_hash in hashes.items():
        assert sha256(ROOT / path) == original_hash
    plot_results(out, daily, history, predictions, metrics)
    result = {'status': 'complete', 'output_directory': str(out), 'seed': 42,
              'epochs_run': len(history), 'best_epoch': best_epoch, 'best_validation_mse_z': best_loss,
              'parameter_count': config['parameter_count'], 'elapsed_seconds': time.perf_counter() - started,
              'stop_reason': 'validation patience 15' if stale >= 15 else 'max epochs 100',
              'checks': 'PASS: dates, values, exclusions, normalization, checkpoint reload, saved metrics, persistence, OLS coefficients, source hashes',
              'test_metrics': metrics.to_dict('records')}
    (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    metric_rows = '\n'.join('| %s | %.6f | %.6f | %.6f | %d |' %
                            (r['model'], r['MAE_m3_s'], r['RMSE_m3_s'], r['NSE'], r['negative_predictions'])
                            for r in result['test_metrics'])
    report = f'''# 河流日流量预测对比实验

完成时间：{dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()}。
输出目录：`{out}`。实际运行耗时 {result['elapsed_seconds']:.2f} 秒。

## 固定实验设计

USGS 01646500站2010—2024年日平均流量，过去连续7天预测下一天；逐日使用已观测历史，非多年递归外推。
训练目标2010—2019年3617个，验证2020—2021年731个，测试2022—2024年1096个。
Approved的估算与修订记录保留；任何输入或目标涉及6天待核实标记的样本排除。原始数据和样本文件哈希未改变。
标准化仅按训练期3646个有效日值拟合，均值 {mean:.9f}、标准差 {std:.9f} m³/s。
模型仅输入7个历史流量，质量标记和日期不作为特征。测试集不参与拟合、早停或模型选择，不在测试期更新参数。

## 方法与实际训练

持续性预测直接使用前一天流量；线性回归为带截距的普通最小二乘，仅拟合训练样本。
CNN：Conv1d(1,16,3,padding=1) → ReLU → Conv1d(16,16,3,padding=1) → ReLU → Flatten → Linear(112,16) → ReLU → Linear(16,1)。
共 {config['parameter_count']} 个参数，输入[batch,1,7]，输出层为线性。
随机种子42，CPU单线程、确定性算法；Adam学习率0.001，批量64，标准化目标MSE，最多100轮，验证损失连续15轮不改善则停止。
实际训练 {len(history)} 轮，最佳验证轮次 {best_epoch}，停止原因：{result['stop_reason']}。
已重新加载并核对最佳模型。训练损失按批次样本数加权；验证损失在完整验证集统一计算。
预测还原为m³/s，没有裁剪、平滑或其他后处理。

## 测试结果

三种方法使用完全相同的1096个测试日期；MAE、RMSE越低越好，NSE越高越好。
NSE按完整测试序列的平方误差和与观测离均差平方和计算，不按批次平均。

| 方法 | MAE（m³/s） | RMSE（m³/s） | NSE | 负预测数 |
|---|---:|---:|---:|---:|
{metric_rows}

## 文件与核对

metrics.csv为测试指标；validation_metrics.csv为验证指标；test_predictions.csv和validation_predictions.csv为逐日结果。
cnn_best.pt为最佳模型权重；linear_model.json保存线性回归系数；experiment_config.json、normalization.json、software_versions.json保存复现配置。
training_history.csv和training.log记录所有训练轮次；同目录保存本次运行的两个脚本副本。
data_splits.png、cnn_loss.png、test_predictions.png、test_scatter.png均为300 dpi。
已核对日期与样本、排除规则、训练期标准化、模型恢复、CSV回读指标、持续性预测和线性模型系数。

## 结果适用范围

本次为固定设置、单一随机种子的一轮对比，不支持统计显著性或跨站泛化结论。
估算与修订值来自当前下载档案，不能声称完全还原历史实时数据可用条件。
本阶段未增加CNN-LSTM、调参搜索或论文正文。
'''
    (out / 'experiment_summary.md').write_text(report, encoding='utf-8')
    print(metrics.to_string(index=False), flush=True)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
