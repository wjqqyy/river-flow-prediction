# -*- coding: utf-8 -*-
"""Add the four agreed seeds and verify/summarize all five retained runs."""
import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch

from run_experiment import ROOT, make_model, sha256


def main(baseline=None):
    if baseline is None:
        process = subprocess.run([sys.executable, '-B', '-u', str(ROOT / 'scripts' / 'run_experiment.py'),
                                  '--seed', '42'], cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
        if process.returncode:
            raise RuntimeError('Seed42 failed: ' + process.stderr)
        paths = [line[len('Output: '):] for line in process.stdout.splitlines() if line.startswith('Output: ')]
        if len(paths) != 1:
            raise ValueError('Missing unique seed42 output')
        baseline = Path(paths[0])
    else:
        baseline = Path(baseline).resolve()
    original_config = json.loads((baseline / 'experiment_config.json').read_text(encoding='utf-8'))
    stamp = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime('%Y%m%dT%H%M%S')
    summary_dir = ROOT / 'outputs' / (stamp + '_stability_5seeds')
    summary_dir.mkdir(exist_ok=False)
    plan = {'seeds': [42, 0, 1, 2, 3], 'retained_original_run': str(baseline),
            'additional_seeds': [0, 1, 2, 3], 'standard_deviation_ddof': 1,
            'policy': 'All agreed seeds retained; no tuning or test-based model selection; seed42 remains the example curve.',
            'reference_config_sha256': sha256(baseline / 'experiment_config.json'),
            'training_script_sha256': sha256(ROOT / 'scripts' / 'run_experiment.py')}
    (summary_dir / 'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    runs = [baseline]
    for seed in plan['additional_seeds']:
        process = subprocess.run([sys.executable, '-B', '-u', str(ROOT / 'scripts' / 'run_experiment.py'),
                                  '--seed', str(seed)], cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
        (summary_dir / ('seed%d_console.log' % seed)).write_text(process.stdout + process.stderr, encoding='utf-8')
        if process.returncode:
            raise RuntimeError('Training seed %d failed; see %s' % (seed, summary_dir))
        output_lines = [line[len('Output: '):] for line in process.stdout.splitlines() if line.startswith('Output: ')]
        if len(output_lines) != 1:
            raise ValueError('Missing unique run output path')
        runs.append(Path(output_lines[0]))
        result = json.loads((runs[-1] / 'result.json').read_text(encoding='utf-8'))
        print('Completed seed %d: %d epochs, best epoch %d' % (seed, result['epochs_run'], result['best_epoch']), flush=True)
    torch.set_num_threads(1)
    per_run = []
    metrics_by_model = []
    original_predictions = {split: pd.read_csv(baseline / (split + '_predictions.csv')) for split in ['validation', 'test']}
    unchanged_keys = set(original_config) - {'seed', 'training_script_sha256'}
    for expected_seed, run in zip(plan['seeds'], runs):
        config = json.loads((run / 'experiment_config.json').read_text(encoding='utf-8'))
        result = json.loads((run / 'result.json').read_text(encoding='utf-8'))
        assert config['seed'] == result['seed'] == expected_seed
        assert result['status'] == 'complete'
        for key in unchanged_keys:
            assert config[key] == original_config[key], (expected_seed, key)
        assert json.loads((run / 'software_versions.json').read_text()) == json.loads((baseline / 'software_versions.json').read_text())
        for name, expected_hash in config['input_sha256'].items():
            assert sha256(ROOT / name) == expected_hash
        checkpoint = torch.load(run / 'cnn_best.pt', map_location='cpu', weights_only=True)
        assert checkpoint['seed'] == expected_seed
        model = make_model()
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        scaler = json.loads((run / 'normalization.json').read_text())
        history = pd.read_csv(run / 'training_history.csv')
        best = history.loc[history['validation_mse_z'].idxmin()]
        assert int(best['epoch']) == checkpoint['epoch'] == result['best_epoch']
        assert np.isclose(best['validation_mse_z'], checkpoint['validation_mse_z'], rtol=1e-12)
        assert len(history) == result['epochs_run']
        assert history['epoch'].tolist() == list(range(1, len(history) + 1))
        assert len(history) == 100 or len(history) - int(best['epoch']) == 15
        for split in ['validation', 'test']:
            samples = pd.read_csv(ROOT / 'data' / 'model' / (split + '.csv'))
            predictions = pd.read_csv(run / (split + '_predictions.csv'))
            metrics = pd.read_csv(run / ('metrics.csv' if split == 'test' else 'validation_metrics.csv'))
            assert predictions['date'].tolist() == samples['target_date'].tolist() == original_predictions[split]['date'].tolist()
            assert np.allclose(predictions['observed_m3_s'], samples['target_m3_s'])
            for column in ['observed_m3_s', 'persistence_m3_s', 'linear_regression_m3_s']:
                assert np.array_equal(predictions[column], original_predictions[split][column])
            x = torch.tensor((samples[config['sample_preparation']['feature_columns']].to_numpy() - scaler['mean_m3_s']) / scaler['std_m3_s'], dtype=torch.float32).unsqueeze(1)
            with torch.no_grad():
                restored = model(x).squeeze(1).numpy().astype(np.float64) * scaler['std_m3_s'] + scaler['mean_m3_s']
            assert np.allclose(restored, predictions['cnn_m3_s'], rtol=1e-12, atol=1e-10)
            for row in metrics.to_dict('records'):
                y, p = predictions['observed_m3_s'], predictions[row['model'] + '_m3_s']
                assert len(predictions) == row['n']
                assert np.isclose(row['MAE_m3_s'], mean_absolute_error(y, p), rtol=1e-12)
                assert np.isclose(row['RMSE_m3_s'], np.sqrt(mean_squared_error(y, p)), rtol=1e-12)
                assert np.isclose(row['NSE'], r2_score(y, p), rtol=1e-12)
                assert row['negative_predictions'] == int((p < 0).sum())
                if split == 'test':
                    metrics_by_model.append(dict(seed=expected_seed, **row))
            if split == 'test':
                cnn = metrics.loc[metrics['model'].eq('cnn')].iloc[0].to_dict()
                per_run.append(dict(seed=expected_seed, epochs_run=len(history), best_epoch=int(best['epoch']),
                                    best_validation_mse_z=float(best['validation_mse_z']),
                                    **{k: v for k, v in cnn.items() if k != 'model'}, output_directory=str(run)))
        verification = {'status': 'PASS', 'seed': expected_seed, 'training_settings_match_original': True,
                        'dates_and_baseline_predictions_match_original': True,
                        'independent_metrics': 'scikit-learn MAE, sqrt(MSE), R2 (= NSE)',
                        'checkpoint_predictions': 'all 731 validation and 1096 test values reproduced',
                        'source_hashes_unchanged': True}
        if expected_seed != 42:
            (run / 'verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    per_run = pd.DataFrame(per_run)
    metrics_by_model = pd.DataFrame(metrics_by_model)
    assert len(per_run) == 5 and per_run['seed'].tolist() == plan['seeds']
    per_run.to_csv(summary_dir / 'cnn_per_seed.csv', index=False, encoding='utf-8-sig')
    metrics_by_model.to_csv(summary_dir / 'all_test_metrics.csv', index=False, encoding='utf-8-sig')
    aggregate_rows = []
    for model in ['persistence', 'linear_regression', 'cnn']:
        part = metrics_by_model.loc[metrics_by_model['model'].eq(model)]
        row = {'model': model, 'runs': 5}
        for metric in ['MAE_m3_s', 'RMSE_m3_s', 'NSE']:
            row.update({metric + '_mean': float(part[metric].mean()), metric + '_std': float(part[metric].std(ddof=1)),
                        metric + '_min': float(part[metric].min()), metric + '_max': float(part[metric].max())})
        row['negative_predictions_total'] = int(part['negative_predictions'].sum())
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(summary_dir / 'metrics_mean_std.csv', index=False, encoding='utf-8-sig')
    wins = {}
    for baseline_model in ['persistence', 'linear_regression']:
        reference = metrics_by_model.loc[metrics_by_model['model'].eq(baseline_model)].iloc[0]
        wins[baseline_model] = {'MAE': int((per_run['MAE_m3_s'] < reference['MAE_m3_s']).sum()),
                                'RMSE': int((per_run['RMSE_m3_s'] < reference['RMSE_m3_s']).sum()),
                                'NSE': int((per_run['NSE'] > reference['NSE']).sum())}
    check = {'status': 'PASS', 'run_directories': [str(path) for path in runs], 'seeds': plan['seeds'],
             'additional_runs': 4, 'original_run_preserved': True, 'standard_deviation_ddof': 1,
             'config_comparison': 'All fields except seed and training script hash match original',
             'test_dates_per_run': 1096, 'baseline_predictions_identical': True,
             'all_checkpoints_and_metrics_independently_verified': True, 'cnn_wins_out_of_5': wins,
             'interpretation': 'Variation across training seeds on a fixed test period; not a confidence interval or significance test.'}
    (summary_dir / 'verification.json').write_text(json.dumps(check, ensure_ascii=False, indent=2), encoding='utf-8')
    rows = '\n'.join('| %d | %d | %d | %.4f | %.4f | %.6f | %d |' %
                     (r.seed, r.epochs_run, r.best_epoch, r.MAE_m3_s, r.RMSE_m3_s, r.NSE, r.negative_predictions)
                     for r in per_run.itertuples())
    average_rows = '\n'.join('| %s | %.4f ± %.4f | %.4f ± %.4f | %.6f ± %.6f |' %
                             (r['model'], r['MAE_m3_s_mean'], r['MAE_m3_s_std'], r['RMSE_m3_s_mean'], r['RMSE_m3_s_std'], r['NSE_mean'], r['NSE_std'])
                             for r in aggregate_rows)
    report = f'''# 5次随机种子稳定性实验

固定种子42、0、1、2、3；沿用已有种子42结果，本次仅新增0、1、2、3共4次训练。
数据、划分、网络、学习率、批量、早停和软件版本与首次实验一致，仅改变随机种子。
每次使用单独Python进程训练，最佳模型仍按验证损失选择；未按测试结果调参或筛选运行。
仍以种子42曲线作为示例，不替换成测试集表现最好的一次。

## 每次CNN测试结果

| 种子 | 训练轮数 | 最佳轮次 | MAE（m³/s） | RMSE（m³/s） | NSE | 负预测数 |
|---|---:|---:|---:|---:|---:|---:|
{rows}

## 均值与标准差

按5次各自的指标汇总，标准差为样本标准差（ddof=1）；不是先平均预测再计算指标，没有构造集成模型。
持续性预测和线性回归是确定性基线，各次预测完全相同；其标准差为0不代表不存在预测误差。

| 方法 | MAE（m³/s）均值 ± 标准差 | RMSE（m³/s）均值 ± 标准差 | NSE均值 ± 标准差 |
|---|---:|---:|---:|
{average_rows}

CNN优于持续性预测的次数：MAE {wins['persistence']['MAE']}/5，RMSE {wins['persistence']['RMSE']}/5，NSE {wins['persistence']['NSE']}/5。
CNN优于线性回归的次数：MAE {wins['linear_regression']['MAE']}/5，RMSE {wins['linear_regression']['RMSE']}/5，NSE {wins['linear_regression']['NSE']}/5。

## 核对与结果解释

已独立复算全部运行的验证/测试指标，并加载每个模型权重重现全部预测；核对了数据哈希、相同测试日期、固定配置、基线一致性和最佳轮次。
各次完整权重、日志、配置、预测和图表分别保存在对应运行目录；本目录plan.json记录预定种子，verification.json记录目录和检查结果。
cnn_per_seed.csv列出每次结果；metrics_mean_std.csv为汇总表；all_test_metrics.csv保留全部基线和CNN指标。
这是固定站点、固定测试期下训练随机性的检查，标准差不等于置信区间，未做统计显著性或跨站泛化验证。
''' 
    (summary_dir / 'stability_summary.md').write_text(report, encoding='utf-8')
    print(per_run.drop(columns=['output_directory']).to_string(index=False), flush=True)
    print(aggregate[['model', 'MAE_m3_s_mean', 'MAE_m3_s_std', 'RMSE_m3_s_mean', 'RMSE_m3_s_std', 'NSE_mean', 'NSE_std']].to_string(index=False), flush=True)
    print('Summary: ' + str(summary_dir), flush=True)
    print(json.dumps(check, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, help='Optional existing seed42 run from identical prepared data')
    main(parser.parse_args().baseline)
