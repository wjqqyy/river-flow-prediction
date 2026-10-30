# -*- coding: utf-8 -*-
"""Verify the five published runs offline without training or editing data."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch

from run_experiment import ROOT, load_data, make_model, sha256

RUNS = [('20261030T005841_seed42', 42), ('20261030T014754_seed0', 0),
        ('20261030T014803_seed1', 1), ('20261030T014813_seed2', 2),
        ('20261030T014822_seed3', 3)]


def main():
    listing = ROOT / 'docs' / 'FILE_MANIFEST.json'
    if listing.exists():
        for item in json.loads(listing.read_text(encoding='utf-8'))['files']:
            path = ROOT / item['path']
            assert path.stat().st_size == item['bytes'], item['path']
            assert sha256(path) == item['sha256'], item['path']
    manifest_path = ROOT / 'data' / 'raw' / '20261029T103540Z' / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    for item in manifest['requests']:
        assert sha256(manifest_path.parent / item['file']) == item['sha256']
    frames, _, _, _, _ = load_data()
    torch.set_num_threads(1)
    rows = []
    baseline_predictions = {}
    reference_config = None
    for directory, seed in RUNS:
        folder = ROOT / 'outputs' / directory
        config = json.loads((folder / 'experiment_config.json').read_text(encoding='utf-8'))
        if reference_config is None:
            reference_config = config
        for key in set(reference_config) - {'seed', 'training_script_sha256'}:
            assert config[key] == reference_config[key], (seed, key)
        for name, digest in config['input_sha256'].items():
            assert sha256(ROOT / Path(name.replace('\\', '/'))) == digest, name
        assert sha256(folder / 'run_experiment.py') == config['training_script_sha256']
        result = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
        assert result['status'] == 'complete' and config['seed'] == result['seed'] == seed
        checkpoint = torch.load(folder / 'cnn_best.pt', map_location='cpu', weights_only=True)
        assert checkpoint['seed'] == seed
        model = make_model()
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        scaler = json.loads((folder / 'normalization.json').read_text(encoding='utf-8'))
        linear = json.loads((folder / 'linear_model.json').read_text(encoding='utf-8'))
        history = pd.read_csv(folder / 'training_history.csv')
        best = history.loc[history['validation_mse_z'].idxmin()]
        assert int(best['epoch']) == checkpoint['epoch'] == result['best_epoch']
        assert len(history) == result['epochs_run']
        assert np.isclose(best['validation_mse_z'], checkpoint['validation_mse_z'], rtol=1e-12)
        for split in ['validation', 'test']:
            frame = frames[split]
            pred = pd.read_csv(folder / (split + '_predictions.csv'))
            metric_file = 'metrics.csv' if split == 'test' else 'validation_metrics.csv'
            metrics = pd.read_csv(folder / metric_file)
            assert pred['date'].tolist() == frame['target_date'].dt.strftime('%Y-%m-%d').tolist()
            np.testing.assert_allclose(pred['observed_m3_s'], frame['target_m3_s'], rtol=1e-12)
            x = frame[config['sample_preparation']['feature_columns']].to_numpy()
            tensor = torch.tensor((x - scaler['mean_m3_s']) / scaler['std_m3_s'], dtype=torch.float32).unsqueeze(1)
            with torch.no_grad():
                restored = model(tensor).squeeze(1).numpy().astype(np.float64)
            restored = restored * scaler['std_m3_s'] + scaler['mean_m3_s']
            np.testing.assert_allclose(restored, pred['cnn_m3_s'], rtol=1e-12, atol=1e-10)
            np.testing.assert_allclose(x[:, -1], pred['persistence_m3_s'], rtol=1e-12)
            np.testing.assert_allclose(x @ np.asarray(linear['coefficients']) + linear['intercept_m3_s'],
                                       pred['linear_regression_m3_s'], rtol=1e-12, atol=1e-10)
            for column in ['observed_m3_s', 'persistence_m3_s', 'linear_regression_m3_s']:
                key = (split, column)
                if key in baseline_predictions:
                    np.testing.assert_array_equal(baseline_predictions[key], pred[column].to_numpy())
                baseline_predictions[key] = pred[column].to_numpy()
            for row in metrics.to_dict('records'):
                y, p = pred['observed_m3_s'], pred[row['model'] + '_m3_s']
                actual = [mean_absolute_error(y, p), np.sqrt(mean_squared_error(y, p)), r2_score(y, p)]
                recorded = [row['MAE_m3_s'], row['RMSE_m3_s'], row['NSE']]
                np.testing.assert_allclose(actual, recorded, rtol=1e-12)
                assert len(pred) == row['n'] and int((p < 0).sum()) == row['negative_predictions']
                if split == 'test':
                    rows.append(dict(seed=seed, **row))
        print('PASS seed %d: checkpoint, dates, baseline predictions and metrics' % seed)
    rows = pd.DataFrame(rows)
    summary = ROOT / 'outputs' / '20261030T014752_stability_5seeds'
    published = pd.read_csv(summary / 'all_test_metrics.csv')
    pd.testing.assert_frame_equal(rows[published.columns].reset_index(drop=True), published, check_dtype=False,
                                  rtol=1e-12, atol=1e-12)
    aggregate = pd.read_csv(summary / 'metrics_mean_std.csv').set_index('model')
    for model, part in rows.groupby('model'):
        for metric in ['MAE_m3_s', 'RMSE_m3_s', 'NSE']:
            np.testing.assert_allclose([part[metric].mean(), part[metric].std(ddof=1)],
                                       [aggregate.loc[model, metric + '_mean'], aggregate.loc[model, metric + '_std']],
                                       rtol=1e-12, atol=1e-12)
    print('PASS: raw-source integrity, prepared samples, all five runs and aggregate statistics')


if __name__ == '__main__':
    main()
