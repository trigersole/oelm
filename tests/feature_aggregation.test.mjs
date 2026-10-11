import assert from 'node:assert/strict';
import test from 'node:test';

import {
  STATIC_STATISTICS,
  aggregateStaticWindow,
} from '../src/data/featureAggregation.js';
import { FEATURE_ORDER } from '../src/data/featureOrder.js';

test('XGB-260 aggregation matches NumPy-style static statistics', () => {
  const frames = [
    { alpha: 1, beta: 2 },
    { alpha: 2, beta: 2 },
    { alpha: 3, beta: 4 },
    { alpha: 4, beta: 8 },
  ];
  const result = aggregateStaticWindow(frames, ['alpha', 'beta']);

  assert.deepEqual(STATIC_STATISTICS, ['mean', 'std', 'min', 'max', 'median']);
  assert.equal(result.alpha_mean, 2.5);
  assert.equal(result.alpha_std, Math.sqrt(1.25));
  assert.equal(result.alpha_min, 1);
  assert.equal(result.alpha_max, 4);
  assert.equal(result.alpha_median, 2.5);
  assert.equal(result.beta_mean, 4);
  assert.equal(result.beta_std, Math.sqrt(6));
  assert.equal(result.beta_median, 3);
  assert.deepEqual(Object.keys(result), [
    'alpha_mean', 'alpha_std', 'alpha_min', 'alpha_max', 'alpha_median',
    'beta_mean', 'beta_std', 'beta_min', 'beta_max', 'beta_median',
  ]);
});

test('browser feature order is the exact 260-feature static schema', () => {
  assert.equal(FEATURE_ORDER.length, 260);
  assert.equal(new Set(FEATURE_ORDER).size, 260);
  assert.ok(FEATURE_ORDER.every(name =>
    STATIC_STATISTICS.some(statistic => name.endsWith(`_${statistic}`))
  ));
  assert.ok(FEATURE_ORDER.every(name => !name.endsWith('_skew') && !name.endsWith('_kurt')));
});
