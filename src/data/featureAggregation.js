export const STATIC_STATISTICS = ['mean', 'std', 'min', 'max', 'median'];

// Keep this calculation aligned with NumPy's mean/std/min/max/median over the
// frame axis. In particular, std is the population standard deviation (ddof=0).
export function aggregateStaticWindow(frames, blendshapeNames) {
  if (!frames.length) return null;

  const result = {};
  for (const name of blendshapeNames) {
    const values = frames.map(frame => Number(frame[name] ?? 0));
    const count = values.length;
    const mean = values.reduce((sum, value) => sum + value, 0) / count;
    const variance = values.reduce(
      (sum, value) => sum + (value - mean) ** 2,
      0,
    ) / count;
    let min = values[0];
    let max = values[0];
    for (const value of values) {
      if (value < min) min = value;
      if (value > max) max = value;
    }
    const sorted = [...values].sort((left, right) => left - right);
    const median = count % 2 === 0
      ? (sorted[count / 2 - 1] + sorted[count / 2]) / 2
      : sorted[Math.floor(count / 2)];

    result[`${name}_mean`] = mean;
    result[`${name}_std`] = Math.sqrt(variance);
    result[`${name}_min`] = min;
    result[`${name}_max`] = max;
    result[`${name}_median`] = median;
  }
  return result;
}
