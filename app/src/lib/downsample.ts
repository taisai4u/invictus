/**
 * Largest Triangle Three Buckets (LTTB) downsampling.
 * Reduces a time series to `threshold` points while preserving visual shape
 * far better than naive nth-point decimation.
 *
 * Reference: Sveinn Steinarsson, "Downsampling Time Series for Visual
 * Representation", 2013.
 */
export function lttb<T>(
  data: T[],
  threshold: number,
  x: (d: T) => number,
  y: (d: T) => number,
): T[] {
  const len = data.length;
  if (threshold >= len || threshold <= 2) return data;

  const sampled: T[] = [data[0]];
  const bucketSize = (len - 2) / (threshold - 2);

  let prevIndex = 0;

  for (let i = 0; i < threshold - 2; i++) {
    const bucketStart = Math.floor((i + 1) * bucketSize) + 1;
    const bucketEnd = Math.min(Math.floor((i + 2) * bucketSize) + 1, len - 1);

    // Average of next bucket (used as the "target" triangle vertex)
    const nextStart = Math.floor((i + 2) * bucketSize) + 1;
    const nextEnd = Math.min(Math.floor((i + 3) * bucketSize) + 1, len - 1);
    let avgX = 0;
    let avgY = 0;
    const nextCount = nextEnd - nextStart;
    if (nextCount > 0) {
      for (let j = nextStart; j < nextEnd; j++) {
        avgX += x(data[j]);
        avgY += y(data[j]);
      }
      avgX /= nextCount;
      avgY /= nextCount;
    } else {
      avgX = x(data[len - 1]);
      avgY = y(data[len - 1]);
    }

    const prevX = x(data[prevIndex]);
    const prevY = y(data[prevIndex]);

    let maxArea = -1;
    let maxIndex = bucketStart;

    for (let j = bucketStart; j < bucketEnd; j++) {
      const area = Math.abs(
        (prevX - avgX) * (y(data[j]) - prevY) -
        (prevX - x(data[j])) * (avgY - prevY),
      );
      if (area > maxArea) {
        maxArea = area;
        maxIndex = j;
      }
    }

    sampled.push(data[maxIndex]);
    prevIndex = maxIndex;
  }

  sampled.push(data[len - 1]);
  return sampled;
}
