import { useMemo, useState } from 'react';
import { extent, max, min } from '@visx/vendor/d3-array';
import * as allCurves from '@visx/curve';
import { LinePath } from '@visx/shape';
import { scaleTime, scaleLinear } from '@visx/scale';
import { MarkerCircle } from '@visx/marker';
import { lttb } from '../lib/downsample';

type CurveType = keyof typeof allCurves;

const DEFAULT_DOWNSAMPLE_THRESHOLD = 2000;

export type CurveProps<T extends Record<string, number>> = {
  series: T[];
  timeKey: keyof T;
  valueKey: keyof T;
  width: number;
  height: number;
  showControls?: boolean;
};

export default function LineChart<T extends Record<string, number>>({ width, height, series, timeKey, valueKey, showControls = true }: CurveProps<T>) {
  const [curveType] = useState<CurveType>('curveLinear');
  const [showPoints, setShowPoints] = useState<boolean>(false);

  const svgHeight = showControls ? height - 40 : height;

  const downsampled = useMemo(
    () =>
      lttb(
        series,
        Math.min(series.length, Math.max(width, DEFAULT_DOWNSAMPLE_THRESHOLD)),
        (d) => d[timeKey],
        (d) => d[valueKey],
      ),
    [series, width, timeKey, valueKey],
  );

  const xScale = useMemo(() => {
    const scale = scaleTime<number>({
      domain: extent(series, (d) => d[timeKey]) as [number, number],
    });
    scale.range([0, width - 50]);
    return scale;
  }, [series, timeKey, width]);

  const yScale = useMemo(() => {
    const scale = scaleLinear<number>({
      domain: [min(series, (d) => d[valueKey]) as number, max(series, (d) => d[valueKey]) as number],
    });
    scale.range([svgHeight - 2, 0]);
    return scale;
  }, [series, valueKey, svgHeight]);

  return (
    <div>
      {showControls && (
        <>
          <br />
        </>
      )}
      <svg width={width} height={svgHeight}>
        {showPoints && (
          <MarkerCircle id="marker-circle" fill="#333" size={2} refX={2} />
        )}
        <rect width={width} height={svgHeight} fill="#efefef" rx={14} ry={14} />
        {width > 8 && (
          <LinePath
            curve={allCurves[curveType]}
            data={downsampled}
            x={(d: T) => xScale(d[timeKey]) ?? 0}
            y={(d: T) => yScale(d[valueKey]) ?? 0}
            stroke="#333"
            strokeWidth={1}
            strokeOpacity={1}
            shapeRendering="geometricPrecision"
            markerMid={showPoints ? 'url(#marker-circle)' : undefined}
          />
        )}
      </svg>
    </div>
  );
}