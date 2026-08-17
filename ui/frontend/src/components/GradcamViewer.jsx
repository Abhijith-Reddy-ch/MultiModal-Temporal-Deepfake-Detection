'use client';

import { useMemo, useState } from 'react';
import clsx from 'clsx';
import { REGION_LABELS } from '@/lib/format';

const REGION_ORDER = ['face', 'eyes', 'lips', 'jaw'];

export default function GradcamViewer({ images }) {
  const regions = useMemo(
    () => REGION_ORDER.filter((r) => images[r]),
    [images]
  );
  const [activeRegion, setActiveRegion] = useState(regions[0]);

  if (regions.length === 0) {
    return (
      <div className="rounded-xl border border-line-grid bg-surface p-6 text-sm text-ink-muted dark:border-line-grid-dark dark:bg-surface-dark">
        No Grad-CAM frames were returned for this video.
      </div>
    );
  }

  const region = activeRegion && images[activeRegion] ? activeRegion : regions[0];
  const frames = Object.entries(images[region] || {}).sort(
    (a, b) => Number(a[0]) - Number(b[0])
  );

  return (
    <div className="rounded-xl border border-line-grid bg-surface p-6 dark:border-line-grid-dark dark:bg-surface-dark">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-ink-secondary dark:text-ink-secondary-dark">
          Grad-CAM attention by region
        </h3>
        <div className="flex gap-1 rounded-lg bg-line-grid/40 p-1 dark:bg-line-grid-dark/40">
          {regions.map((r) => (
            <button
              key={r}
              type="button"
              onClick={() => setActiveRegion(r)}
              className={clsx(
                'rounded-md px-3 py-1 text-xs font-medium transition-colors',
                r === region
                  ? 'bg-series-1 text-white'
                  : 'text-ink-secondary hover:bg-line-grid dark:text-ink-secondary-dark dark:hover:bg-line-grid-dark'
              )}
            >
              {REGION_LABELS[r] || r}
            </button>
          ))}
        </div>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
        {frames.map(([frameIdx, b64]) => (
          <figure key={frameIdx} className="overflow-hidden rounded-lg border border-line-grid dark:border-line-grid-dark">
            <img
              src={`data:image/jpeg;base64,${b64}`}
              alt={`Grad-CAM overlay, ${REGION_LABELS[region] || region} region, frame ${frameIdx}`}
              className="aspect-square w-full object-cover"
            />
            <figcaption className="px-2 py-1 text-center text-xs text-ink-muted">
              Frame {frameIdx}
            </figcaption>
          </figure>
        ))}
      </div>
    </div>
  );
}
