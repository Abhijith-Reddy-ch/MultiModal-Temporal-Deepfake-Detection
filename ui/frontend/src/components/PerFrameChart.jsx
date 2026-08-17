'use client';

import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { pct } from '@/lib/format';

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-md border border-line-grid bg-surface px-3 py-2 text-xs shadow-sm dark:border-line-grid-dark dark:bg-surface-dark">
      <p className="font-medium">Frame {label}</p>
      {payload.map((p) => (
        <p key={p.dataKey} className="text-ink-muted">
          {p.name}: {pct(p.value)}
        </p>
      ))}
    </div>
  );
}

export default function PerFrameChart({ frames }) {
  const data = frames.map((f) => ({
    frame: f.frame,
    Visual: f.visual_importance,
    Audio: f.audio_importance,
  }));

  return (
    <div className="rounded-xl border border-line-grid bg-surface p-6 dark:border-line-grid-dark dark:bg-surface-dark">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-ink-secondary dark:text-ink-secondary-dark">
          Attention rollout per frame
        </h3>
        <ul className="flex gap-4 text-xs">
          <li className="flex items-center gap-1.5">
            <span className="h-0.5 w-3 rounded bg-series-1" aria-hidden="true" />
            Visual
          </li>
          <li className="flex items-center gap-1.5">
            <span className="h-0.5 w-3 rounded bg-series-2" aria-hidden="true" />
            Audio
          </li>
        </ul>
      </div>
      <div className="mt-4" style={{ width: '100%', height: 220 }}>
        <ResponsiveContainer>
          <LineChart data={data} margin={{ left: 0, right: 16, top: 8, bottom: 4 }}>
            <XAxis
              dataKey="frame"
              tickFormatter={(v) => `F${v}`}
              stroke="#898781"
              tick={{ fontSize: 11, fill: '#898781' }}
              axisLine={{ stroke: '#c3c2b7' }}
              tickLine={false}
            />
            <YAxis
              tickFormatter={(v) => `${Math.round(v * 100)}%`}
              stroke="#898781"
              tick={{ fontSize: 11, fill: '#898781' }}
              axisLine={false}
              tickLine={false}
              width={40}
            />
            <Tooltip content={<ChartTooltip />} />
            <Line type="monotone" dataKey="Visual" stroke="#2a78d6" strokeWidth={2} dot={{ r: 4 }} />
            <Line type="monotone" dataKey="Audio" stroke="#1baf7a" strokeWidth={2} dot={{ r: 4 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
