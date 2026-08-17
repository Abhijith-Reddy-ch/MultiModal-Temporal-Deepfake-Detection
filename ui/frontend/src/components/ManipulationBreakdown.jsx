'use client';

import { BarChart, Bar, XAxis, YAxis, Tooltip, Cell, ResponsiveContainer } from 'recharts';
import { manipulationLabel, pct } from '@/lib/format';

function ChartTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const { name, value } = payload[0].payload;
  return (
    <div className="rounded-md border border-line-grid bg-surface px-3 py-2 text-xs shadow-sm dark:border-line-grid-dark dark:bg-surface-dark">
      <p className="font-medium">{name}</p>
      <p className="text-ink-muted">{pct(value)}</p>
    </div>
  );
}

export default function ManipulationBreakdown({ breakdown }) {
  const data = Object.entries(breakdown)
    .map(([key, value]) => ({ key, name: manipulationLabel(key), value }))
    .sort((a, b) => b.value - a.value);
  const topKey = data[0]?.key;

  return (
    <div className="rounded-xl border border-line-grid bg-surface p-6 dark:border-line-grid-dark dark:bg-surface-dark">
      <h3 className="text-sm font-medium text-ink-secondary dark:text-ink-secondary-dark">
        Manipulation type likelihood
      </h3>
      <div className="mt-4" style={{ width: '100%', height: 260 }}>
        <ResponsiveContainer>
          <BarChart data={data} layout="vertical" margin={{ left: 8, right: 24, top: 4, bottom: 4 }}>
            <XAxis
              type="number"
              domain={[0, 1]}
              tickFormatter={(v) => `${Math.round(v * 100)}%`}
              stroke="#898781"
              tick={{ fontSize: 11, fill: '#898781' }}
              axisLine={{ stroke: '#c3c2b7' }}
              tickLine={false}
            />
            <YAxis
              type="category"
              dataKey="name"
              width={150}
              stroke="#898781"
              tick={{ fontSize: 12, fill: '#52514e' }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip content={<ChartTooltip />} cursor={{ fill: 'rgba(137,135,129,0.08)' }} />
            <Bar dataKey="value" radius={[0, 4, 4, 0]} maxBarSize={20}>
              {data.map((entry) => (
                <Cell key={entry.key} fill={entry.key === topKey ? '#2a78d6' : '#9ec5f4'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
