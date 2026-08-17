import { ShieldAlert, ShieldCheck } from 'lucide-react';
import { pct } from '@/lib/format';

export default function VerdictTile({ prediction, confidence, fakeProbability }) {
  const isFake = prediction === 'Fake';
  const color = isFake ? 'text-status-critical' : 'text-status-good';
  const Icon = isFake ? ShieldAlert : ShieldCheck;

  return (
    <div className="rounded-xl border border-line-grid bg-surface p-6 dark:border-line-grid-dark dark:bg-surface-dark">
      <div className="flex items-center gap-3">
        <Icon className={`h-9 w-9 ${color}`} aria-hidden="true" />
        <div>
          <p className="text-xs uppercase tracking-wide text-ink-muted">Verdict</p>
          <p className={`text-4xl font-semibold ${color}`}>{prediction}</p>
        </div>
      </div>
      <dl className="mt-5 grid grid-cols-2 gap-4 text-sm">
        <div>
          <dt className="text-ink-muted">Confidence</dt>
          <dd className="tabular text-lg font-medium">{pct(confidence)}</dd>
        </div>
        <div>
          <dt className="text-ink-muted">Fake probability</dt>
          <dd className="tabular text-lg font-medium">{pct(fakeProbability)}</dd>
        </div>
      </dl>
    </div>
  );
}
