import { pct } from '@/lib/format';

export default function ModalityChart({ visual, audio, title = 'Modality contribution' }) {
  const visualPct = visual * 100;
  const audioPct = audio * 100;
  const visualLabelFits = visualPct >= 12;
  const audioLabelFits = audioPct >= 12;

  return (
    <div className="rounded-xl border border-line-grid bg-surface p-6 dark:border-line-grid-dark dark:bg-surface-dark">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-ink-secondary dark:text-ink-secondary-dark">{title}</h3>
        <ul className="flex gap-4 text-xs">
          <li className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full bg-series-1" aria-hidden="true" />
            Visual
          </li>
          <li className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full bg-series-2" aria-hidden="true" />
            Audio
          </li>
        </ul>
      </div>

      <div className="mt-4 flex h-8 w-full overflow-hidden rounded-md" role="img" aria-label={`Visual ${pct(visual)}, audio ${pct(audio)}`}>
        <div
          className="flex items-center justify-center bg-series-1 text-xs font-medium text-white"
          style={{ width: `${visualPct}%`, marginRight: audioPct > 0 ? '2px' : 0 }}
        >
          {visualLabelFits ? pct(visual, 0) : ''}
        </div>
        <div
          className="flex items-center justify-center bg-series-2 text-xs font-medium text-white"
          style={{ width: `${audioPct}%` }}
        >
          {audioLabelFits ? pct(audio, 0) : ''}
        </div>
      </div>

      {(!visualLabelFits || !audioLabelFits) && (
        <div className="mt-2 flex justify-between text-xs text-ink-muted">
          <span>Visual {pct(visual, 0)}</span>
          <span>Audio {pct(audio, 0)}</span>
        </div>
      )}
    </div>
  );
}
