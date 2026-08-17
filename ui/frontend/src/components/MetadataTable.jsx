import { AlertTriangle } from 'lucide-react';
import { bytes, seconds } from '@/lib/format';

const ROWS = [
  ['Duration', (m) => seconds(m.duration)],
  ['File size', (m) => bytes(m.file_size)],
  ['Container', (m) => m.container_format],
  ['Resolution', (m) => (m.width && m.height ? `${m.width}x${m.height}` : '—')],
  ['Video codec', (m) => m.video_codec],
  ['Audio codec', (m) => m.audio_codec],
  ['Frame rate', (m) => (m.fps ? `${m.fps.toFixed(2)} fps` : '—')],
  ['Variable frame rate', (m) => (m.vfr ? 'Yes' : 'No')],
];

const ANOMALY_CHECKS = [
  ['vfr', 'Variable frame rate detected'],
  ['missing_audio', 'No audio stream'],
  ['missing_video', 'No video stream'],
];

export default function MetadataTable({ metadata }) {
  const anomalies = ANOMALY_CHECKS.filter(([key]) => metadata[key]);
  if (!metadata.has_creation_time) anomalies.push(['has_creation_time', 'Missing creation timestamp']);

  return (
    <div className="rounded-xl border border-line-grid bg-surface p-6 dark:border-line-grid-dark dark:bg-surface-dark">
      <h3 className="text-sm font-medium text-ink-secondary dark:text-ink-secondary-dark">Container metadata</h3>

      <dl className="mt-4 grid grid-cols-2 gap-x-6 gap-y-3 text-sm sm:grid-cols-4">
        {ROWS.map(([label, get]) => (
          <div key={label}>
            <dt className="text-xs text-ink-muted">{label}</dt>
            <dd className="tabular font-medium">{get(metadata) ?? '—'}</dd>
          </div>
        ))}
      </dl>

      {anomalies.length > 0 && (
        <div className="mt-5 rounded-lg bg-status-warning/10 p-3">
          <div className="flex items-center gap-2 text-xs font-medium text-status-serious">
            <AlertTriangle className="h-4 w-4" aria-hidden="true" />
            Forensic anomalies
          </div>
          <ul className="mt-1.5 list-inside list-disc text-xs text-ink-secondary dark:text-ink-secondary-dark">
            {anomalies.map(([key, label]) => (
              <li key={key}>{label}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
