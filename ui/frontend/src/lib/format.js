export function pct(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return `${(value * 100).toFixed(digits)}%`;
}

export function bytes(n) {
  if (!n) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v.toFixed(v < 10 && i > 0 ? 1 : 0)} ${units[i]}`;
}

export function seconds(n) {
  if (!n && n !== 0) return '—';
  return `${n.toFixed(2)}s`;
}

const MANIPULATION_LABELS = {
  real: 'Real',
  faceswap: 'Face Swap',
  'faceswap-wav2lip': 'Face Swap + Wav2Lip',
  fsgan: 'FSGAN',
  'fsgan-wav2lip': 'FSGAN + Wav2Lip',
  rtvc: 'Voice Clone (RTVC)',
  wav2lip: 'Wav2Lip',
  video_retalking: 'Video Retalking',
  unknown_fake: 'Unknown Manipulation',
};

export function manipulationLabel(key) {
  return MANIPULATION_LABELS[key] || key;
}

export const REGION_LABELS = {
  face: 'Face',
  eyes: 'Eyes',
  lips: 'Lips',
  jaw: 'Jaw',
};
