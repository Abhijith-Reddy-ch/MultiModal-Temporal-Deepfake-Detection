'use client';

import { useState } from 'react';
import { AlertCircle } from 'lucide-react';
import HealthBadge from '@/components/HealthBadge';
import UploadPanel from '@/components/UploadPanel';
import VerdictTile from '@/components/VerdictTile';
import ModalityChart from '@/components/ModalityChart';
import ManipulationBreakdown from '@/components/ManipulationBreakdown';
import MetadataTable from '@/components/MetadataTable';
import PerFrameChart from '@/components/PerFrameChart';
import GradcamViewer from '@/components/GradcamViewer';
import { predict, getGradcam } from '@/lib/api';
import { pct } from '@/lib/format';

export default function Home() {
  const [file, setFile] = useState(null);
  const [busy, setBusy] = useState(null);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [gradcam, setGradcam] = useState(null);

  function handleFileSelected(next) {
    setFile(next);
    setResult(null);
    setGradcam(null);
    setError(null);
  }

  async function handlePredict() {
    if (!file) return;
    setBusy('predict');
    setError(null);
    setGradcam(null);
    try {
      const data = await predict(file);
      setResult(data);
    } catch (err) {
      setError(err.message || 'Prediction failed');
    } finally {
      setBusy(null);
    }
  }

  async function handleExplain() {
    if (!file) return;
    setBusy('explain');
    setError(null);
    try {
      const data = await getGradcam(file);
      setGradcam(data);
    } catch (err) {
      setError(err.message || 'Grad-CAM failed');
    } finally {
      setBusy(null);
    }
  }

  return (
    <main className="mx-auto max-w-4xl px-6 py-10">
      <header className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">DeepFake Detection</h1>
          <p className="mt-1 text-sm text-ink-secondary dark:text-ink-secondary-dark">
            Upload a video to get a verdict, modality breakdown, and Grad-CAM explainability.
          </p>
        </div>
        <HealthBadge />
      </header>

      <UploadPanel
        file={file}
        onFileSelected={handleFileSelected}
        onPredict={handlePredict}
        onExplain={handleExplain}
        busy={busy}
      />

      {error && (
        <div className="mt-6 flex items-center gap-2 rounded-lg bg-status-critical/10 px-4 py-3 text-sm text-status-critical">
          <AlertCircle className="h-4 w-4 shrink-0" aria-hidden="true" />
          {error}
        </div>
      )}

      {result && (
        <section className="mt-8 space-y-6">
          <VerdictTile
            prediction={result.prediction}
            confidence={result.confidence}
            fakeProbability={result.fake_probability}
          />
          <ModalityChart visual={result.modality_scores.visual} audio={result.modality_scores.audio} />
          <ManipulationBreakdown breakdown={result.manipulation_type_breakdown} />
          <MetadataTable metadata={result.metadata} />
        </section>
      )}

      {gradcam && (
        <section className="mt-8 space-y-6">
          <div className="flex items-center justify-between rounded-xl border border-line-grid bg-surface p-4 text-sm dark:border-line-grid-dark dark:bg-surface-dark">
            <span className="text-ink-secondary dark:text-ink-secondary-dark">Fake probability</span>
            <span className="tabular text-lg font-semibold">{pct(gradcam.fake_probability)}</span>
          </div>
          <ModalityChart
            visual={gradcam.gmu_gate_visual_weight}
            audio={gradcam.gmu_gate_audio_weight}
            title="GMU gate weight"
          />
          <PerFrameChart frames={gradcam.attention_rollout_per_frame} />
          <ManipulationBreakdown breakdown={gradcam.manipulation_type_breakdown} />
          <GradcamViewer images={gradcam.gradcam_images} />
        </section>
      )}
    </main>
  );
}
