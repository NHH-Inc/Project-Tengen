import type { RunResult } from '../api/shapes';

export function AnalysisPanel({
  result,
  loading,
  error,
}: {
  result: RunResult | null;
  loading: boolean;
  error: string | null;
}) {
  if (loading) {
    return <section className="panel"><p className="empty">Loading analysis details…</p></section>;
  }
  if (error) {
    return <section className="panel"><p className="error">Could not load analysis details: {error}</p></section>;
  }
  if (!result) {
    return (
      <section className="panel">
        <div className="panel-head"><h2>Analysis details</h2></div>
        <p className="empty">This job has no result file yet.</p>
      </section>
    );
  }

  const detectorConfigured = result.modelVersion !== 'detector-unconfigured';
  const analyzedPercent = result.framesTotal > 0
    ? Math.round((100 * result.framesAnalyzed) / result.framesTotal)
    : 0;

  return (
    <>
      <section className="panel">
        <div className="panel-head">
          <h2>Analysis details</h2>
          <span className={`analysis-status ${detectorConfigured ? 'ready' : 'waiting'}`}>
            {detectorConfigured ? 'detector used' : 'detector not configured'}
          </span>
        </div>
        <div className="analysis-grid">
          <Metric label="Video frames decoded" value={result.framesTotal.toLocaleString()} />
          <Metric label="Frames sent to detector" value={`${result.framesAnalyzed.toLocaleString()} · ${analyzedPercent}%`} />
          <Metric label="Robot tracks" value={result.tracksEmitted.toLocaleString()} />
          <Metric label="Events emitted" value={result.eventsEmitted.toLocaleString()} />
          <Metric label="Box sample rate" value={`${result.boxSampleRate.toFixed(2)} Hz`} />
          <Metric label="Frames skipped at cuts" value={result.framesSkippedShotChange.toLocaleString()} />
          <Metric
            label="Field mapping"
            value={result.homographyOk ? (result.homographySource ?? 'available') : 'not available'}
          />
          <Metric label="Model" value={result.modelVersion} mono />
        </div>
      </section>

      {!detectorConfigured && (
        <section className="panel analysis-callout waiting">
          <h2>This run intentionally has no robot boxes</h2>
          <p>
            The video pipeline worked, but this machine has no trained RF-DETR model selected.
            Train/export the model on Robert’s CUDA computer, then create the ignored local
            <code> analysis/config/detector.local.json</code> and set <code>FRC_DETECTOR_CONFIG</code>
            before queuing a new job.
          </p>
        </section>
      )}

      {detectorConfigured && result.tracksEmitted === 0 && (
        <section className="panel analysis-callout warning">
          <h2>Detector ran but found no tracks</h2>
          <p>Check the reviewed training data, model path, score threshold, and camera angle before treating this match as empty.</p>
        </section>
      )}
    </>
  );
}

function Metric({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="analysis-metric">
      <span>{label}</span>
      <strong className={mono ? 'mono' : undefined}>{value}</strong>
    </div>
  );
}
