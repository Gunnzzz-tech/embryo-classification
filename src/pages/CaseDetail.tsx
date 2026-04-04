import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { dataUrlToBlob, mlApiBase, predictLiveBirth } from '../api/mlPredict';
import { useCases } from '../context/CasesContext';
import type { EmbryoCase, EmbryoImage, EmbryoMlAssessment, VerificationStatus } from '../types';
import styles from './CaseDetail.module.css';

const statusLabels: Record<VerificationStatus, string> = {
  pending_review: 'Pending assessment',
  first_review: 'Clinical review',
  second_review: 'QA sign-off',
  verified: 'Verified',
  flagged: 'Flagged',
  discrepancy: 'Discrepancy',
};

type MlAssessmentOk = Extract<EmbryoMlAssessment, { probabilities: [number, number] }>;

/** Per-embryo ML, with legacy fallback: case-level score applied only to embryo #1 in old data. */
function getEmbryoMl(case_: EmbryoCase, index: number): EmbryoMlAssessment | undefined {
  const img = case_.embryoImages?.[index];
  if (img?.mlAssessment) return img.mlAssessment;
  if (index === 0 && case_.mlAssessment) return case_.mlAssessment;
  return undefined;
}

function embryoNeedsMlFetch(case_: EmbryoCase, index: number): boolean {
  const imgs = case_.embryoImages;
  if (!imgs?.[index]) return false;
  if (imgs[index].mlAssessment) return false;
  if (index === 0 && case_.mlAssessment) return false;
  return true;
}

function listOkEmbryoMls(case_: EmbryoCase): Array<{ embryoNumber: number; ml: MlAssessmentOk }> {
  const out: Array<{ embryoNumber: number; ml: MlAssessmentOk }> = [];
  (case_.embryoImages ?? []).forEach((img, i) => {
    const m = getEmbryoMl(case_, i);
    if (m && !('error' in m)) out.push({ embryoNumber: img.embryoNumber, ml: m });
  });
  return out;
}

function firstEmbryoMlForCaseRecord(
  images: EmbryoImage[],
  previousCaseMl?: EmbryoMlAssessment
): EmbryoMlAssessment | undefined {
  return images[0]?.mlAssessment ?? previousCaseMl;
}

function MlLbScores({
  ml,
  patientAge,
  title,
}: {
  ml: MlAssessmentOk;
  patientAge?: number;
  title?: string;
}) {
  const p0 = ml.probabilities[0];
  const p1 = ml.probabilities[1];
  return (
    <div className={styles.mlScores}>
      {title && <h3 className={styles.mlEmbryoTitle}>{title}</h3>}
      <dl className={styles.mlProbDl}>
        <dt>P(LB = 0)</dt>
        <dd>
          <strong>{(p0 * 100).toFixed(1)}%</strong>
          <span className={styles.mlProbNote}>
            {' '}
            — estimated chance the label is <strong>no live birth</strong> (as coded in training data).
          </span>
        </dd>
        <dt>P(LB = 1)</dt>
        <dd>
          <strong>{(p1 * 100).toFixed(1)}%</strong>
          <span className={styles.mlProbNote}>
            {' '}
            — estimated chance the label is <strong>live birth</strong> (same coding).
          </span>
        </dd>
      </dl>
      <p className={styles.mlLine}>
        <span className={styles.mlLabel}>Argmax class</span> <strong>{ml.predictedLbClass}</strong> (0 = LB negative,
        1 = LB positive)
        {' · '}
        <span className={styles.mlLabel}>sum</span> P(LB=0)+P(LB=1) = <strong>{((p0 + p1) * 100).toFixed(1)}%</strong>
      </p>
      <div className={styles.mlExplain}>
        <p className={styles.mlExplainLead}>What this means (mechanically)</p>
        <p>
          The model outputs two probabilities over two classes. After softmax they always sum to <strong>1</strong>.{' '}
          <strong>P(LB = 0)</strong> is the model&apos;s estimate that this image matches patterns seen for{' '}
          <strong>LB = 0</strong> in training; <strong>P(LB = 1)</strong> is the same for <strong>LB = 1</strong>. For
          example, if P(LB=1) = 0.73 and P(LB=0) = 0.27, the model is leaning toward the positive class on learned
          image patterns, with 0.73 + 0.27 = 1.
        </p>
        <p className={styles.mlMuted}>
          Based on this embryo image
          {patientAge != null ? ` and patient age ${patientAge}` : ''}. This is not a calibrated clinical probability —
          validate before any real-world use.
        </p>
      </div>
    </div>
  );
}

export function CaseDetail() {
  const { id } = useParams<{ id: string }>();
  const { getCaseById, updateCase } = useCases();
  const case_ = id ? getCaseById(id) : undefined;
  const [mlLoading, setMlLoading] = useState(false);

  const mlBase = mlApiBase();
  const embryoGallery = case_?.embryoImages ?? [];
  const pendingMlCount = case_
    ? embryoGallery.filter((_, i) => embryoNeedsMlFetch(case_, i)).length
    : 0;

  useEffect(() => {
    if (!id || !mlBase || !case_ || pendingMlCount === 0) return;

    const imgs = case_.embryoImages ?? [];
    const pending = imgs.map((_, i) => i).filter((i) => embryoNeedsMlFetch(case_, i));
    let cancelled = false;
    setMlLoading(true);

    (async () => {
      const next = [...imgs];
      try {
        await Promise.all(
          pending.map(async (i) => {
            const img = imgs[i];
            if (!img) return;
            try {
              const blob = await dataUrlToBlob(img.dataUrl);
              const data = await predictLiveBirth(mlBase, blob, { age: case_.patientAge });
              if (cancelled) return;
              next[i] = {
                ...img,
                mlAssessment: {
                  assessedAt: new Date().toISOString(),
                  predictedLbClass: data.predicted_lb_class,
                  probabilityLb1: data.probability_lb_1,
                  probabilities: data.probabilities,
                },
              };
            } catch (e) {
              if (cancelled) return;
              next[i] = {
                ...img,
                mlAssessment: {
                  assessedAt: new Date().toISOString(),
                  error: e instanceof Error ? e.message : 'Prediction request failed',
                },
              };
            }
          })
        );
        if (!cancelled) {
          updateCase(id, {
            embryoImages: next,
            mlAssessment: firstEmbryoMlForCaseRecord(next, case_.mlAssessment),
          });
        }
      } finally {
        if (!cancelled) setMlLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [id, mlBase, case_, pendingMlCount, updateCase]);

  if (!case_) {
    return (
      <div className={styles.page}>
        <p>Case not found.</p>
        <Link to="/cases">← Back to cases</Link>
      </div>
    );
  }

  const isVerified = case_.status === 'verified';
  const isFlagged = case_.status === 'flagged';
  const firstCaptureDay = embryoGallery[0]?.captureDay;
  const okLb = listOkEmbryoMls(case_);
  const singleLb = okLb.length === 1 ? okLb[0] : undefined;
  const mlErrors = (case_.embryoImages ?? [])
    .map((img, i) => {
      const m = getEmbryoMl(case_, i);
      return m && 'error' in m ? { embryoNumber: img.embryoNumber, message: m.error } : null;
    })
    .filter((x): x is { embryoNumber: number; message: string } => x != null);

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <Link to="/cases" className={styles.back}>← Cases</Link>
        <h1>{case_.caseId}</h1>
        <p className={styles.subtitle}>
          Patient ref: {case_.patientRef} · Cycle: {case_.cycleDate}
        </p>
        <span className={`${styles.statusBadge} ${styles[case_.status]}`}>
          {statusLabels[case_.status]}
        </span>
      </header>

      <div className={styles.grid}>
        <section className={styles.section}>
          <h2>Result summary</h2>
          <dl className={styles.dl}>
            <dt>Embryos assessed</dt>
            <dd>{case_.embryoCount}</dd>
            <dt>Selected for transfer</dt>
            <dd>{case_.selectedCount}</dd>
            <dt>Primary grade</dt>
            <dd><strong>{case_.primaryGrade}</strong></dd>
            <dt>Secondary grade</dt>
            <dd><strong>{case_.secondaryGrade}</strong></dd>
            {embryoGallery.length > 0 && (
              <>
                <dt>Development day (capture)</dt>
                <dd>
                  {embryoGallery.length === 1 && firstCaptureDay
                    ? firstCaptureDay
                    : `${embryoGallery.length} embryos — see gallery`}
                </dd>
                <dt>Live birth (LB)</dt>
                <dd>
                  {singleLb && (
                    <>
                      <strong>P(LB=1) {(singleLb.ml.probabilityLb1 * 100).toFixed(1)}%</strong>
                      <span className={styles.lbSummaryMuted}>
                        {' · '}P(LB=0) {(singleLb.ml.probabilities[0] * 100).toFixed(1)}% · predicted class{' '}
                        {singleLb.ml.predictedLbClass}
                        {' · '}
                        <span className={styles.lbSummaryMuted}>embryo #{singleLb.embryoNumber}</span>
                      </span>
                    </>
                  )}
                  {okLb.length > 1 && (
                    <span className={styles.lbSummaryMuted}>
                      {okLb.map(({ embryoNumber, ml }, j) => (
                        <span key={embryoNumber}>
                          #{embryoNumber}: P(LB=1) {(ml.probabilityLb1 * 100).toFixed(1)}%
                          {j < okLb.length - 1 ? ' · ' : ''}
                        </span>
                      ))}
                      {' — '}
                      full breakdown below.
                    </span>
                  )}
                  {okLb.length === 0 && mlErrors.length > 0 && (
                    <span className={styles.mlErrorInline}>
                      Unavailable —{' '}
                      {mlErrors.map((e) => `#${e.embryoNumber}: ${e.message}`).join('; ')}
                    </span>
                  )}
                  {mlBase && pendingMlCount > 0 && (
                    <span className={styles.lbSummaryMuted}>
                      {mlLoading
                        ? `Computing P(LB=0) and P(LB=1) for ${pendingMlCount} image(s)…`
                        : 'Starting model…'}
                    </span>
                  )}
                  {mlBase &&
                    okLb.length === 0 &&
                    mlErrors.length === 0 &&
                    pendingMlCount === 0 &&
                    embryoGallery.length > 0 && (
                      <span className={styles.lbSummaryMuted}>No LB scores stored for this case yet.</span>
                    )}
                  {!mlBase && (
                    <span className={styles.lbSummaryMuted}>
                      Not computed — the predict API is not configured. Use the checklist under{' '}
                      <strong>Live birth (LB) model</strong> below.
                    </span>
                  )}
                </dd>
              </>
            )}
            {case_.patientAge != null && (
              <>
                <dt>Patient age</dt>
                <dd>{case_.patientAge}</dd>
              </>
            )}
            {case_.retrievalDate && (
              <>
                <dt>Retrieval date</dt>
                <dd>{case_.retrievalDate}</dd>
              </>
            )}
            {case_.fertilizationMethod && (
              <>
                <dt>Fertilization</dt>
                <dd>{case_.fertilizationMethod}</dd>
              </>
            )}
          </dl>
          {embryoGallery.length > 0 && (
            <>
              <p className={styles.gradeHint}>
                Gardner-style grades (A–D) are not inferred from pixels in this build; enter them when adding a full
                patient record. Live birth (LB) scores come from the model below when the API is configured.
              </p>
              <div className={styles.mlPanel}>
                <p className={styles.mlScoresTitle}>Live birth (LB) model</p>
                {okLb.length > 0 && (
                  <p className={styles.mlMuted}>
                    Scores are computed per image. If different embryos still show almost the same P(LB=1), the bundled
                    checkpoint may be weak or mostly reflecting class balance in training—not reliable image-specific
                    discrimination until you re-train or use a stronger model.
                  </p>
                )}
                {!mlBase && (
                  <div className={styles.mlSetup}>
                    <p className={styles.mlMuted}>
                      The app calls a small Python service for <strong>P(LB=0)</strong> and <strong>P(LB=1)</strong>{' '}
                      (PyTorch does not run in the browser here).
                    </p>
                    <details className={styles.mlSetupDetails}>
                      <summary className={styles.mlSetupSummary}>Enable the API (3 steps)</summary>
                      <ol className={styles.mlSetupList}>
                        <li>
                          <code className={styles.codeInline}>cd blastocyst_ml</code> then{' '}
                          <code className={styles.codeInline}>
                            uvicorn serve_predict:app --host 127.0.0.1 --port 8000
                          </code>
                        </li>
                        <li>
                          Project root: create <code className={styles.codeInline}>.env.local</code> with{' '}
                          <code className={styles.codeInline}>VITE_ML_API_URL=http://127.0.0.1:8000</code>
                        </li>
                        <li>
                          Restart <code className={styles.codeInline}>npm run dev</code> and reload this case.
                        </li>
                      </ol>
                    </details>
                  </div>
                )}
                {mlBase && mlLoading && pendingMlCount > 0 && (
                  <p className={styles.mlMuted}>
                    Running live-birth model for {pendingMlCount} image(s)…
                  </p>
                )}
                {mlErrors.length > 0 && (
                  <div className={styles.mlErrorList}>
                    {mlErrors.map((e) => (
                      <p key={e.embryoNumber} className={styles.mlError}>
                        Embryo #{e.embryoNumber}: {e.message}
                      </p>
                    ))}
                  </div>
                )}
                {okLb.map(({ embryoNumber, ml }) => (
                  <MlLbScores
                    key={embryoNumber}
                    title={`Embryo #${embryoNumber}`}
                    ml={ml}
                    patientAge={case_.patientAge}
                  />
                ))}
              </div>
            </>
          )}
        </section>

        <section className={styles.section}>
          <h2>Assessment trail</h2>
          <div className={styles.trail}>
            {case_.firstReviewer && (
              <div className={styles.trailItem}>
                <span className={styles.trailLabel}>Clinical review</span>
                <span>{case_.firstReviewer}</span>
              </div>
            )}
            {case_.secondReviewer && (
              <div className={styles.trailItem}>
                <span className={styles.trailLabel}>QA sign-off</span>
                <span>{case_.secondReviewer}</span>
              </div>
            )}
            {case_.verifiedAt && (
              <div className={styles.trailItem}>
                <span className={styles.trailLabel}>Result locked at</span>
                <span>{new Date(case_.verifiedAt).toLocaleString()}</span>
              </div>
            )}
            {!case_.firstReviewer && (
              <p className={styles.muted}>System assessment only; optional clinical review when flagged.</p>
            )}
          </div>
        </section>
      </div>

      {case_.notes && (
        <section className={styles.section}>
          <h2>Notes</h2>
          <p className={styles.notes}>{case_.notes}</p>
        </section>
      )}

      {isFlagged && case_.discrepancyReason && (
        <section className={styles.section}>
          <h2>Flagged for QA review</h2>
          <div className={styles.flaggedBox}>
            {case_.discrepancyReason}
          </div>
        </section>
      )}

      {case_.embryoImages && case_.embryoImages.length > 0 && (
        <section className={styles.section}>
          <h2>Embryo images (fertility assessment)</h2>
          <div className={styles.imageGrid}>
            {case_.embryoImages.map((img, idx) => {
              const em = getEmbryoMl(case_, idx);
              return (
                <div key={img.id} className={styles.imageCard}>
                  <img src={img.dataUrl} alt={`Embryo ${img.embryoNumber}`} className={styles.embryoImg} />
                  <div className={styles.imageMeta}>
                    <span>Embryo #{img.embryoNumber}</span>
                    <span>{img.captureDay}</span>
                    {img.magnification && <span>{img.magnification}</span>}
                    {img.initialGrade && <span>Grade: {img.initialGrade}</span>}
                    {img.blastocystStage && <span>Stage: {img.blastocystStage}</span>}
                    {em && !('error' in em) && (
                      <span className={styles.embryoMlBadge}>
                        P(LB=1) {(em.probabilityLb1 * 100).toFixed(1)}%
                      </span>
                    )}
                    {em && 'error' in em && (
                      <span className={styles.embryoMlErr}>LB model unavailable</span>
                    )}
                    <span className={styles.captureTime}>
                      {new Date(img.capturedAt).toLocaleString()}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </section>
      )}

      {isVerified && (
        <div className={styles.locked}>
          Result locked. No further edits. See <Link to="/audit">Audit log</Link> for full history.
        </div>
      )}
    </div>
  );
}
