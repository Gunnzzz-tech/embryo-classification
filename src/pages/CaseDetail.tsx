import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { assessmentFromHeatmap, dataUrlToBlob, gradeText, mlApiBase, predictGardnerGradesWithHeatmap } from '../api/mlPredict';
import { InsightsPanel } from '../components/InsightsPanel';
import { useAuth } from '../context/AuthContext';
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

type MlAssessmentOk = Extract<EmbryoMlAssessment, { predictedExp: number }>;

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

const AXIS_INFO: Record<'EXP' | 'ICM' | 'TE', string> = {
  EXP: 'Expansion — how developed the fluid-filled cavity looks',
  ICM: 'Inner cell mass — the cells that would go on to form the fetus',
  TE: 'Trophectoderm — the outer cell layer that would go on to form the placenta',
};

export function CaseDetail() {
  const { id } = useParams<{ id: string }>();
  const { getCaseById, updateCase } = useCases();
  const case_ = id ? getCaseById(id) : undefined;
  const { auth } = useAuth();
  const [mlLoading, setMlLoading] = useState(false);
  const [rejecting, setRejecting] = useState(false);
  const [rejectReason, setRejectReason] = useState('');
  const [rejectError, setRejectError] = useState<string | null>(null);

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
              const data = await predictGardnerGradesWithHeatmap(mlBase, blob);
              if (cancelled) return;
              next[i] = {
                ...img,
                mlAssessment: assessmentFromHeatmap(data),
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

  const isAdmin = auth.signedIn && auth.role === 'admin';
  const reviewerName = auth.signedIn ? `Dr. ${auth.doctorId}` : 'Unknown';

  const approve = () => {
    const ok = window.confirm(`Approve this assessment for ${case_.caseId}? The result will be locked.`);
    if (!ok) return;
    updateCase(case_.id, {
      status: 'verified',
      firstReviewer: reviewerName,
      verifiedAt: new Date().toISOString(),
      discrepancyReason: undefined,
    });
    setRejecting(false);
  };

  const reject = () => {
    const reason = rejectReason.trim();
    if (!reason) {
      setRejectError('Please write a short reason so the team knows what to recheck.');
      return;
    }
    updateCase(case_.id, {
      status: 'flagged',
      firstReviewer: reviewerName,
      discrepancyReason: reason,
      verifiedAt: undefined,
    });
    setRejecting(false);
    setRejectReason('');
    setRejectError(null);
  };

  const isVerified = case_.status === 'verified';
  const isFlagged = case_.status === 'flagged';
  const okMl = listOkEmbryoMls(case_);
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

      {embryoGallery.length > 0 && (
        <section className={styles.section}>
          <h2>Where the model looked (Grad-CAM)</h2>
          {!mlBase && (
            <div className={styles.mlSetup}>
              <p className={styles.mlMuted}>
                The app calls a small Python service for the model (PyTorch does not run in the browser here).
              </p>
              <details className={styles.mlSetupDetails}>
                <summary className={styles.mlSetupSummary}>Enable the API (3 steps)</summary>
                <ol className={styles.mlSetupList}>
                  <li>
                    <code className={styles.codeInline}>cd blastocyst_ml</code> then{' '}
                    <code className={styles.codeInline}>uvicorn serve_predict_gardner:app --host 127.0.0.1 --port 8000</code>
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
          {mlBase && pendingMlCount > 0 && (
            <p className={styles.mlMuted}>
              {mlLoading ? `Running the model for ${pendingMlCount} image(s)…` : 'Starting model…'}
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
          {okMl.map(({ embryoNumber, ml }) =>
            ml.heatmaps ? (
              <div key={embryoNumber} className={styles.bigHeatmapGroup}>
                {okMl.length > 1 && <p className={styles.bigHeatmapTitle}>Embryo #{embryoNumber}</p>}
                <div className={styles.bigHeatmapRow}>
                  {(['EXP', 'ICM', 'TE'] as const).map((label) => (
                    <figure key={label} className={styles.bigHeatmapCard} title={AXIS_INFO[label]}>
                      <img
                        src={ml.heatmaps![label.toLowerCase() as 'exp' | 'icm' | 'te']}
                        alt={`Grad-CAM heatmap for ${label} score`}
                        className={styles.bigHeatmapImg}
                      />
                      <figcaption className={styles.bigHeatmapLabel}>{label}</figcaption>
                    </figure>
                  ))}
                </div>
              </div>
            ) : null
          )}
          {okMl.length > 0 && (
            <p className={styles.mlMuted}>
              Warmer colours mark the parts of the image that pushed the model most toward its answer. The maps are
              coarse and look alike for all three grades, so they show roughly where the model looked, not exact
              outlines.
            </p>
          )}
          {(case_.firstReviewer || case_.secondReviewer || case_.verifiedAt) && (
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
            </div>
          )}
        </section>
      )}

      {mlBase && okMl.length > 0 && (
        <InsightsPanel
          baseUrl={mlBase}
          embryos={okMl.map(({ embryoNumber, ml }) => ({ embryoNumber, ml }))}
          clinical={case_.clinical}
          onSaveClinical={(clinical) => updateCase(case_.id, { clinical, patientAge: clinical.femaleAge })}
        />
      )}

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
                        EXP {em.predictedExp} · ICM {gradeText('icm', em.predictedIcm)} · TE {gradeText('te', em.predictedTe)}
                      </span>
                    )}
                    {em && 'error' in em && (
                      <span className={styles.embryoMlErr}>Gardner model unavailable</span>
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

      {!isVerified && (
        <section className={styles.section}>
          <h2>Review decision</h2>
          {isAdmin ? (
            <div className={styles.decisionBox}>
              <p className={styles.decisionText}>
                Look at the image, the heatmaps and the transfer priority above, then approve or reject this assessment.
                {case_.status === 'pending_review' ? '' : ` Current status: ${statusLabels[case_.status]}.`}
              </p>
              {!rejecting && (
                <div className={styles.decisionActions}>
                  <button type="button" className={styles.approveBtn} onClick={approve}>
                    Approve
                  </button>
                  <button type="button" className={styles.rejectBtn} onClick={() => setRejecting(true)}>
                    Reject
                  </button>
                </div>
              )}
              {rejecting && (
                <div className={styles.rejectBox}>
                  <label className={styles.rejectLabel} htmlFor="reject-reason">
                    Reason for rejecting
                  </label>
                  <textarea
                    id="reject-reason"
                    className={styles.rejectInput}
                    rows={3}
                    value={rejectReason}
                    onChange={(e) => {
                      setRejectReason(e.target.value);
                      setRejectError(null);
                    }}
                    placeholder="For example: image is not a blastocyst, wrong day, grades need an embryologist's recheck"
                  />
                  {rejectError && <p className={styles.mlError}>{rejectError}</p>}
                  <div className={styles.decisionActions}>
                    <button type="button" className={styles.rejectBtn} onClick={reject}>
                      Confirm reject
                    </button>
                    <button
                      type="button"
                      className={styles.cancelBtn}
                      onClick={() => {
                        setRejecting(false);
                        setRejectError(null);
                      }}
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <p className={styles.muted}>Only an admin can approve or reject an assessment.</p>
          )}
        </section>
      )}

      {isVerified && (
        <div className={styles.locked}>
          Result approved{case_.firstReviewer ? ` by ${case_.firstReviewer}` : ''} and locked. No further edits. See <Link to="/audit">Audit log</Link> for full history.
        </div>
      )}
    </div>
  );
}
