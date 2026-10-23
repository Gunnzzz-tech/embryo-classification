import { useEffect, useMemo, useState } from 'react';
import { assessCase, explainCase, toApiClinical, toApiModelOutput, type ExplainResponse, type OutlookResponse } from '../api/mlPredict';
import type { ClinicalInputs, EmbryoMlAssessment } from '../types';
import { ChatPanel } from './ChatPanel';
import { ClinicalInputsForm, clinicalToForm, parseClinical, type ClinicalFormValues } from './ClinicalInputsForm';
import styles from './InsightsPanel.module.css';

type OkAssessment = Extract<EmbryoMlAssessment, { predictedExp: number }>;
export type InsightEmbryo = { embryoNumber: number; ml: OkAssessment };

type Props = {
  baseUrl: string;
  embryos: InsightEmbryo[];
  clinical?: ClinicalInputs;
  onSaveClinical: (c: ClinicalInputs) => void;
};

const BAND_CLASS = { High: 'bandHigh', Medium: 'bandMedium', Low: 'bandLow' } as const;
const HEAD_ORDER = ['exp', 'icm', 'te'] as const;

export function InsightsPanel({ baseUrl, embryos, clinical, onSaveClinical }: Props) {
  const [selected, setSelected] = useState<number>(embryos[0]?.embryoNumber ?? 1);
  const [results, setResults] = useState<Record<number, ExplainResponse>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [outlooks, setOutlooks] = useState<Record<number, OutlookResponse>>({});
  const [outlookBusy, setOutlookBusy] = useState(false);
  const [outlookError, setOutlookError] = useState<string | null>(null);
  const [editing, setEditing] = useState(false);
  const [form, setForm] = useState<ClinicalFormValues>(clinicalToForm(clinical));
  const [formErrors, setFormErrors] = useState<ReturnType<typeof parseClinical>['errors']>({});

  // Re-run /explain only when the model output or clinical inputs actually change.
  const signature = useMemo(
    () => JSON.stringify([embryos.map((e) => [e.embryoNumber, toApiModelOutput(e.ml)]), toApiClinical(clinical)]),
    [embryos, clinical]
  );

  useEffect(() => {
    const ctrl = new AbortController();
    setLoading(true);
    setError(null);
    (async () => {
      try {
        const entries = await Promise.all(
          embryos.map(async (e) => [e.embryoNumber, await explainCase(baseUrl, e.ml, clinical, ctrl.signal)] as const)
        );
        if (!ctrl.signal.aborted) setResults(Object.fromEntries(entries));
      } catch (e) {
        if (!ctrl.signal.aborted) setError(e instanceof Error ? e.message : 'Could not load the explanation.');
      } finally {
        if (!ctrl.signal.aborted) setLoading(false);
      }
    })();
    return () => ctrl.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [signature, baseUrl]);

  // An outlook is only valid for the exact grades and patient details it was written from.
  useEffect(() => {
    setOutlooks({});
    setOutlookError(null);
  }, [signature]);

  const generateOutlook = async (e: InsightEmbryo) => {
    setOutlookBusy(true);
    setOutlookError(null);
    try {
      const res = await assessCase(baseUrl, e.ml, clinical);
      setOutlooks((prev) => ({ ...prev, [e.embryoNumber]: res }));
    } catch (err) {
      setOutlookError(err instanceof Error ? err.message : 'Could not create the outlook.');
    } finally {
      setOutlookBusy(false);
    }
  };

  useEffect(() => {
    if (!embryos.some((e) => e.embryoNumber === selected) && embryos[0]) setSelected(embryos[0].embryoNumber);
  }, [embryos, selected]);

  const current = embryos.find((e) => e.embryoNumber === selected) ?? embryos[0];
  const data = current ? results[current.embryoNumber] : undefined;
  const ranking = useMemo(
    () =>
      embryos
        .flatMap((e) => {
          const r = results[e.embryoNumber];
          return r ? [{ n: e.embryoNumber, p: r.priority }] : [];
        })
        .sort((a, b) => b.p.score - a.p.score),
    [embryos, results]
  );
  const outlook = current ? outlooks[current.embryoNumber] : undefined;
  const scripted = embryos.some((e) => e.ml.demoScripted);

  const saveClinical = () => {
    const { value, errors } = parseClinical(form);
    setFormErrors(errors);
    if (!value) return;
    onSaveClinical(value);
    setEditing(false);
  };

  if (!current) return null;

  return (
    <section className={styles.wrap}>
      <h2 className={styles.h2}>Why this result, and what it means</h2>

      {scripted && (
        <div className={styles.demoNote}>
          Demo mode is on: the confidence percentages in the grade tiles above are scripted. The explanations and priority
          score below use the model&apos;s real output.
        </div>
      )}

      <div className={styles.clinicalBar}>
        {clinical ? (
          <p className={styles.clinicalLine}>
            <strong>Clinical inputs:</strong> age {clinical.femaleAge}
            {clinical.amh != null && ` · AMH ${clinical.amh}`}
            {clinical.fsh != null && ` · FSH ${clinical.fsh}`}
            {clinical.bmi != null && ` · BMI ${clinical.bmi}`}
            {clinical.priorCycles != null && ` · ${clinical.priorCycles} prior cycles`}
            {clinical.failedTransfers != null && ` · ${clinical.failedTransfers} failed transfers`}
            {clinical.infertilityCause && ` · ${clinical.infertilityCause.replace('_', ' ')}`}
          </p>
        ) : (
          <p className={styles.clinicalLine}>No clinical inputs for this case. The priority score uses the image only.</p>
        )}
        <button type="button" className={styles.linkBtn} onClick={() => { setForm(clinicalToForm(clinical)); setFormErrors({}); setEditing((v) => !v); }}>
          {editing ? 'Cancel' : clinical ? 'Edit' : 'Add clinical inputs'}
        </button>
      </div>
      {editing && (
        <div className={styles.editor}>
          <ClinicalInputsForm values={form} onChange={setForm} errors={formErrors} />
          <button type="button" className={styles.primaryBtn} onClick={saveClinical}>
            Save and update
          </button>
        </div>
      )}

      {embryos.length > 1 && (
        <div className={styles.rankBox}>
          <p className={styles.rankTitle}>Ranking within this case (heuristic)</p>
          <ol className={styles.rankList}>
            {ranking.map((r) => (
              <li key={r.n}>
                <button type="button" className={`${styles.rankBtn} ${r.n === current.embryoNumber ? styles.rankActive : ''}`} onClick={() => setSelected(r.n)}>
                  Embryo #{r.n}
                </button>
                <span className={`${styles.band} ${styles[BAND_CLASS[r.p.band]]}`}>{r.p.band}</span>
                <span className={styles.rankScore}>{r.p.score}/100</span>
              </li>
            ))}
          </ol>
        </div>
      )}

      {loading && !data && <p className={styles.muted}>Preparing the explanation…</p>}
      {error && (
        <p className={styles.error}>
          Could not load the explanation. Update the ML API (<code>blastocyst_ml/serve_predict_gardner.py</code>) so it has the /explain and /chat endpoints. Details: {error}
        </p>
      )}

      {data && (
        <>
          <div className={styles.outlook}>
            <div className={styles.outlookHead}>
              <div>
                <p className={styles.kicker}>Estimated outlook, embryo #{current.embryoNumber}</p>
                <p className={styles.muted}>Written from the image grades and the patient details above. Not a pregnancy prediction.</p>
              </div>
              {outlook && (
                <span className={`${styles.outlookLabel} ${styles[BAND_CLASS[outlook.band]]}`}>{outlook.outlook}</span>
              )}
            </div>

            {!outlook && (
              <button type="button" className={styles.primaryBtn} onClick={() => generateOutlook(current)} disabled={outlookBusy}>
                {outlookBusy ? 'Writing the outlook…' : 'Generate estimated outlook'}
              </button>
            )}
            {outlookError && <p className={styles.error}>{outlookError}</p>}

            {outlook && (
              <>
                <p className={styles.outlookSummary}>{outlook.summary}</p>
                <div className={styles.outlookCols}>
                  {outlook.strengths.length > 0 && (
                    <div>
                      <p className={styles.flagsTitle}>Looks favourable</p>
                      <ul className={styles.outlookList}>{outlook.strengths.map((t) => <li key={t}>{t}</li>)}</ul>
                    </div>
                  )}
                  {outlook.concerns.length > 0 && (
                    <div>
                      <p className={styles.flagsTitle}>Less favourable or uncertain</p>
                      <ul className={styles.outlookList}>{outlook.concerns.map((t) => <li key={t}>{t}</li>)}</ul>
                    </div>
                  )}
                  {outlook.discuss.length > 0 && (
                    <div>
                      <p className={styles.flagsTitle}>For the care team to discuss</p>
                      <ul className={styles.outlookList}>{outlook.discuss.map((t) => <li key={t}>{t}</li>)}</ul>
                    </div>
                  )}
                </div>
                <p className={styles.muted}>
                  {outlook.source === 'llm' ? `Wording written by the language model (${outlook.model}).` : outlook.notice}{' '}
                  The label comes from the heuristic score ({outlook.score}/100), not from the language model.
                </p>
                <p className={styles.disclaimer}>{outlook.disclaimer}</p>
                <button type="button" className={styles.linkBtn} onClick={() => generateOutlook(current)} disabled={outlookBusy}>
                  {outlookBusy ? 'Rewriting…' : 'Rewrite'}
                </button>
              </>
            )}
          </div>

          <div className={styles.priority}>
            <div className={styles.priorityHead}>
              <div>
                <p className={styles.kicker}>Transfer priority, embryo #{current.embryoNumber} (heuristic)</p>
                <p className={styles.formula}>{data.priority.formula}</p>
              </div>
              <div className={`${styles.scoreBadge} ${styles[BAND_CLASS[data.priority.band]]}`}>
                <span className={styles.scoreNum}>{data.priority.score}</span>
                <span className={styles.scoreBand}>{data.priority.band}</span>
              </div>
            </div>

            <table className={styles.table}>
              <thead>
                <tr><th>Factor</th><th>Value</th><th className={styles.num}>Points</th></tr>
              </thead>
              <tbody>
                {data.priority.morphology.map((p) => (
                  <tr key={p.factor}><td>{p.factor}</td><td>{p.value}</td><td className={styles.num}>{p.points}/{p.max}</td></tr>
                ))}
                {data.priority.clinical.map((p) => (
                  <tr key={p.factor} title={p.reason}><td>{p.factor}</td><td>{p.value}</td><td className={styles.num}>{p.points > 0 ? `+${p.points}` : p.points}</td></tr>
                ))}
              </tbody>
            </table>
            {data.priority.clinical_provided && data.priority.not_provided.length > 0 && (
              <p className={styles.muted}>Not provided, so treated as neutral: {data.priority.not_provided.join(', ')}.</p>
            )}
            {data.priority.low_confidence_penalty > 0 && (
              <p className={styles.muted}>Reduced by {data.priority.low_confidence_penalty} because average model confidence is under 60%.</p>
            )}
            {data.priority.flags.length > 0 && (
              <div className={styles.flags}>
                <p className={styles.flagsTitle}>Points doctors commonly weigh</p>
                <ul>
                  {data.priority.flags.map((f) => <li key={f}>{f}</li>)}
                </ul>
              </div>
            )}
            <p className={styles.disclaimer}>{data.priority.disclaimer}</p>
          </div>

          <div className={styles.heads}>
            {HEAD_ORDER.map((h) => {
              const d = data.heads[h];
              return (
                <article key={h} className={styles.headCard}>
                  <header className={styles.headHeader}>
                    <strong>{d.name}</strong>
                    <span className={`${styles.pill} ${d.not_assessable ? styles.pillNeutral : d.low_confidence ? styles.pillWarn : styles.pillOk}`}>
                      {d.not_assessable ? 'not assessable' : `${Math.round(d.confidence * 100)}% sure`}
                    </span>
                  </header>
                  <p className={styles.explain}>{d.explanation}</p>
                </article>
              );
            })}
          </div>

          <ul className={styles.notes}>
            {data.notes.map((n) => <li key={n}>{n}</li>)}
          </ul>
        </>
      )}

      <ChatPanel baseUrl={baseUrl} ml={current.ml} clinical={clinical} resetKey={`${current.embryoNumber}-${signature.length}`} />
    </section>
  );
}
