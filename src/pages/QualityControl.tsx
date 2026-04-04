import { mockQualityMetrics, mockProtocolChecks } from '../mockData';
import styles from './QualityControl.module.css';

export function QualityControl() {
  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <h1>Quality control</h1>
        <p className={styles.subtitle}>Protocol compliance and metrics to ensure true, reproducible results</p>
      </header>

      <section className={styles.section}>
        <h2>Protocol compliance</h2>
        <ul className={styles.checkList}>
          {mockProtocolChecks.map((p) => (
            <li key={p.id} className={styles.checkItem}>
              <span className={p.completed ? styles.checkDone : styles.checkPending}>
                {p.completed ? '✓' : '○'}
              </span>
              <div>
                <strong>{p.name}</strong>
                {p.required && <span className={styles.required}>Required</span>}
                {p.completed && p.completedBy && (
                  <p className={styles.meta}>Completed by {p.completedBy} · {p.completedAt}</p>
                )}
              </div>
            </li>
          ))}
        </ul>
      </section>

      <section className={styles.section}>
        <h2>Lab metrics</h2>
        <div className={styles.metrics}>
          {mockQualityMetrics.map((m) => (
            <div key={m.id} className={`${styles.metricCard} ${styles[m.status]}`}>
              <span className={styles.metricName}>{m.name}</span>
              <span className={styles.metricValue}>{m.value}{m.unit ?? ''}</span>
              <span className={styles.metricStatus}>{m.status}</span>
              <span className={styles.metricDate}>Last checked: {m.lastChecked}</span>
            </div>
          ))}
        </div>
      </section>

      <section className={styles.section}>
        <h2>True results commitment</h2>
        <p className={styles.commitment}>
          This admin panel is built for ML-based embryo fertility assessment: the system
          (machine learning, later stage) will predict fertility from embryo images—no dual
          human grading. One system result per case; optional clinical or QA review only when
          the system flags a case. Every action is recorded in the audit log for traceability
          and regulatory compliance.
        </p>
      </section>
    </div>
  );
}
