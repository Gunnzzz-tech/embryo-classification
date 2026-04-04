import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useCases } from '../context/CasesContext';
import { mockQualityMetrics } from '../mockData';
import styles from './Dashboard.module.css';

function AnimatedCounter({ target, suffix = '' }: { target: number; suffix?: string }) {
  const [count, setCount] = useState(0);

  useEffect(() => {
    if (target === 0) { setCount(0); return; }
    const duration = 1200;
    const steps = 30;
    const increment = target / steps;
    let current = 0;
    const timer = setInterval(() => {
      current += increment;
      if (current >= target) {
        setCount(target);
        clearInterval(timer);
      } else {
        setCount(Math.floor(current));
      }
    }, duration / steps);
    return () => clearInterval(timer);
  }, [target]);

  return <>{count}{suffix}</>;
}

export function Dashboard() {
  const { cases } = useCases();
  const pending = cases.filter((c) => c.status === 'pending_review' || c.status === 'first_review').length;
  const verifiedToday = cases.filter((c) => c.status === 'verified').length;
  const flagged = cases.filter((c) => c.status === 'flagged').length;
  const passRate = mockQualityMetrics.filter((m) => m.status === 'pass').length / mockQualityMetrics.length * 100;

  const [visible, setVisible] = useState(false);
  useEffect(() => { setVisible(true); }, []);

  return (
    <div className={`${styles.page} ${visible ? styles.visible : ''}`}>
      <header className={styles.header}>
        <div className={styles.headerContent}>
          <div className={styles.headerBadge}>Overview</div>
          <h1>Dashboard</h1>
          <p className={styles.subtitle}>Verification status and quality overview</p>
        </div>
        <div className={styles.headerDecor}>
          <div className={styles.orb1} />
          <div className={styles.orb2} />
        </div>
      </header>

      <section className={styles.cards}>
        <div className={`${styles.card} ${styles.cardAccent}`} style={{ animationDelay: '0.05s' }}>
          <div className={styles.cardIcon}>⏳</div>
          <div className={styles.cardBody}>
            <span className={styles.cardLabel}>Pending verification</span>
            <span className={styles.cardValue}><AnimatedCounter target={pending} /></span>
          </div>
          <Link to="/cases?status=pending_review" className={styles.cardLink}>
            Review cases <span className={styles.arrow}>→</span>
          </Link>
        </div>
        <div className={`${styles.card} ${styles.cardSuccess}`} style={{ animationDelay: '0.1s' }}>
          <div className={styles.cardIcon}>✓</div>
          <div className={styles.cardBody}>
            <span className={styles.cardLabel}>Verified (locked)</span>
            <span className={styles.cardValue}><AnimatedCounter target={verifiedToday} /></span>
          </div>
          <span className={styles.cardHint}>System assessment complete</span>
        </div>
        <div className={`${styles.card} ${styles.cardWarning}`} style={{ animationDelay: '0.15s' }}>
          <div className={styles.cardIcon}>⚑</div>
          <div className={styles.cardBody}>
            <span className={styles.cardLabel}>Flagged for QA</span>
            <span className={styles.cardValue}><AnimatedCounter target={flagged} /></span>
          </div>
          {flagged > 0 && (
            <Link to="/cases?status=flagged" className={styles.cardLink}>
              Review <span className={styles.arrow}>→</span>
            </Link>
          )}
        </div>
        <div className={`${styles.card} ${styles.cardPurple}`} style={{ animationDelay: '0.2s' }}>
          <div className={styles.cardIcon}>◈</div>
          <div className={styles.cardBody}>
            <span className={styles.cardLabel}>Quality metrics pass</span>
            <span className={styles.cardValue}><AnimatedCounter target={Math.round(passRate)} suffix="%" /></span>
          </div>
          <span className={styles.cardHint}>All protocols</span>
        </div>
      </section>

      <section className={styles.section} style={{ animationDelay: '0.25s' }}>
        <h2>True results framework</h2>
        <div className={styles.framework}>
          {[
            { num: 1, title: 'Image & data input', desc: 'Embryo images and cycle data are captured for each case. The system uses this to assess fertility—no dual human grading.', color: 'var(--accent)' },
            { num: 2, title: 'ML fertility assessment', desc: 'The system (machine learning, later stage) predicts whether the embryo is fertile from the image. One system result; cases needing human check are flagged for QA.', color: 'var(--teal)' },
            { num: 3, title: 'Audit trail', desc: 'Every action is logged with user, timestamp, and before/after values so results are traceable and accountable.', color: 'var(--purple)' },
            { num: 4, title: 'Result lock', desc: 'Once the system has produced an assessment, the result is locked. Optional clinical review only when the system flags a case.', color: 'var(--success)' },
          ].map((step, i) => (
            <div key={step.num} className={styles.step} style={{ animationDelay: `${0.3 + i * 0.08}s` }}>
              <span className={styles.stepNum} style={{ background: `color-mix(in srgb, ${step.color} 20%, transparent)`, color: step.color }}>
                {step.num}
              </span>
              <div className={styles.connector} />
              <strong>{step.title}</strong>
              <p>{step.desc}</p>
            </div>
          ))}
        </div>
      </section>

      <section className={styles.section} style={{ animationDelay: '0.35s' }}>
        <h2>Quality metrics</h2>
        <div className={styles.metrics}>
          {mockQualityMetrics.map((m, i) => (
            <div key={m.id} className={`${styles.metric} ${styles[m.status]}`} style={{ animationDelay: `${0.4 + i * 0.06}s` }}>
              <span className={styles.metricName}>{m.name}</span>
              <span className={styles.metricValue}>{m.value}{m.unit ?? ''}</span>
              <span className={styles.metricStatus}>{m.status}</span>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
