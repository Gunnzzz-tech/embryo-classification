import { useState, useMemo } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { useCases } from '../context/CasesContext';
import type { EmbryoCase, VerificationStatus } from '../types';
import styles from './Cases.module.css';

const statusLabels: Record<VerificationStatus, string> = {
  pending_review: 'Pending assessment',
  first_review: 'Clinical review',
  second_review: 'QA sign-off',
  verified: 'Verified',
  flagged: 'Flagged',
  discrepancy: 'Discrepancy',
};

export function Cases() {
  const { cases } = useCases();
  const [searchParams] = useSearchParams();
  const filterFromUrl = searchParams.get('status') ?? 'all';
  const [filter, setFilter] = useState<string>(filterFromUrl);

  const filtered = useMemo(() => {
    if (filter === 'all') return cases;
    return cases.filter((c) => c.status === filter);
  }, [filter, cases]);

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <h1>Cases & verification</h1>
        <p className={styles.subtitle}>Embryo cases and system assessment results</p>
        <Link to="/add-patient" className={styles.addLink}>+ Add patient record</Link>
      </header>

      <div className={styles.toolbar}>
        <div className={styles.filters}>
          {['all', 'pending_review', 'first_review', 'second_review', 'verified', 'flagged'].map((s) => (
            <button
              key={s}
              className={filter === s ? `${styles.filterBtn} ${styles.active}` : styles.filterBtn}
              onClick={() => setFilter(s)}
            >
              {s === 'all' ? 'All' : statusLabels[s as VerificationStatus]}
            </button>
          ))}
        </div>
      </div>

      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>Case ID</th>
              <th>Patient ref</th>
              <th>Cycle date</th>
              <th>Embryos</th>
              <th>Selected</th>
              <th>Grade</th>
              <th>Status</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((row) => (
              <CaseRow key={row.id} row={row} />
            ))}
          </tbody>
        </table>
      </div>

      {filtered.length === 0 && (
        <p className={styles.empty}>No cases match the selected filter.</p>
      )}
    </div>
  );
}

function CaseRow({ row }: { row: EmbryoCase }) {
  const gradeStr = `${row.primaryGrade}/${row.secondaryGrade}`;
  return (
    <tr>
      <td><code className={styles.code}>{row.caseId}</code></td>
      <td>{row.patientRef}</td>
      <td>{row.cycleDate}</td>
      <td>{row.embryoCount}</td>
      <td>{row.selectedCount}</td>
      <td><span className={styles.grade}>{gradeStr}</span></td>
      <td><StatusBadge status={row.status} /></td>
      <td>
        <Link to={`/cases/${row.id}`} className={styles.detailLink}>View</Link>
      </td>
    </tr>
  );
}

function StatusBadge({ status }: { status: VerificationStatus }) {
  const cls = status === 'verified' ? styles.badgeVerified
    : status === 'flagged' ? styles.badgeFlagged
    : status === 'pending_review' ? styles.badgePending
    : styles.badgeOther;
  return <span className={`${styles.badge} ${cls}`}>{statusLabels[status]}</span>;
}
