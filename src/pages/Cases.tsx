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

const STATUS_FILTERS = ['pending_review', 'flagged', 'verified'] as const;
type StatusFilter = (typeof STATUS_FILTERS)[number];

export function Cases() {
  const { cases, clearAllCases } = useCases();
  const [searchParams] = useSearchParams();
  const statusFromUrl = searchParams.get('status');
  const [filter, setFilter] = useState<StatusFilter>(
    STATUS_FILTERS.includes(statusFromUrl as StatusFilter) ? (statusFromUrl as StatusFilter) : 'pending_review'
  );

  const filtered = useMemo(() => cases.filter((c) => c.status === filter), [filter, cases]);

  const handleClearAll = () => {
    if (cases.length === 0) return;
    const ok = window.confirm(
      `Delete all ${cases.length} patient record${cases.length === 1 ? '' : 's'}? This cannot be undone.`
    );
    if (ok) clearAllCases();
  };

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <h1>Cases & verification</h1>
        <p className={styles.subtitle}>Embryo cases and system assessment results</p>
        <div className={styles.headerActions}>
          <Link to="/add-patient" className={styles.addLink}>+ Add patient record</Link>
          <button type="button" className={styles.clearAllBtn} onClick={handleClearAll}>
            Clear all records
          </button>
        </div>
      </header>

      <div className={styles.toolbar}>
        <div className={styles.filters}>
          {STATUS_FILTERS.map((s) => (
            <button
              key={s}
              className={filter === s ? `${styles.filterBtn} ${styles.active}` : styles.filterBtn}
              onClick={() => setFilter(s)}
            >
              {statusLabels[s]}
            </button>
          ))}
        </div>
      </div>

      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>Image</th>
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
      <td>
        {row.embryoImages?.[0] ? (
          <img src={row.embryoImages[0].dataUrl} alt={`Embryo for ${row.caseId}`} className={styles.thumb} />
        ) : (
          <span className={styles.thumbEmpty} />
        )}
      </td>
      <td><code className={styles.code}>{row.caseId}</code></td>
      <td>{row.patientRef}</td>
      <td>{row.cycleDate}</td>
      <td>{row.embryoCount}</td>
      <td>{row.selectedCount}</td>
      <td><span className={styles.grade}>{gradeStr}</span></td>
      <td><StatusBadge status={row.status} /></td>
      <td>
        <Link to={`/cases/${row.id}`} className={row.status === 'pending_review' ? `${styles.detailLink} ${styles.reviewLink}` : styles.detailLink}>
          {row.status === 'pending_review' ? 'Review' : 'View'}
        </Link>
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
