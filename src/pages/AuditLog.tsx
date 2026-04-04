import { mockAuditLog } from '../mockData';
import type { AuditEntry } from '../types';
import styles from './AuditLog.module.css';

const actionLabels: Record<AuditEntry['action'], string> = {
  result_entered: 'Result entered',
  first_review: 'First review',
  second_review: 'Second review',
  verified: 'Verified',
  flagged: 'Flagged',
  corrected: 'Corrected',
  protocol_change: 'Protocol change',
};

export function AuditLog() {
  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <h1>Audit log</h1>
        <p className={styles.subtitle}>Immutable record of all result-related actions for true results accountability</p>
      </header>

      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>Time (UTC)</th>
              <th>User</th>
              <th>Action</th>
              <th>Case ID</th>
              <th>Details</th>
              <th>Changes</th>
            </tr>
          </thead>
          <tbody>
            {mockAuditLog.map((entry) => (
              <tr key={entry.id}>
                <td className={styles.mono}>{new Date(entry.timestamp).toLocaleString()}</td>
                <td><code className={styles.code}>{entry.userId}</code></td>
                <td><span className={styles.action}>{actionLabels[entry.action]}</span></td>
                <td><code className={styles.code}>{entry.caseId}</code></td>
                <td>{entry.details}</td>
                <td>
                  {entry.previousValue != null && (
                    <span className={styles.change}>
                      {entry.previousValue} → {entry.newValue ?? '—'}
                    </span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
