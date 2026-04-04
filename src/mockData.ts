import type { EmbryoCase, AuditEntry, QualityMetric, ProtocolCheck } from './types';

export const mockCases: EmbryoCase[] = [
  {
    id: '1',
    caseId: 'EF-2025-0842',
    patientRef: 'PT-****4521',
    cycleDate: '2025-03-01',
    embryoCount: 8,
    selectedCount: 2,
    primaryGrade: 'A',
    secondaryGrade: 'A',
    status: 'verified',
    firstReviewer: 'Dr. S. Chen',
    secondReviewer: 'Dr. M. Patel',
    verifiedAt: '2025-03-02T14:32:00Z',
  },
  {
    id: '2',
    caseId: 'EF-2025-0843',
    patientRef: 'PT-****7832',
    cycleDate: '2025-03-02',
    embryoCount: 5,
    selectedCount: 1,
    primaryGrade: 'A',
    secondaryGrade: 'B',
    status: 'second_review',
    firstReviewer: 'Dr. S. Chen',
    secondReviewer: undefined,
    notes: 'Borderline morphology; second opinion requested.',
  },
  {
    id: '3',
    caseId: 'EF-2025-0844',
    patientRef: 'PT-****1190',
    cycleDate: '2025-03-02',
    embryoCount: 6,
    selectedCount: 0,
    primaryGrade: 'C',
    secondaryGrade: 'C',
    status: 'pending_review',
  },
  {
    id: '4',
    caseId: 'EF-2025-0841',
    patientRef: 'PT-****3345',
    cycleDate: '2025-02-28',
    embryoCount: 4,
    selectedCount: 1,
    primaryGrade: 'B',
    secondaryGrade: 'A',
    status: 'flagged',
    firstReviewer: 'Dr. M. Patel',
    discrepancyReason: 'Grade change between primary and secondary assessment; QA review required.',
  },
  {
    id: '5',
    caseId: 'EF-2025-0845',
    patientRef: 'PT-****6612',
    cycleDate: '2025-03-03',
    embryoCount: 7,
    selectedCount: 2,
    primaryGrade: 'A',
    secondaryGrade: 'A',
    status: 'first_review',
    firstReviewer: 'Dr. S. Chen',
  },
];

export const mockAuditLog: AuditEntry[] = [
  { id: 'a1', timestamp: '2025-03-03T10:15:00Z', userId: 'lab-tech-02', action: 'result_entered', caseId: 'EF-2025-0845', details: 'Initial embryo grading entered' },
  { id: 'a2', timestamp: '2025-03-03T09:42:00Z', userId: 'dr-chen', action: 'first_review', caseId: 'EF-2025-0843', details: 'First review completed', newValue: 'A/B' },
  { id: 'a3', timestamp: '2025-03-02T16:00:00Z', userId: 'dr-patel', action: 'verified', caseId: 'EF-2025-0842', details: 'System result verified; result locked.' },
  { id: 'a4', timestamp: '2025-03-02T15:58:00Z', userId: 'dr-chen', action: 'second_review', caseId: 'EF-2025-0842', details: 'Second review concurred with primary.' },
  { id: 'a5', timestamp: '2025-03-02T14:20:00Z', userId: 'qa-system', action: 'flagged', caseId: 'EF-2025-0841', details: 'Grade discrepancy detected', previousValue: 'B/B', newValue: 'B/A' },
];

export const mockQualityMetrics: QualityMetric[] = [
  { id: 'q1', name: 'Inter-observer concordance', value: 96.2, unit: '%', status: 'pass', lastChecked: '2025-03-01' },
  { id: 'q2', name: 'Blastocyst culture rate', value: 62, unit: '%', status: 'pass', lastChecked: '2025-03-02' },
  { id: 'q3', name: 'Equipment calibration', value: 'Within range', status: 'pass', lastChecked: '2025-03-03' },
  { id: 'q4', name: 'Re-read discrepancy rate', value: 2.1, unit: '%', status: 'warn', lastChecked: '2025-03-02' },
];

export const mockProtocolChecks: ProtocolCheck[] = [
  { id: 'p1', name: 'ML fertility model (image-based assessment)', required: true, completed: true, completedBy: 'System', completedAt: '2025-03-03' },
  { id: 'p2', name: 'Clinical review when flagged', required: true, completed: true, completedBy: 'Dr. Chen / Dr. Patel', completedAt: '2025-03-03' },
  { id: 'p3', name: 'Consent and ID verification', required: true, completed: true, completedBy: 'Nurse J. Lee', completedAt: '2025-03-02' },
  { id: 'p4', name: 'Chain of custody log', required: true, completed: true, completedBy: 'Lab', completedAt: '2025-03-03' },
];
