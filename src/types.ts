export type VerificationStatus = 'pending_review' | 'first_review' | 'second_review' | 'verified' | 'flagged' | 'discrepancy';

export type EmbryoGrade = 'A' | 'B' | 'C' | 'D' | 'N/A';

/** Day of development for embryo image (used for fertility assessment from image) */
export type EmbryoDay = 'Day 1' | 'Day 2' | 'Day 3' | 'Day 4' | 'Day 5' | 'Day 6';

/** Image stored as data URL for demo; in production would be server URL */
export interface EmbryoImage {
  id: string;
  /** Data URL or blob URL for display */
  dataUrl: string;
  /** Embryo number within cycle (1-based) */
  embryoNumber: number;
  /** Day of development when image was captured */
  captureDay: EmbryoDay;
  /** Capture timestamp (ISO) */
  capturedAt: string;
  /** Microscope magnification if known (e.g. "400x") */
  magnification?: string;
  /** Initial morphology grade from image assessment */
  initialGrade?: EmbryoGrade;
  /** Blastocyst stage if Day 5/6 (e.g. "3AA", "4AB") */
  blastocystStage?: string;
  /** LB model output for this specific image (case-level mlAssessment may mirror embryo #1 for legacy data) */
  mlAssessment?: EmbryoMlAssessment;
}

/** Result of optional local ML API (live-birth model; not Gardner grading). */
export type EmbryoMlAssessment =
  | {
      assessedAt: string;
      predictedLbClass: 0 | 1;
      /** Model-estimated P(live birth), class 1 */
      probabilityLb1: number;
      probabilities: [number, number];
    }
  | {
      assessedAt: string;
      error: string;
    };

export interface EmbryoCase {
  id: string;
  caseId: string;
  patientRef: string;
  cycleDate: string;
  embryoCount: number;
  selectedCount: number;
  primaryGrade: EmbryoGrade;
  secondaryGrade: EmbryoGrade;
  status: VerificationStatus;
  firstReviewer?: string;
  secondReviewer?: string;
  verifiedAt?: string;
  notes?: string;
  discrepancyReason?: string;
  /** Embryo images for fertility detection by picture */
  embryoImages?: EmbryoImage[];
  /** Patient/cycle fields for record-keeping */
  patientAge?: number;
  retrievalDate?: string;
  fertilizationMethod?: 'IVF' | 'ICSI' | 'IVF+ICSI';
  /** Filled when VITE_ML_API_URL is set and prediction runs on case open */
  mlAssessment?: EmbryoMlAssessment;
}

/** Input for adding a new patient record with embryo image(s) */
export interface PatientRecordInput {
  patientRef: string;
  patientAge?: number;
  cycleDate: string;
  retrievalDate?: string;
  fertilizationMethod?: 'IVF' | 'ICSI' | 'IVF+ICSI';
  embryoCount: number;
  selectedCount: number;
  primaryGrade: EmbryoGrade;
  secondaryGrade: EmbryoGrade;
  notes?: string;
  embryoImages: Array<{
    file: File;
    embryoNumber: number;
    captureDay: EmbryoDay;
    magnification?: string;
    initialGrade?: EmbryoGrade;
    blastocystStage?: string;
  }>;
}

export interface AuditEntry {
  id: string;
  timestamp: string;
  userId: string;
  action: 'result_entered' | 'first_review' | 'second_review' | 'verified' | 'flagged' | 'corrected' | 'protocol_change';
  caseId: string;
  details: string;
  previousValue?: string;
  newValue?: string;
}

export interface QualityMetric {
  id: string;
  name: string;
  value: string | number;
  unit?: string;
  status: 'pass' | 'warn' | 'fail';
  lastChecked: string;
}

export interface ProtocolCheck {
  id: string;
  name: string;
  required: boolean;
  completed: boolean;
  completedBy?: string;
  completedAt?: string;
}
