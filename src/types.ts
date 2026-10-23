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

/** Grad-CAM heatmap overlays (data: URLs, ready for an <img src>) explaining an EXP/ICM/TE score. */
export interface EmbryoMlHeatmaps {
  exp: string;
  icm: string;
  te: string;
}

export type InfertilityCause =
  | 'unexplained' | 'male_factor' | 'tubal' | 'ovulatory' | 'endometriosis' | 'diminished_reserve' | 'uterine' | 'other';

/** What doctors weigh alongside the image. Used only for the transfer-priority score and the explanations; the image model never sees it. */
export interface ClinicalInputs {
  /** Age of the egg source at retrieval, years (required) */
  femaleAge: number;
  /** AMH, ng/mL */
  amh?: number;
  /** Basal (day 2-3) FSH, mIU/mL */
  fsh?: number;
  /** Body-mass index, kg/m2 */
  bmi?: number;
  /** Previous completed IVF/ICSI cycles */
  priorCycles?: number;
  /** Previous embryo transfers that did not result in a pregnancy */
  failedTransfers?: number;
  infertilityCause?: InfertilityCause;
}

/** Where a Grad-CAM map concentrates (numbers computed by the ML API). */
export interface HeadFocus {
  central_fraction: number;
  peripheral_fraction: number;
  spread: number;
  centroid_offset: number;
  pattern: 'central' | 'peripheral' | 'mixed' | 'diffuse';
}

/** Result of optional local ML API (Gardner grading: EXP/ICM/TE, not live-birth outcome). */
export type EmbryoMlAssessment =
  | {
      assessedAt: string;
      predictedExp: number;
      predictedIcm: number;
      predictedTe: number;
      probabilitiesExp: number[];
      probabilitiesIcm: number[];
      probabilitiesTe: number[];
      /** Grad-CAM overlays showing which region of the image drove each score. Absent for older
       * stored assessments computed before heatmaps were added, or if the API call for them failed. */
      heatmaps?: EmbryoMlHeatmaps;
      /** The model's real, un-folded probabilities (ICM/TE keep their 4th "not assessable" class). The explanation
       * and chat use these, never the displayed (possibly demo-scripted) numbers. */
      rawProbabilities?: { exp: number[]; icm: number[]; te: number[] };
      /** Where each head's heatmap concentrates. */
      focus?: { exp?: HeadFocus; icm?: HeadFocus; te?: HeadFocus };
      /** True when the displayed confidence numbers were rewritten by the upload demo override. */
      demoScripted?: boolean;
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
  /** Clinical context entered by the clinician (age, AMH, FSH, BMI, history, cause) */
  clinical?: ClinicalInputs;
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
